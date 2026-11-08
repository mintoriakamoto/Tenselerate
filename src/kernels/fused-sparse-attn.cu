// fused-sparse-attn.cu – Fused sparse attention kernel for Hermes 1.7B
// Combines: local window + strided sparsity patterns into single kernel
// Target: 40-50% reduction in attention compute vs. dense attention
//
// Sparsity pattern: attend to [local_window + stride_indices]
// Example: seq_len=4096, local_window=64, stride=8
// Each token attends to: local neighbors ([-32, +32]) + every 8th token in document

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <algorithm>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8
#define LOCAL_WINDOW 64   // Local attention window size
#define STRIDE_FACTOR 8   // Attend to every Nth token (strided pattern)

// CUDA softmax with numerical stability
__device__ void softmax_block(float* logits, int seq_len, int tidx) {
    // Find max for numerical stability
    float max_val = -1e9f;
    for (int i = tidx; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, logits[i]);
    }
    __syncthreads();

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tidx; i < seq_len; i += blockDim.x) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    __syncthreads();

    // Normalize
    for (int i = tidx; i < seq_len; i += blockDim.x) {
        logits[i] /= (sum + 1e-9f);
    }
}

// Fused sparse attention kernel: float32
// Computes: out = softmax(Q @ K^T / sqrt(d_k)) @ V
// With sparsity: only attend to local + strided positions
__global__ void fused_sparse_attn_f32(
    const float* __restrict__ Q,          // [batch*seq, n_heads, d_k]
    const float* __restrict__ K,          // [batch*seq, n_heads, d_k]
    const float* __restrict__ V,          // [batch*seq, n_heads, d_v]
    float* __restrict__ out,              // [batch*seq, n_heads, d_v]
    int batch_seq,
    int n_heads,
    int d_k,
    int d_v,
    int local_window_size,
    int stride
) {
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Query position (which token we're computing attention for)
    const int q_pos = bx * BLOCK_SIZE_Y + ty;
    if (q_pos >= batch_seq) return;

    // Head index
    const int head = by;
    if (head >= n_heads) return;

    // Load Q for this position
    __shared__ float Q_shared[32];  // d_k = 32 max
    for (int i = tx; i < d_k; i += BLOCK_SIZE_X) {
        Q_shared[i] = Q[q_pos * n_heads * d_k + head * d_k + i];
    }
    __syncthreads();

    // Compute attention scores for this query
    // Only attend to local + strided positions
    __shared__ float attention_scores[256];  // Max sequence window

    float max_score = -1e9f;
    int n_keys = 0;

    // 1. Local window: [-local_window/2, +local_window/2]
    int local_start = max(0, q_pos - local_window_size / 2);
    int local_end = min(batch_seq, q_pos + local_window_size / 2);

    for (int k_pos = local_start + tx; k_pos < local_end; k_pos += BLOCK_SIZE_X) {
        // Q @ K^T / sqrt(d_k)
        float score = 0.0f;
        for (int i = 0; i < d_k; ++i) {
            float k_val = K[k_pos * n_heads * d_k + head * d_k + i];
            score += Q_shared[i] * k_val;
        }
        score /= sqrtf((float)d_k);

        int score_idx = k_pos - local_start;
        if (score_idx < 256) {
            attention_scores[score_idx] = score;
            max_score = fmaxf(max_score, score);
            n_keys++;
        }
    }
    __syncthreads();

    // 2. Strided pattern: attend to every stride-th token
    for (int k_idx = tx; k_idx < batch_seq; k_idx += stride) {
        if (k_idx < local_start || k_idx >= local_end) {  // Skip already computed local
            float score = 0.0f;
            for (int i = 0; i < d_k; ++i) {
                float k_val = K[k_idx * n_heads * d_k + head * d_k + i];
                score += Q_shared[i] * k_val;
            }
            score /= sqrtf((float)d_k);

            // Store in separate shared memory region (strided scores)
            int strided_idx = k_idx / stride;
            if (strided_idx < 256) {
                attention_scores[128 + strided_idx] = score;
                max_score = fmaxf(max_score, score);
                n_keys++;
            }
        }
    }
    __syncthreads();

    // 3. Softmax normalization
    float sum_exp = 0.0f;
    for (int i = tx; i < n_keys; i += BLOCK_SIZE_X) {
        if (i < 256) {
            attention_scores[i] = expf(attention_scores[i] - max_score);
            sum_exp += attention_scores[i];
        }
    }
    __syncthreads();

    for (int i = tx; i < n_keys; i += BLOCK_SIZE_X) {
        if (i < 256) {
            attention_scores[i] /= (sum_exp + 1e-9f);
        }
    }
    __syncthreads();

    // 4. Weighted sum: attention @ V
    for (int i = tx; i < d_v; i += BLOCK_SIZE_X) {
        float output_val = 0.0f;

        // Local window contribution
        for (int k_pos = local_start; k_pos < local_end; ++k_pos) {
            int score_idx = k_pos - local_start;
            if (score_idx < 256) {
                float attn_weight = attention_scores[score_idx];
                float v_val = V[k_pos * n_heads * d_v + head * d_v + i];
                output_val += attn_weight * v_val;
            }
        }

        // Strided pattern contribution
        for (int k_idx = 0; k_idx < batch_seq; k_idx += stride) {
            if (k_idx >= local_start && k_idx < local_end) continue;  // Skip local

            int strided_idx = k_idx / stride;
            if (strided_idx < 256) {
                float attn_weight = attention_scores[128 + strided_idx];
                float v_val = V[k_idx * n_heads * d_v + head * d_v + i];
                output_val += attn_weight * v_val;
            }
        }

        out[q_pos * n_heads * d_v + head * d_v + i] = output_val;
    }
}

// Fused sparse attention kernel: bfloat16
__global__ void fused_sparse_attn_bf16(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ out,
    int batch_seq,
    int n_heads,
    int d_k,
    int d_v,
    int local_window_size,
    int stride
) {
    // Same logic as float32, but with bf16 conversions
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int q_pos = bx * BLOCK_SIZE_Y + ty;
    if (q_pos >= batch_seq) return;

    const int head = by;
    if (head >= n_heads) return;

    // Load Q
    __shared__ float Q_shared[32];
    for (int i = tx; i < d_k; i += BLOCK_SIZE_X) {
        Q_shared[i] = __bfloat162float(Q[q_pos * n_heads * d_k + head * d_k + i]);
    }
    __syncthreads();

    __shared__ float attention_scores[256];

    float max_score = -1e9f;
    int n_keys = 0;

    int local_start = max(0, q_pos - local_window_size / 2);
    int local_end = min(batch_seq, q_pos + local_window_size / 2);

    // Local attention
    for (int k_pos = local_start + tx; k_pos < local_end; k_pos += BLOCK_SIZE_X) {
        float score = 0.0f;
        for (int i = 0; i < d_k; ++i) {
            float k_val = __bfloat162float(K[k_pos * n_heads * d_k + head * d_k + i]);
            score += Q_shared[i] * k_val;
        }
        score /= sqrtf((float)d_k);

        int score_idx = k_pos - local_start;
        if (score_idx < 256) {
            attention_scores[score_idx] = score;
            max_score = fmaxf(max_score, score);
            n_keys++;
        }
    }
    __syncthreads();

    // Strided attention
    for (int k_idx = tx; k_idx < batch_seq; k_idx += stride) {
        if (k_idx < local_start || k_idx >= local_end) {
            float score = 0.0f;
            for (int i = 0; i < d_k; ++i) {
                float k_val = __bfloat162float(K[k_idx * n_heads * d_k + head * d_k + i]);
                score += Q_shared[i] * k_val;
            }
            score /= sqrtf((float)d_k);

            int strided_idx = k_idx / stride;
            if (strided_idx < 256) {
                attention_scores[128 + strided_idx] = score;
                max_score = fmaxf(max_score, score);
                n_keys++;
            }
        }
    }
    __syncthreads();

    // Softmax
    float sum_exp = 0.0f;
    for (int i = tx; i < n_keys; i += BLOCK_SIZE_X) {
        if (i < 256) {
            attention_scores[i] = expf(attention_scores[i] - max_score);
            sum_exp += attention_scores[i];
        }
    }
    __syncthreads();

    for (int i = tx; i < n_keys; i += BLOCK_SIZE_X) {
        if (i < 256) {
            attention_scores[i] /= (sum_exp + 1e-9f);
        }
    }
    __syncthreads();

    // Output
    for (int i = tx; i < d_v; i += BLOCK_SIZE_X) {
        float output_val = 0.0f;

        for (int k_pos = local_start; k_pos < local_end; ++k_pos) {
            int score_idx = k_pos - local_start;
            if (score_idx < 256) {
                float attn_weight = attention_scores[score_idx];
                float v_val = __bfloat162float(V[k_pos * n_heads * d_v + head * d_v + i]);
                output_val += attn_weight * v_val;
            }
        }

        for (int k_idx = 0; k_idx < batch_seq; k_idx += stride) {
            if (k_idx >= local_start && k_idx < local_end) continue;

            int strided_idx = k_idx / stride;
            if (strided_idx < 256) {
                float attn_weight = attention_scores[128 + strided_idx];
                float v_val = __bfloat162float(V[k_idx * n_heads * d_v + head * d_v + i]);
                output_val += attn_weight * v_val;
            }
        }

        out[q_pos * n_heads * d_v + head * d_v + i] = __float2bfloat16(output_val);
    }
}

// Host wrappers
extern "C" cudaError_t launch_fused_sparse_attn_f32(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int batch_seq,
    int n_heads,
    int d_k,
    int d_v,
    int local_window_size,
    int stride,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((batch_seq + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, n_heads);

    fused_sparse_attn_f32<<<grid, block, 0, stream>>>(
        Q, K, V, out, batch_seq, n_heads, d_k, d_v, local_window_size, stride
    );
    return cudaGetLastError();
}

extern "C" cudaError_t launch_fused_sparse_attn_bf16(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* out,
    int batch_seq,
    int n_heads,
    int d_k,
    int d_v,
    int local_window_size,
    int stride,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((batch_seq + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, n_heads);

    fused_sparse_attn_bf16<<<grid, block, 0, stream>>>(
        Q, K, V, out, batch_seq, n_heads, d_k, d_v, local_window_size, stride
    );
    return cudaGetLastError();
}
