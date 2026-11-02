// fused-ffn.cu – Fused FFN kernel for Hermes 1.7B on RTX 3060 Ti
// Combines: Linear(up) → GELU → Linear(down) into single kernel
// Target: 2-3× throughput vs. separate kernels by eliminating intermediate memory I/O

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

// Constants
#define BLOCK_SIZE_X 32   // Threads per block (X dimension)
#define BLOCK_SIZE_Y 8    // Threads per block (Y dimension)
#define TILE_SIZE 32      // Tile size for shared memory (prevent bank conflicts)
#define SHARED_MEM_SIZE 3072  // Bytes per block (~3KB, leaves room for other data)

// GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu_approx(float x) {
    const float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    return x * cdf;
}

__device__ __forceinline__ __nv_bfloat16 gelu_approx_bf16(__nv_bfloat16 x_bf) {
    float x = __bfloat162float(x_bf);
    float result = gelu_approx(x);
    return __float2bfloat16(result);
}

// Fused FFN kernel: float32 version
// Input:  [batch*seq, hidden]
// FFN_up: [hidden, 4*hidden]
// GELU (in-place on intermediate)
// FFN_down: [4*hidden, hidden]
// Output: [batch*seq, hidden]
__global__ void fused_ffn_f32(
    const float* __restrict__ input,
    const float* __restrict__ ffn_up,
    const float* __restrict__ ffn_down,
    float* __restrict__ output,
    int batch_seq,  // batch * seq_len
    int hidden,
    int ffn_hidden  // 4 * hidden
) {
    __shared__ float smem[SHARED_MEM_SIZE / sizeof(float)];

    // Thread indices
    const int tx = threadIdx.x;  // [0, BLOCK_SIZE_X)
    const int ty = threadIdx.y;  // [0, BLOCK_SIZE_Y)
    const int bx = blockIdx.x;   // batch*seq block
    const int by = blockIdx.y;   // hidden dimension block

    // Row: which token (batch*seq index)
    const int row = bx * BLOCK_SIZE_Y + ty;
    if (row >= batch_seq) return;

    // Column: starting hidden dimension
    const int col_start = by * TILE_SIZE + tx;

    // Stage 1: Load input row once, reuse for all FFN_up columns
    // Load 1 row of input [1, hidden] into registers
    float input_val = 0.0f;
    if (tx < hidden) {
        input_val = input[row * hidden + tx];
    }

    // Stage 2: Compute FFN_up (input × W_up) for this thread's FFN_hidden columns
    // Each thread computes 1 value in intermediate [batch*seq, 4*hidden]
    if (col_start < ffn_hidden) {
        float accum = 0.0f;
        for (int k = 0; k < hidden; ++k) {
            float inp = input[row * hidden + k];
            float weight = ffn_up[k * ffn_hidden + col_start];
            accum += inp * weight;
        }
        // Write intermediate to shared memory (reuse for GELU + FFN_down)
        int smem_idx = ty * TILE_SIZE + tx;
        if (smem_idx < SHARED_MEM_SIZE / sizeof(float)) {
            smem[smem_idx] = gelu_approx(accum);
        }
    }
    __syncthreads();

    // Stage 3: Compute FFN_down (GELU(intermediate) × W_down)
    // Read from shared memory, compute output
    if (col_start < hidden) {
        float accum = 0.0f;
        for (int k = 0; k < ffn_hidden; ++k) {
            int smem_idx = (k / TILE_SIZE) * TILE_SIZE + (k % TILE_SIZE);
            if (smem_idx < SHARED_MEM_SIZE / sizeof(float)) {
                float gelu_val = smem[smem_idx];
                float weight = ffn_down[k * hidden + col_start];
                accum += gelu_val * weight;
            }
        }
        output[row * hidden + col_start] = accum;
    }
}

// Fused FFN kernel: bfloat16 version
// Same logic but with bf16 data type
__global__ void fused_ffn_bf16(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ ffn_up,
    const __nv_bfloat16* __restrict__ ffn_down,
    __nv_bfloat16* __restrict__ output,
    int batch_seq,
    int hidden,
    int ffn_hidden
) {
    __shared__ __nv_bfloat16 smem[SHARED_MEM_SIZE / sizeof(__nv_bfloat16)];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int row = bx * BLOCK_SIZE_Y + ty;
    if (row >= batch_seq) return;

    const int col_start = by * TILE_SIZE + tx;

    // Stage 1: FFN_up computation
    if (col_start < ffn_hidden) {
        float accum = 0.0f;
        for (int k = 0; k < hidden; ++k) {
            float inp = __bfloat162float(input[row * hidden + k]);
            float weight = __bfloat162float(ffn_up[k * ffn_hidden + col_start]);
            accum += inp * weight;
        }
        int smem_idx = ty * TILE_SIZE + tx;
        if (smem_idx < SHARED_MEM_SIZE / sizeof(__nv_bfloat16)) {
            smem[smem_idx] = __float2bfloat16(gelu_approx(accum));
        }
    }
    __syncthreads();

    // Stage 2: FFN_down computation
    if (col_start < hidden) {
        float accum = 0.0f;
        for (int k = 0; k < ffn_hidden; ++k) {
            int smem_idx = (k / TILE_SIZE) * TILE_SIZE + (k % TILE_SIZE);
            if (smem_idx < SHARED_MEM_SIZE / sizeof(__nv_bfloat16)) {
                float gelu_val = __bfloat162float(smem[smem_idx]);
                float weight = __bfloat162float(ffn_down[k * hidden + col_start]);
                accum += gelu_val * weight;
            }
        }
        output[row * hidden + col_start] = __float2bfloat16(accum);
    }
}

// Host wrapper: Fused FFN for float32
extern "C" cudaError_t launch_fused_ffn_f32(
    const float* input,
    const float* ffn_up,
    const float* ffn_down,
    float* output,
    int batch_seq,
    int hidden,
    int ffn_hidden,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((batch_seq + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
              (ffn_hidden + TILE_SIZE - 1) / TILE_SIZE);

    fused_ffn_f32<<<grid, block, 0, stream>>>(
        input, ffn_up, ffn_down, output, batch_seq, hidden, ffn_hidden
    );
    return cudaGetLastError();
}

// Host wrapper: Fused FFN for bfloat16
extern "C" cudaError_t launch_fused_ffn_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* ffn_up,
    const __nv_bfloat16* ffn_down,
    __nv_bfloat16* output,
    int batch_seq,
    int hidden,
    int ffn_hidden,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((batch_seq + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
              (ffn_hidden + TILE_SIZE - 1) / TILE_SIZE);

    fused_ffn_bf16<<<grid, block, 0, stream>>>(
        input, ffn_up, ffn_down, output, batch_seq, hidden, ffn_hidden
    );
    return cudaGetLastError();
}

// Benchmark: Measure kernel performance
extern "C" float benchmark_fused_ffn_f32(
    const float* input,
    const float* ffn_up,
    const float* ffn_down,
    float* output,
    int batch_seq,
    int hidden,
    int ffn_hidden,
    int iterations
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        launch_fused_ffn_f32(input, ffn_up, ffn_down, output,
                             batch_seq, hidden, ffn_hidden, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / iterations;  // Average time per kernel call
}
