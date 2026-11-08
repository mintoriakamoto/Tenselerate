// fused-sparse-attn.h – Public interface for sparse attention kernel
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifdef __cplusplus
extern "C" {
#endif

// Launch sparse attention kernel for float32
// Computes: out = softmax(Q @ K^T / sqrt(d_k)) @ V
// With sparsity: attend to [local_window + stride_indices]
//
// Args:
//   Q:                 [batch*seq, n_heads, d_k] query embeddings
//   K:                 [batch*seq, n_heads, d_k] key embeddings
//   V:                 [batch*seq, n_heads, d_v] value embeddings
//   out:               [batch*seq, n_heads, d_v] output (pre-allocated)
//   batch_seq:         batch * seq_len
//   n_heads:           number of attention heads
//   d_k:               dimension per head (key/query)
//   d_v:               dimension per head (value, usually = d_k)
//   local_window_size: size of local attention window (e.g., 64)
//   stride:            attend to every Nth token in strided pattern (e.g., 8)
//   stream:            CUDA stream
//
// Sparsity pattern example:
//   seq_len=4096, d_k=64, n_heads=32
//   local_window_size=64, stride=8
//   → Each token attends to ~96 positions (64 local + 512 strided)
//   → Reduction: 4096 → 96 = ~96% sparsity
//
cudaError_t launch_fused_sparse_attn_f32(
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
);

// BF16 version
cudaError_t launch_fused_sparse_attn_bf16(
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
);

#ifdef __cplusplus
}
#endif
