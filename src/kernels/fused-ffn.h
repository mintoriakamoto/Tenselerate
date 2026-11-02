// fused-ffn.h – Public interface for fused FFN kernel
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Launch fused FFN kernel for float32
// Computes: output = FFN_down(GELU(input @ FFN_up))
//
// Args:
//   input:      [batch*seq, hidden] float32 tensor
//   ffn_up:     [hidden, 4*hidden] float32 weight matrix
//   ffn_down:   [4*hidden, hidden] float32 weight matrix
//   output:     [batch*seq, hidden] float32 result (pre-allocated)
//   batch_seq:  batch * seq_len (number of tokens)
//   hidden:     embedding dimension (e.g., 2048 for Hermes)
//   ffn_hidden: 4 * hidden (intermediate dimension)
//   stream:     CUDA stream for async execution
//
// Returns: CUDA error code
cudaError_t launch_fused_ffn_f32(
    const float* input,
    const float* ffn_up,
    const float* ffn_down,
    float* output,
    int batch_seq,
    int hidden,
    int ffn_hidden,
    cudaStream_t stream
);

// Launch fused FFN kernel for bfloat16
// Same signature, but all data is __nv_bfloat16
cudaError_t launch_fused_ffn_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* ffn_up,
    const __nv_bfloat16* ffn_down,
    __nv_bfloat16* output,
    int batch_seq,
    int hidden,
    int ffn_hidden,
    cudaStream_t stream
);

// Benchmark: Measure kernel throughput
// Runs kernel 'iterations' times and returns average time (milliseconds)
float benchmark_fused_ffn_f32(
    const float* input,
    const float* ffn_up,
    const float* ffn_down,
    float* output,
    int batch_seq,
    int hidden,
    int ffn_hidden,
    int iterations
);

#ifdef __cplusplus
}
#endif
