# Integration Guide: Fused Kernels into llama.cpp

## Overview

This document describes how to integrate Phase 2 fused kernels (FFN and sparse attention) into the inference loop in `src/llama.cpp`.

## Architecture

```
llama_decode()
    ├─ For each layer:
    │   ├─ Attention: Q, K, V × dense_attention() [OLD]
    │   │  →  fused_sparse_attn() [NEW]
    │   ├─ FFN: linear_up → gelu → linear_down [OLD]
    │   │  →  fused_ffn() [NEW]
    │   └─ Layer norm + residual
    └─ Return logits
```

## Step 1: Add Kernel Declarations

In `src/llama.cpp`, add after includes:

```cpp
#if defined(GGML_USE_CUDA)
    #include "kernels/fused-ffn.h"
    #include "kernels/fused-sparse-attn.h"
#endif
```

## Step 2: Create Executor Classes

Add to `src/llama.cpp` (in anonymous namespace):

```cpp
namespace {
    struct FusedKernelExecutors {
        std::unique_ptr<tenselerate::FusedFFNExecutor> ffn;
        std::unique_ptr<tenselerate::FusedSparseAttnExecutor> attn;
        
        bool initialized = false;
        
        void init(int hidden, int n_heads, int d_k, int d_v) {
            #if defined(GGML_USE_CUDA)
            try {
                ffn = std::make_unique<tenselerate::FusedFFNExecutor>(
                    hidden, hidden * 4, tenselerate::FusedFFNExecutor::FLOAT32
                );
                attn = std::make_unique<tenselerate::FusedSparseAttnExecutor>(
                    n_heads, d_k, d_v, 64, 8  // local_window=64, stride=8
                );
                initialized = true;
                fprintf(stderr, "[llama] Fused kernels initialized (FFN + sparse attention)\n");
            } catch (const std::exception& e) {
                fprintf(stderr, "[llama] Warning: Failed to init fused kernels: %s\n", e.what());
                initialized = false;
            }
            #endif
        }
    };
} // namespace

// Global executor (initialized on first model load)
FusedKernelExecutors g_fused_kernels;
```

## Step 3: Replace Dense Attention

In `llama_decode_internal()`, find the attention computation:

**Before (Dense Attention):**
```cpp
// Q @ K^T / sqrt(d_k)
ggml_mul_mat(ctx, ggml_cont(ctx, K), Q);
// Apply causal mask + softmax
// @ V
```

**After (Sparse Attention):**
```cpp
if (g_fused_kernels.initialized && use_sparse_attention) {
    // Use fused sparse attention
    cudaError_t err = g_fused_kernels.attn->execute(
        Q_data, K_data, V_data, attention_out,
        batch_seq, n_heads, d_k, d_v,
        64, 8,  // local_window, stride
        stream
    );
    if (err != cudaSuccess) {
        fprintf(stderr, "[llama] Sparse attention failed, falling back to dense\n");
        // Fall back to dense attention
    }
} else {
    // Dense attention (fallback)
}
```

## Step 4: Replace FFN

In the FFN layer computation:

**Before (Separate Kernels):**
```cpp
// Linear up: hidden → 4*hidden
ggml_mul_mat(ctx, W_up, hidden_state);
// GELU activation
ggml_gelu(ctx, intermediate);
// Linear down: 4*hidden → hidden
ggml_mul_mat(ctx, W_down, gelu_output);
```

**After (Fused FFN):**
```cpp
if (g_fused_kernels.initialized && use_fused_ffn) {
    // Use fused FFN
    cudaError_t err = g_fused_kernels.ffn->execute(
        hidden_state_data,
        ffn_up_weight_data,
        ffn_down_weight_data,
        ffn_output_data,
        batch_seq,
        stream
    );
    if (err != cudaSuccess) {
        fprintf(stderr, "[llama] Fused FFN failed, falling back to separate kernels\n");
        // Fall back to separate kernels
    }
} else {
    // Separate kernels (fallback)
}
```

## Step 5: Control Flags

Add command-line options:

```cpp
// In llama_model_params
struct llama_model_params params {
    bool use_fused_kernels = true;      // Enable Phase 2 optimizations
    bool use_sparse_attention = true;   // Use sparse attention pattern
    bool use_fused_ffn = true;          // Use fused FFN
};

// In llama_cli main()
if (arg == "--no-fused-kernels") {
    params.use_fused_kernels = false;
    params.use_sparse_attention = false;
    params.use_fused_ffn = false;
}
```

## Step 6: Fallback Strategy

Always provide fallback to dense kernels:

```cpp
// Wrapper: try optimized kernel, fallback to dense
int attention_with_fallback(
    const float* Q, const float* K, const float* V,
    float* output,
    int batch_seq, int n_heads, int d_k, int d_v,
    bool prefer_sparse
) {
    if (prefer_sparse && g_fused_kernels.initialized) {
        cudaError_t err = g_fused_kernels.attn->execute(
            Q, K, V, output, batch_seq, n_heads, d_k, d_v, 64, 8, nullptr
        );
        if (err == cudaSuccess) return 0;
        
        fprintf(stderr, "[llama] Sparse attention failed (err=%d), using dense\n", err);
    }
    
    // Dense attention fallback
    return dense_attention(Q, K, V, output, batch_seq, n_heads, d_k, d_v);
}
```

## Step 7: Numerical Correctness Validation

Add unit test before production deployment:

```cpp
// benches/test-fused-integration.cu
void test_fused_ffn_vs_separate() {
    // Generate random input
    float* input = allocate_gpu(batch_seq * hidden);
    fill_random(input, batch_seq * hidden);
    
    // Run fused FFN
    float* output_fused = allocate_gpu(batch_seq * hidden);
    launch_fused_ffn_f32(input, W_up, W_down, output_fused, batch_seq, hidden, 4*hidden, nullptr);
    
    // Run separate kernels (reference)
    float* output_separate = allocate_gpu(batch_seq * hidden);
    separate_ffn(input, W_up, W_down, output_separate);
    
    // Compare: max error < 1e-3 (acceptable for float32)
    float max_error = compare(output_fused, output_separate, batch_seq * hidden);
    assert(max_error < 1e-3);
}
```

## Performance Expectations

### RTX 3060 Ti (12GB VRAM)

| Operation | Baseline | Optimized | Speedup |
|-----------|----------|-----------|---------|
| FFN layer (1 token) | 0.85 ms | 0.35 ms | 2.4× |
| Attention (seq=64) | 1.20 ms | 0.60 ms | 2.0× |
| Attention (seq=256) | 3.50 ms | 1.05 ms | 3.3× |
| Full decode pass (128 tokens) | ~85 ms | ~35 ms | 2.4× |

### Expected Results After Integration

- **Decode throughput:** 12 → 25-30 tok/s (+120-150%)
- **Memory I/O:** Reduced by ~50% (less DRAM traffic)
- **Latency per token:** 85 ms → 35 ms (60% reduction)

## Debugging

### CUDA Error Checking

```cpp
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[CUDA Error] %s: %s\n", #call, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)
```

### Memory Layout Validation

Verify contiguous layouts before kernel call:

```cpp
// Verify Q is contiguous [batch*seq, n_heads, d_k]
assert(Q->nb[0] == n_heads * d_k * sizeof(float));
assert(Q->nb[1] == d_k * sizeof(float));
```

### Benchmark Before/After

```bash
# Baseline (dense kernels)
./bin/llama-cli -m model.gguf --no-fused-kernels -n 128

# Optimized (fused kernels)
./bin/llama-cli -m model.gguf -n 128

# Profile
nsys profile -o profile-fused ./bin/llama-cli -m model.gguf -n 128
```

## Rollback Plan

If fused kernels cause issues:

1. Set `use_fused_kernels = false` in model params
2. Rebuild: `cmake --build build`
3. Deploy: All calls fall back to dense kernels automatically
4. No recompilation of core llama.cpp needed (runtime switch)

## Next Steps

1. ✅ Phase 2.1: Fused FFN kernel
2. ✅ Phase 2.2: Sparse attention kernel
3. ⏳ Phase 2.3: Integration (THIS DOCUMENT)
4. ⏳ Phase 2.4: Hardware validation + benchmarking
5. ⏳ Phase 3: Speculative decoding
6. ⏳ Phase 4: Ternary quantization
7. ⏳ Phase 5: Advanced multi-agent batching

---

## References

- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Fused Operations](https://arxiv.org/abs/1811.12625)
- [Sparse Attention](https://arxiv.org/abs/1904.10509)
