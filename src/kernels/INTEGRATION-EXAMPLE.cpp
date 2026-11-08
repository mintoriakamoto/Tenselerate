// INTEGRATION-EXAMPLE.cpp – Example of using OptimizationManager in decode loop
// This shows WHERE and HOW to call fused kernels from llama-context.cpp

#include "llama-optimizations.hpp"

// ============================================================================
// EXAMPLE 1: Initialize optimizations at context creation time
// ============================================================================

// In llama_context constructor (llama-context.cpp, ~line 84):
void example_initialize_optimizations(
    int hidden_dim,
    int n_heads,
    int d_k,
    int d_v
) {
    // Initialize the optimization manager once per context
    tenselerate::get_optimization_manager().initialize(
        hidden_dim, n_heads, d_k, d_v
    );

    // Optionally configure
    tenselerate::OptimizationConfig config;
    config.use_fused_ffn = true;
    config.use_sparse_attention = true;
    config.sparse_local_window = 64;
    config.sparse_stride = 8;
    config.fallback_on_error = true;  // Always safe

    tenselerate::get_optimization_manager().set_config(config);
}

// ============================================================================
// EXAMPLE 2: Replace FFN computation in the inference loop
// ============================================================================

// In decode loop (llama-context.cpp, wherever FFN is computed):

// BEFORE (3 separate kernels):
void old_ffn_computation(
    const float* hidden_state,
    const float* ffn_up_weight,
    const float* ffn_down_weight,
    float* output,
    int batch_seq,
    int hidden
) {
    // Linear expansion
    dense_gemm(hidden_state, ffn_up_weight, intermediate, batch_seq, hidden, 4*hidden);

    // GELU activation
    gelu_forward(intermediate);

    // Linear projection
    dense_gemm(intermediate, ffn_down_weight, output, batch_seq, 4*hidden, hidden);
}

// AFTER (1 fused kernel with fallback):
void new_ffn_computation(
    const float* hidden_state,
    const float* ffn_up_weight,
    const float* ffn_down_weight,
    float* output,
    int batch_seq,
    int hidden
) {
    // Try optimized fused kernel
    bool success = tenselerate::get_optimization_manager().try_fused_ffn(
        hidden_state,
        ffn_up_weight,
        ffn_down_weight,
        output,
        batch_seq,
        hidden
    );

    if (!success) {
        // Fallback to separate kernels (always safe)
        dense_gemm(hidden_state, ffn_up_weight, intermediate, batch_seq, hidden, 4*hidden);
        gelu_forward(intermediate);
        dense_gemm(intermediate, ffn_down_weight, output, batch_seq, 4*hidden, hidden);
    }
}

// ============================================================================
// EXAMPLE 3: Replace dense attention with sparse attention
// ============================================================================

// BEFORE (dense attention):
void old_attention_computation(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int batch_seq,
    int n_heads,
    int d_k,
    int d_v
) {
    // Dense: Q @ K^T / sqrt(d_k) → softmax → @ V
    dense_attention(Q, K, V, output, batch_seq, n_heads, d_k, d_v);
}

// AFTER (sparse attention with fallback):
void new_attention_computation(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int batch_seq,
    int n_heads,
    int d_k,
    int d_v
) {
    // Try optimized sparse attention
    bool success = tenselerate::get_optimization_manager().try_sparse_attention(
        Q, K, V, output,
        batch_seq, n_heads, d_k, d_v
    );

    if (!success) {
        // Fallback to dense attention
        dense_attention(Q, K, V, output, batch_seq, n_heads, d_k, d_v);
    }
}

// ============================================================================
// EXAMPLE 4: Report statistics at end of inference
// ============================================================================

void print_optimization_report() {
    tenselerate::get_optimization_manager().print_stats();

    // Example output:
    // [llama-opt] Phase 2 Statistics:
    //   FFN kernel calls:      128 (fallbacks: 0)
    //   Attention calls:       64 (fallbacks: 0)
    //   Avg FFN time:          0.345 ms
    //   Avg attention time:    0.890 ms
}

// ============================================================================
// EXAMPLE 5: Configuration at runtime
// ============================================================================

void example_configure_optimizations() {
    auto& mgr = tenselerate::get_optimization_manager();

    // Get current config
    auto config = mgr.get_config();

    // Modify
    config.use_fused_ffn = true;
    config.use_sparse_attention = true;
    config.sparse_local_window = 128;  // Larger window = more compute but better quality
    config.sparse_stride = 16;          // Larger stride = less compute but more sparse
    config.fallback_on_error = true;    // Always maintain safety

    // Set back
    mgr.set_config(config);
}

// ============================================================================
// EXAMPLE 6: Compilation Test
// ============================================================================

/*
To compile and test:

1. Build with fused kernels:
   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
   make -C build -j$(nproc) 2>&1 | grep -E "llama|CUDA|Error|kernel"

2. Expected output:
   [100%] Linking CXX shared library bin/libllama.so
   [llama] CUDA architectures: 86
   [llama] Phase 2 fused kernels enabled (FFN + sparse attention)
   [Built target llama]

3. Run inference:
   ./bin/llama-cli -m model.gguf -n 128

   At end, you should see:
   [llama-opt] Phase 2 Statistics:
     FFN kernel calls:      256 (fallbacks: 0)
     Attention calls:       128 (fallbacks: 0)
     ...

4. If you want to disable optimizations:
   - Set optimize_config.use_fused_ffn = false
   - Or build without GGML_CUDA
*/

// ============================================================================
// NEXT STEPS
// ============================================================================

/*
To fully integrate:

1. Locate actual FFN computation in llama-context.cpp decode loop
   - Search for: ggml_mul_mat + ggml_gelu + ggml_mul_mat pattern

2. Locate actual attention computation
   - Search for: Q @ K^T / sqrt(d_k) → softmax → @ V

3. Replace with calls to:
   - tenselerate::get_optimization_manager().try_fused_ffn(...)
   - tenselerate::get_optimization_manager().try_sparse_attention(...)

4. Test on RTX 3060 Ti:
   - Measure baseline (no optimizations)
   - Measure with Phase 2 (fused kernels)
   - Compare: should see 2-3× FFN speedup, 1.8-2.0× attention speedup

5. Benchmark full inference:
   - Before: ~12 tok/s
   - After Phase 1+2: ~25-40 tok/s
*/
