// llama-optimizations.cpp – Implementation of optimization orchestration

#include "llama-optimizations.hpp"

namespace tenselerate {

// Implementation of C interface
} // namespace tenselerate

// Global C interface
extern "C" {

void tenselerate_opt_initialize(int hidden, int n_heads, int d_k, int d_v) {
    #ifdef GGML_USE_CUDA
    try {
        tenselerate::get_optimization_manager().initialize(hidden, n_heads, d_k, d_v);
    } catch (const std::exception& e) {
        fprintf(stderr, "[llama-opt] Initialization failed: %s\n", e.what());
    }
    #endif
}

int tenselerate_opt_try_fused_ffn(
    const void* input,
    const void* ffn_up,
    const void* ffn_down,
    void* output,
    int batch_seq,
    int hidden
) {
    #ifdef GGML_USE_CUDA
    return tenselerate::get_optimization_manager().try_fused_ffn(
        input, ffn_up, ffn_down, output, batch_seq, hidden
    ) ? 1 : 0;
    #else
    return 0;  // Not available
    #endif
}

int tenselerate_opt_try_sparse_attn(
    const void* Q,
    const void* K,
    const void* V,
    void* output,
    int batch_seq,
    int n_heads,
    int d_k,
    int d_v
) {
    #ifdef GGML_USE_CUDA
    return tenselerate::get_optimization_manager().try_sparse_attention(
        Q, K, V, output, batch_seq, n_heads, d_k, d_v
    ) ? 1 : 0;
    #else
    return 0;  // Not available
    #endif
}

void tenselerate_opt_print_stats() {
    #ifdef GGML_USE_CUDA
    tenselerate::get_optimization_manager().print_stats();
    #endif
}

}
