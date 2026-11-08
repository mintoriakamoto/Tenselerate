// llama-optimizations.hpp – High-level optimization orchestration
// Integrates Phase 2 fused kernels into inference without modifying core loop

#pragma once

#include <memory>
#include <cstdint>
#include <vector>
#include <stdexcept>

#ifdef GGML_USE_CUDA
    #include "kernels/fused-ffn.h"
    #include "kernels/fused-sparse-attn.h"
#endif

namespace tenselerate {

// Optimization configuration
struct OptimizationConfig {
    bool use_fused_ffn = true;
    bool use_sparse_attention = true;

    int sparse_local_window = 64;  // Local attention window size
    int sparse_stride = 8;          // Stride for sparse attention

    bool fallback_on_error = true;  // Fall back to dense if kernel fails
};

// Optimization statistics
struct OptimizationStats {
    uint64_t ffn_calls = 0;
    uint64_t ffn_fallbacks = 0;
    uint64_t attn_calls = 0;
    uint64_t attn_fallbacks = 0;

    float total_ffn_time_ms = 0.0f;
    float total_attn_time_ms = 0.0f;
};

// Phase 2 Optimization Manager
class OptimizationManager {
public:
    OptimizationManager() = default;

    // Initialize optimizations for a specific model config
    void initialize(
        int hidden_dim,
        int n_heads,
        int d_k,
        int d_v
    ) {
        #ifdef GGML_USE_CUDA
        try {
            // Initialize FFN executor
            ffn_executor_ = std::make_unique<tenselerate::FusedFFNExecutor>(
                hidden_dim,
                hidden_dim * 4,
                tenselerate::FusedFFNExecutor::FLOAT32
            );

            // Initialize sparse attention executor
            attn_executor_ = std::make_unique<tenselerate::FusedSparseAttnExecutor>(
                n_heads, d_k, d_v,
                config_.sparse_local_window,
                config_.sparse_stride
            );

            initialized_ = true;
            fprintf(stderr, "[llama-opt] Phase 2 optimizations initialized\n");
        } catch (const std::exception& e) {
            fprintf(stderr, "[llama-opt] Warning: Failed to initialize optimizations: %s\n", e.what());
            initialized_ = false;
        }
        #else
        fprintf(stderr, "[llama-opt] CUDA not available, optimizations disabled\n");
        initialized_ = false;
        #endif
    }

    // Execute FFN with fallback strategy
    bool try_fused_ffn(
        const void* input,
        const void* ffn_up,
        const void* ffn_down,
        void* output,
        int batch_seq,
        int hidden_dim
    ) {
        if (!initialized_ || !config_.use_fused_ffn) {
            return false;  // Caller should use fallback
        }

        #ifdef GGML_USE_CUDA
        try {
            cudaError_t err = ffn_executor_->execute(
                input, ffn_up, ffn_down, output, batch_seq, nullptr
            );

            if (err == cudaSuccess) {
                stats_.ffn_calls++;
                return true;
            } else {
                if (config_.fallback_on_error) {
                    stats_.ffn_fallbacks++;
                    fprintf(stderr, "[llama-opt] FFN kernel failed (err=%d), using fallback\n", err);
                    return false;
                } else {
                    throw std::runtime_error("FFN kernel failed");
                }
            }
        } catch (const std::exception& e) {
            if (config_.fallback_on_error) {
                stats_.ffn_fallbacks++;
                fprintf(stderr, "[llama-opt] FFN execution error: %s\n", e.what());
                return false;
            } else {
                throw;
            }
        }
        #endif

        return false;
    }

    // Execute sparse attention with fallback
    bool try_sparse_attention(
        const void* Q,
        const void* K,
        const void* V,
        void* output,
        int batch_seq,
        int n_heads,
        int d_k,
        int d_v
    ) {
        if (!initialized_ || !config_.use_sparse_attention) {
            return false;
        }

        #ifdef GGML_USE_CUDA
        try {
            cudaError_t err = attn_executor_->execute(
                Q, K, V, output,
                batch_seq, n_heads, d_k, d_v,
                config_.sparse_local_window,
                config_.sparse_stride,
                nullptr
            );

            if (err == cudaSuccess) {
                stats_.attn_calls++;
                return true;
            } else {
                if (config_.fallback_on_error) {
                    stats_.attn_fallbacks++;
                    fprintf(stderr, "[llama-opt] Attention kernel failed (err=%d), using fallback\n", err);
                    return false;
                } else {
                    throw std::runtime_error("Attention kernel failed");
                }
            }
        } catch (const std::exception& e) {
            if (config_.fallback_on_error) {
                stats_.attn_fallbacks++;
                fprintf(stderr, "[llama-opt] Attention execution error: %s\n", e.what());
                return false;
            } else {
                throw;
            }
        }
        #endif

        return false;
    }

    // Configuration
    void set_config(const OptimizationConfig& config) {
        config_ = config;
    }

    OptimizationConfig get_config() const {
        return config_;
    }

    // Statistics
    OptimizationStats get_stats() const {
        return stats_;
    }

    void reset_stats() {
        stats_ = OptimizationStats();
    }

    void print_stats() const {
        fprintf(stderr, "\n[llama-opt] Phase 2 Statistics:\n");
        fprintf(stderr, "  FFN kernel calls:      %llu (fallbacks: %llu)\n",
                stats_.ffn_calls, stats_.ffn_fallbacks);
        fprintf(stderr, "  Attention calls:       %llu (fallbacks: %llu)\n",
                stats_.attn_calls, stats_.attn_fallbacks);
        if (stats_.ffn_calls > 0) {
            fprintf(stderr, "  Avg FFN time:          %.3f ms\n",
                    stats_.total_ffn_time_ms / stats_.ffn_calls);
        }
        if (stats_.attn_calls > 0) {
            fprintf(stderr, "  Avg attention time:    %.3f ms\n",
                    stats_.total_attn_time_ms / stats_.attn_calls);
        }
    }

    bool is_initialized() const {
        return initialized_;
    }

private:
    #ifdef GGML_USE_CUDA
    std::unique_ptr<tenselerate::FusedFFNExecutor> ffn_executor_;
    std::unique_ptr<tenselerate::FusedSparseAttnExecutor> attn_executor_;
    #endif

    OptimizationConfig config_;
    OptimizationStats stats_;
    bool initialized_ = false;
};

// Global optimization manager (singleton)
inline OptimizationManager& get_optimization_manager() {
    static OptimizationManager manager;
    return manager;
}

} // namespace tenselerate

// C interface for easy integration
extern "C" {
    void tenselerate_opt_initialize(int hidden, int n_heads, int d_k, int d_v);
    int tenselerate_opt_try_fused_ffn(
        const void* input,
        const void* ffn_up,
        const void* ffn_down,
        void* output,
        int batch_seq,
        int hidden
    );
    int tenselerate_opt_try_sparse_attn(
        const void* Q,
        const void* K,
        const void* V,
        void* output,
        int batch_seq,
        int n_heads,
        int d_k,
        int d_v
    );
    void tenselerate_opt_print_stats();
}
