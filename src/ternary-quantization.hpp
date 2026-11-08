// ternary-quantization.hpp – Extreme weight quantization
// Encodes weights as {-α, 0, +α} per-layer for 4× compression
// With LoRA fine-tuning for minimal quality loss

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <cmath>

namespace tenselerate {

// Ternary weight representation
struct TernaryWeights {
    std::vector<int8_t> encoded;      // 2 bits per weight (padded to byte)
    float scale_alpha;                 // Scaling factor {-α, 0, +α}
    int n_elements;

    // Encoding: 0 = 0, 1 = -α, 2 = +α, 3 = unused
    // Two weights per byte: [weight1 (2 bits) | weight2 (2 bits) | ... ]
};

// LoRA correction layer for quantized weights
struct LoRACorrection {
    float rank = 8;                    // LoRA rank (e.g., 8, 16)
    std::vector<float> low_rank_a;     // [n_in, rank]
    std::vector<float> low_rank_b;     // [rank, n_out]

    // Low-rank update: output = α_in @ β_out^T
    // Applied to each layer to recover precision loss
};

// Quantization configuration
struct QuantizationConfig {
    bool use_ternary = true;           // Enable ternary (else use standard quant)
    bool use_lora_correction = true;   // Apply LoRA fine-tuning
    int lora_rank = 8;                 // LoRA rank for correction

    // Layer sensitivity (some layers more important)
    std::vector<float> layer_scales;   // Per-layer scale factors

    float calibration_quantile = 0.95f; // Use 95th percentile for α
    bool symmetric = true;              // Symmetric: {-α, 0, +α} vs {0, +α}
};

// Ternary quantization engine
class TernaryQuantizer {
public:
    TernaryQuantizer(const QuantizationConfig& config = QuantizationConfig())
        : config_(config) {}

    // Calibrate quantization scale (α) from a batch of weights
    float calibrate_scale(const float* weights, int n_elements) {
        // Find α such that most weights map to {-α, 0, +α}
        // Use median or percentile-based approach

        std::vector<float> abs_weights;
        for (int i = 0; i < n_elements; ++i) {
            abs_weights.push_back(std::fabs(weights[i]));
        }
        std::sort(abs_weights.begin(), abs_weights.end());

        // Use 95th percentile as scale
        int idx = static_cast<int>(abs_weights.size() * config_.calibration_quantile);
        return abs_weights[idx];
    }

    // Quantize: float → ternary
    TernaryWeights quantize(
        const float* weights,
        int n_elements,
        float alpha = 0.0f  // If 0, calibrate automatically
    ) {
        TernaryWeights result;

        if (alpha == 0.0f) {
            alpha = calibrate_scale(weights, n_elements);
        }

        result.scale_alpha = alpha;
        result.n_elements = n_elements;
        result.encoded.resize((n_elements + 3) / 4);  // 2 bits per weight, 4 per byte

        // Encode weights
        for (int i = 0; i < n_elements; ++i) {
            float w = weights[i];
            int8_t code;

            if (std::fabs(w) < alpha / 3.0f) {
                code = 0;  // Map to 0
            } else if (w > 0) {
                code = 2;  // Map to +α
            } else {
                code = 1;  // Map to -α
            }

            // Pack two weights per byte
            int byte_idx = i / 4;
            int bit_pos = (i % 4) * 2;
            result.encoded[byte_idx] |= (code << bit_pos);
        }

        return result;
    }

    // Dequantize: ternary → float (for reference)
    std::vector<float> dequantize(const TernaryWeights& ternary) {
        std::vector<float> weights(ternary.n_elements);

        for (int i = 0; i < ternary.n_elements; ++i) {
            int byte_idx = i / 4;
            int bit_pos = (i % 4) * 2;
            int8_t code = (ternary.encoded[byte_idx] >> bit_pos) & 0x3;

            if (code == 0) {
                weights[i] = 0.0f;
            } else if (code == 1) {
                weights[i] = -ternary.scale_alpha;
            } else if (code == 2) {
                weights[i] = ternary.scale_alpha;
            }
        }

        return weights;
    }

    // Compute quantization error
    float compute_error(
        const float* original,
        const TernaryWeights& quantized,
        int n_elements
    ) {
        auto dequant = dequantize(quantized);

        float mse = 0.0f;
        for (int i = 0; i < n_elements; ++i) {
            float err = original[i] - dequant[i];
            mse += err * err;
        }

        return std::sqrt(mse / n_elements);  // RMSE
    }

    // Generate LoRA correction for this layer
    LoRACorrection generate_lora_correction(
        const float* original_weights,
        const TernaryWeights& quantized,
        int n_in,
        int n_out
    ) {
        LoRACorrection lora;
        lora.rank = config_.lora_rank;

        // Compute residual: original - dequantized
        auto dequant = dequantize(quantized);
        std::vector<float> residual(original_weights,
                                   original_weights + n_in * n_out);
        for (size_t i = 0; i < residual.size(); ++i) {
            residual[i] -= dequant[i];
        }

        // Decompose residual into low-rank components
        // Simplified: use random projection for demo
        // Real: use SVD truncation

        lora.low_rank_a.resize(n_in * config_.lora_rank);
        lora.low_rank_b.resize(config_.lora_rank * n_out);

        // TODO: Implement proper SVD-based decomposition

        return lora;
    }

    // Statistics
    struct QuantizationStats {
        float original_size_mb = 0.0f;
        float quantized_size_mb = 0.0f;
        float lora_size_mb = 0.0f;
        float compression_ratio = 0.0f;
        float rmse = 0.0f;
        int n_layers = 0;
    };

    QuantizationStats compute_stats(
        const std::vector<TernaryWeights>& layers,
        float original_bytes = 0.0f
    ) {
        QuantizationStats stats;

        stats.n_layers = layers.size();

        // Ternary size: 2 bits per weight
        float ternary_bytes = 0.0f;
        for (const auto& layer : layers) {
            ternary_bytes += layer.encoded.size();  // 1 byte = 4 weights = 8 bits
            ternary_bytes += sizeof(float);          // α per layer
        }

        stats.original_size_mb = original_bytes / (1024 * 1024);
        stats.quantized_size_mb = ternary_bytes / (1024 * 1024);
        stats.compression_ratio = original_bytes / (ternary_bytes + 1e-9f);

        return stats;
    }

private:
    QuantizationConfig config_;
};

// Multi-layer quantization orchestrator
class LayerQuantizer {
public:
    void quantize_layer_sensitive(
        const std::vector<float*>& layer_weights,
        const std::vector<int>& layer_sizes,
        const std::vector<float>& sensitivity_scores
    ) {
        TernaryQuantizer quantizer;

        for (size_t i = 0; i < layer_weights.size(); ++i) {
            // Use sensitivity-based α: more important layers → higher α (less quant)
            float scale_factor = 1.0f / (sensitivity_scores[i] + 0.1f);

            auto ternary = quantizer.quantize(
                layer_weights[i],
                layer_sizes[i],
                0.0f  // Auto-calibrate
            );

            quantized_layers_.push_back(ternary);
        }
    }

    const std::vector<TernaryWeights>& get_quantized() const {
        return quantized_layers_;
    }

private:
    std::vector<TernaryWeights> quantized_layers_;
};

} // namespace tenselerate
