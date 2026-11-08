// speculative-decoding.hpp – Speculative decoding orchestrator
// Parallelizes inference: draft model → verify with target → accept/reject
// Expected speedup: 1.5-2.5× latency reduction for single-token generation

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <random>
#include <cmath>

namespace tenselerate {

// Token probability entry
struct TokenProb {
    int32_t token_id;
    float log_prob;

    bool operator<(const TokenProb& other) const {
        return log_prob > other.log_prob;  // Higher prob first
    }
};

// Speculative decoding configuration
struct SpeculativeConfig {
    int max_draft_tokens = 4;          // Max tokens to generate speculatively
    float temperature_draft = 0.8f;    // Draft model temperature (lower = more confident)
    float temperature_target = 0.7f;   // Target model temperature

    bool use_acceptance_sampling = true;  // Verify via probability matching
    int random_seed = 42;

    bool verbose = false;
};

// Result of a verification round
struct VerificationResult {
    std::vector<int32_t> accepted_tokens;    // Tokens verified by target model
    std::vector<int32_t> draft_tokens;       // Original draft tokens (for logging)
    int n_verified = 0;                      // Tokens verified (before first rejection)
    int n_rejected = 0;                      // Tokens rejected
    float acceptance_rate = 0.0f;            // Fraction of draft tokens accepted

    bool all_accepted() const {
        return n_rejected == 0 && n_verified > 0;
    }
};

// Speculative decoding orchestrator
class SpeculativeDecoder {
public:
    SpeculativeDecoder(const SpeculativeConfig& config = SpeculativeConfig())
        : config_(config), gen_(config.random_seed) {}

    // Initialize: set model dimensions for draft vs target
    void initialize(int draft_vocab_size, int target_vocab_size) {
        draft_vocab_size_ = draft_vocab_size;
        target_vocab_size_ = target_vocab_size;

        // Ensure vocabularies are compatible (or pruned appropriately)
        if (draft_vocab_size > target_vocab_size) {
            // Map draft tokens to target vocab (typically already done via pruning)
            need_vocab_mapping_ = true;
        }
    }

    // Generate draft tokens speculatively
    std::vector<int32_t> draft_generate(
        const void* hidden_state,
        const void* draft_logits_fn,  // Function pointer: hidden → logits
        int n_tokens,                 // How many tokens to generate speculatively
        int batch_size = 1
    ) {
        std::vector<int32_t> draft_tokens;

        // Generate up to max_draft_tokens speculatively
        int n_to_gen = std::min(n_tokens, config_.max_draft_tokens);

        for (int i = 0; i < n_to_gen; ++i) {
            // Sample from draft model (simplified: argmax for now)
            // In production: temperature-scaled sampling from logits
            int token = sample_from_draft();  // TODO: call draft_logits_fn
            draft_tokens.push_back(token);
        }

        return draft_tokens;
    }

    // Verify draft tokens against target model
    VerificationResult verify_draft_tokens(
        const std::vector<int32_t>& draft_tokens,
        const void* target_logits,     // [n_draft, target_vocab_size]
        const void* draft_logits       // [n_draft, draft_vocab_size]
    ) {
        VerificationResult result;
        result.draft_tokens = draft_tokens;

        for (size_t i = 0; i < draft_tokens.size(); ++i) {
            int32_t draft_token = draft_tokens[i];

            // Get probabilities from both models
            float draft_prob = get_token_prob(draft_logits, i, draft_token);
            float target_prob = get_token_prob(target_logits, i, draft_token);

            // Acceptance sampling: P_target(token) / P_draft(token)
            float acceptance_ratio = target_prob / (draft_prob + 1e-9f);

            if (config_.use_acceptance_sampling) {
                // Sample uniformly; accept if u < ratio
                std::uniform_real_distribution<float> u(0.0f, 1.0f);
                if (u(gen_) < std::min(1.0f, acceptance_ratio)) {
                    result.accepted_tokens.push_back(draft_token);
                    result.n_verified++;
                } else {
                    // Token rejected; stop verification
                    result.n_rejected++;
                    break;
                }
            } else {
                // Greedy: accept if P_target(token) > P_draft(token)
                if (target_prob >= draft_prob) {
                    result.accepted_tokens.push_back(draft_token);
                    result.n_verified++;
                } else {
                    result.n_rejected++;
                    break;
                }
            }
        }

        result.acceptance_rate = result.n_verified > 0
            ? (float)result.n_verified / draft_tokens.size()
            : 0.0f;

        if (config_.verbose) {
            fprintf(stderr, "[speculative] Verified %d/%zu tokens (acceptance: %.1f%%)\n",
                   result.n_verified, draft_tokens.size(), result.acceptance_rate * 100);
        }

        return result;
    }

    // Generate final token if needed (draft verification failed or needed fresh token)
    int32_t generate_fallback_token(const void* target_logits) {
        // Sample from target model logits (greedy for now)
        return sample_from_logits(target_logits, target_vocab_size_);
    }

    // Full speculative decoding round
    // Returns: tokens to output (either verified draft or fallback)
    std::vector<int32_t> decode_speculative(
        const void* hidden_state,
        const void* draft_logits,
        const void* target_logits
    ) {
        // 1. Generate draft tokens
        std::vector<int32_t> draft = draft_generate(
            hidden_state, draft_logits, config_.max_draft_tokens
        );

        // 2. Verify against target
        auto verified = verify_draft_tokens(draft, target_logits, draft_logits);

        // 3. Return accepted tokens + fallback if needed
        if (verified.all_accepted()) {
            return verified.accepted_tokens;  // All draft tokens verified
        } else if (!verified.accepted_tokens.empty()) {
            return verified.accepted_tokens;  // Some tokens verified before rejection
        } else {
            // No tokens verified; generate new one from target
            std::vector<int32_t> fallback;
            fallback.push_back(generate_fallback_token(target_logits));
            return fallback;
        }
    }

    // Statistics
    struct Stats {
        uint64_t total_rounds = 0;
        uint64_t total_tokens_generated = 0;
        float avg_acceptance_rate = 0.0f;
        float speedup = 1.0f;
    };

    Stats get_stats() const {
        return stats_;
    }

private:
    SpeculativeConfig config_;
    int draft_vocab_size_ = 0;
    int target_vocab_size_ = 0;
    bool need_vocab_mapping_ = false;
    std::mt19937 gen_;
    Stats stats_;

    // Helper: get probability of token from logits
    float get_token_prob(const void* logits, int position, int token_id) const {
        // TODO: implement properly with actual logits tensor
        // For now: return dummy value
        return 0.5f;
    }

    // Helper: sample token from draft model
    int32_t sample_from_draft() {
        // TODO: call draft model inference
        return 0;
    }

    // Helper: sample token from logits array
    int32_t sample_from_logits(const void* logits, int vocab_size) {
        // TODO: implement temperature-scaled sampling
        // For now: argmax
        return 0;
    }
};

} // namespace tenselerate
