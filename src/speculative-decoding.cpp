// speculative-decoding.cpp – Implementation of speculative decoding
// Manages parallel draft generation and target verification

#include "speculative-decoding.hpp"
#include <algorithm>
#include <cstring>

namespace tenselerate {

// Batch composition for speculative decoding
// Combines: [previous_tokens | draft_1 | draft_2 | draft_3 | draft_4]
// Allows parallel evaluation of draft tokens against target model

struct SpeculativeBatch {
    std::vector<int32_t> token_stream;      // Composed token sequence
    std::vector<int32_t> draft_token_ids;   // Which tokens are drafts
    std::vector<int32_t> verification_indices;  // Indices to sample from target logits

    int n_draft_tokens = 0;
    int n_total_tokens = 0;

    void compose(
        const std::vector<int32_t>& previous_tokens,
        const std::vector<int32_t>& draft_tokens
    ) {
        token_stream.clear();
        draft_token_ids.clear();
        verification_indices.clear();

        // Add previous tokens (context)
        token_stream.insert(token_stream.end(),
                           previous_tokens.begin(),
                           previous_tokens.end());

        // Record indices for verification
        for (size_t i = 0; i < draft_tokens.size(); ++i) {
            verification_indices.push_back(token_stream.size());
            token_stream.push_back(draft_tokens[i]);
            draft_token_ids.push_back(draft_tokens[i]);
        }

        n_draft_tokens = draft_tokens.size();
        n_total_tokens = token_stream.size();
    }
};

// Acceptance sampling: decide if draft token matches target distribution
bool accept_draft_token(
    float target_prob,
    float draft_prob,
    float temperature_ratio = 1.0f
) {
    // Normalized acceptance probability
    // Accept if: target_prob / draft_prob > uniform(0, 1)
    float acceptance_prob = std::min(1.0f, (target_prob / (draft_prob + 1e-9f)) * temperature_ratio);

    // Return true if accepted (in practice: sample from Bernoulli)
    return acceptance_prob > 0.5f;  // Simplified: should use proper sampling
}

// Token verification: check if draft token matches target model's top choice
struct TokenVerifier {
    static bool verify(
        int32_t draft_token,
        const float* target_logits,
        int vocab_size,
        const float* draft_logits = nullptr,
        bool use_acceptance_sampling = true,
        float temperature_target = 1.0f,
        float temperature_draft = 1.0f
    ) {
        if (!target_logits || !draft_logits) {
            // Fall back to greedy: check if draft token is target's argmax
            int target_argmax = 0;
            float max_logit = target_logits[0];
            for (int i = 1; i < vocab_size; ++i) {
                if (target_logits[i] > max_logit) {
                    max_logit = target_logits[i];
                    target_argmax = i;
                }
            }
            return draft_token == target_argmax;
        }

        // Full acceptance sampling
        float target_prob = softmax_prob(target_logits, draft_token, vocab_size, temperature_target);
        float draft_prob = softmax_prob(draft_logits, draft_token, vocab_size, temperature_draft);

        return accept_draft_token(target_prob, draft_prob);
    }

private:
    static float softmax_prob(
        const float* logits,
        int token_id,
        int vocab_size,
        float temperature
    ) {
        // Simplified: just return exp(logit / T) / sum(exp(logits / T))
        // In production: use numerically stable softmax

        float scaled_logit = logits[token_id] / temperature;
        float sum_exp = 0.0f;

        for (int i = 0; i < vocab_size; ++i) {
            sum_exp += std::exp(logits[i] / temperature);
        }

        return std::exp(scaled_logit) / (sum_exp + 1e-9f);
    }
};

// Multi-agent speculative decoding for swarm scheduler
class SpeculativeAgentPool {
public:
    struct AgentDraftState {
        uint32_t agent_id;
        std::vector<int32_t> draft_tokens;
        std::vector<int32_t> verified_tokens;
        int n_accepted = 0;
        int n_rejected = 0;
    };

    void register_agent(uint32_t agent_id) {
        agent_drafts_[agent_id] = AgentDraftState{.agent_id = agent_id};
    }

    // Compose batch of all agents' draft tokens for parallel verification
    SpeculativeBatch compose_verification_batch(
        const std::vector<uint32_t>& agent_ids,
        const std::vector<std::vector<int32_t>>& draft_sequences
    ) {
        SpeculativeBatch batch;

        for (size_t i = 0; i < agent_ids.size(); ++i) {
            uint32_t aid = agent_ids[i];
            auto& drafts = draft_sequences[i];

            // Record this agent's draft tokens
            agent_drafts_[aid].draft_tokens = drafts;
        }

        // Total tokens to verify in parallel
        int total_drafts = 0;
        for (const auto& seq : draft_sequences) {
            total_drafts += seq.size();
        }

        batch.n_draft_tokens = total_drafts;
        return batch;
    }

    // Verify results and distribute to agents
    void distribute_verified(
        const std::vector<uint32_t>& agent_ids,
        const std::vector<std::vector<int32_t>>& verified_per_agent
    ) {
        for (size_t i = 0; i < agent_ids.size(); ++i) {
            auto& state = agent_drafts_[agent_ids[i]];
            state.verified_tokens = verified_per_agent[i];
            state.n_accepted += verified_per_agent[i].size();
        }
    }

private:
    std::unordered_map<uint32_t, AgentDraftState> agent_drafts_;
};

// Efficient batch verification on GPU
// Allows verifying N draft tokens from M agents in parallel
class BatchVerifier {
public:
    struct BatchConfig {
        int max_batch_size = 256;  // Max tokens per verification batch
        int max_agents = 32;       // Max agents in parallel
        bool use_cuda = true;
    };

    BatchVerifier(const BatchConfig& config = BatchConfig()) : config_(config) {}

    // Verify all draft tokens in a batch
    std::vector<std::vector<int32_t>> verify_batch(
        const std::vector<std::vector<int32_t>>& draft_tokens,
        const std::vector<void*>& target_logits,  // Per-draft logits
        const std::vector<void*>& draft_logits,   // For acceptance sampling
        int vocab_size
    ) {
        std::vector<std::vector<int32_t>> verified;

        for (size_t i = 0; i < draft_tokens.size(); ++i) {
            std::vector<int32_t> agent_verified;

            for (int32_t token : draft_tokens[i]) {
                bool accept = TokenVerifier::verify(
                    token,
                    static_cast<const float*>(target_logits[i]),
                    vocab_size,
                    static_cast<const float*>(draft_logits[i]),
                    true,
                    1.0f, 1.0f
                );

                if (accept) {
                    agent_verified.push_back(token);
                } else {
                    break;  // Stop on first rejection
                }
            }

            verified.push_back(agent_verified);
        }

        return verified;
    }

private:
    BatchConfig config_;
};

} // namespace tenselerate
