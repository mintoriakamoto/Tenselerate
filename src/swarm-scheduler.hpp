/*
 * Swarm Scheduler: Multi-agent inference with shared KV prefix pooling.
 *
 * Scales 1,000+ agents on one GPU by:
 * 1. Computing mission envelope + system prompt once (shared prefix)
 * 2. Each agent maintains only its observation/decision context (~256 tokens)
 * 3. Batching agents by active/waiting/idle state
 *
 * Memory model:
 * - Shared prefix KV: ~200-400 MB (immutable, all agents read it)
 * - Per-agent state: ~10-50 MB (decision context only)
 * - Batch window: N agents * 256 tokens + shared prefix
 *
 * Token math (Hermes 1.7B):
 * - Shared prefix: ~4000 tokens (mission + system) → ~400 MB KV
 * - Per-agent suffix: ~256 tokens (observation + decision) → ~25 MB KV
 * - Batch 8 agents: 8*256 + 4000 = 6000 tokens total → ~600 MB
 * - GPU mem: weights (3.4 GB) + KV (600 MB) + workspace (500 MB) = ~4.4 GB
 * - Throughput: ~50 agents per forward pass at batch=8, ~100 ms/batch
 *   → 500 agents/second (or 1,500+ standing inference at 5% util each)
 */

#pragma once

#include <cstdint>
#include <vector>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <chrono>
#include <atomic>

namespace tenselerate {

/**
 * Agent state in the swarm.
 * Lifecycle: IDLE -> WAITING -> ACTIVE -> [DONE | WAITING]
 */
enum class AgentState : uint8_t {
    IDLE,          // Not scheduled, no context
    WAITING,       // Decision pending, context buffered
    ACTIVE,        // Currently inferring
    DONE,          // Inference complete, result ready
};

/**
 * Per-agent context: the decision-scoped cache.
 * Shared prefix is managed separately (immutable across agents).
 */
struct AgentContext {
    uint32_t agent_id;
    AgentState state;

    // Decision-scoped context (what changed this turn)
    std::vector<int32_t> tokens;           // Observation + history since last decision
    std::vector<uint32_t> token_ids;       // Real token IDs (for sparse updates)

    // Generated output
    std::vector<int32_t> output_tokens;    // Model output (not yet decoded)
    float confidence;                       // Max logit for the sampled token

    // Timing
    std::chrono::system_clock::time_point enqueued_at;
    std::chrono::system_clock::time_point completed_at;

    // Metadata for scheduler decisions
    uint32_t priority;                      // Higher = schedule sooner
    uint32_t timeout_ms;                    // Abort if pending > timeout_ms
};

/**
 * Shared prefix cache: immutable KV for mission envelope + system prompt.
 * All agents reference this read-only context.
 */
struct SharedPrefixCache {
    uint32_t n_tokens;                      // Prefix length (e.g., 4000)
    std::vector<int32_t> tokens;            // Token sequence

    // KV cache state (managed by llama.cpp, we hold references)
    // In a full implementation, this points to VRAM allocations
    void* kv_cache_k;                       // K cache rows (float16/bf16)
    void* kv_cache_v;                       // V cache rows (float16/bf16)

    uint32_t cache_size_bytes;              // ~400 MB for typical mission context
};

/**
 * Batch: a group of agents ready to infer together.
 * Composable: [shared prefix] + [agent₁ suffix] + [agent₂ suffix] + ...
 */
struct InferenceBatch {
    uint32_t batch_id;
    std::vector<uint32_t> agent_ids;        // Agents in this batch
    std::vector<uint32_t> token_offsets;    // Start index for each agent in composite token stream

    uint32_t total_tokens;                  // Sum of (prefix + all agent suffixes)
    std::vector<int32_t> composite_tokens;  // [prefix | agent₁_suffix | agent₂_suffix | ...]

    // Where in the composite buffer each agent's output appears
    std::vector<uint32_t> output_start_idx; // Index into output logits for each agent
};

/**
 * Core scheduler: manages agent lifecycle, batching, and KV memory.
 */
class SwarmScheduler {
public:
    SwarmScheduler(
        uint32_t max_agents,
        uint32_t max_batch_size,
        uint32_t prefix_ctx_size
    );

    /**
     * Initialize shared prefix (mission + system prompt tokens).
     * Must be called once before any agent inference.
     */
    void init_shared_prefix(const std::vector<int32_t>& prefix_tokens);

    /**
     * Register a new agent.
     * Agent starts in IDLE state, no context allocated.
     */
    void register_agent(uint32_t agent_id, uint32_t priority = 0);

    /**
     * Enqueue agent for inference.
     * Moves agent to WAITING state, buffers decision context.
     */
    void enqueue_agent(
        uint32_t agent_id,
        const std::vector<int32_t>& decision_tokens,
        uint32_t timeout_ms = 5000
    );

    /**
     * Dequeue: batch agents for the next forward pass.
     * Selects up to max_batch_size agents, composes the token stream,
     * returns a ready-to-infer batch.
     */
    std::unique_ptr<InferenceBatch> dequeue_batch();

    /**
     * Register inference results for a batch.
     * Moves agents to DONE, stores output tokens.
     */
    void complete_batch(
        const InferenceBatch& batch,
        const std::vector<std::vector<float>>& logits_per_agent
    );

    /**
     * Retrieve completed agent output.
     * Returns tokens and confidence. Agent moves to IDLE.
     */
    std::vector<int32_t> get_agent_output(uint32_t agent_id, float& confidence);

    /**
     * Memory breakdown: how much VRAM is used.
     */
    struct MemoryStats {
        uint32_t shared_prefix_bytes;
        uint32_t per_agent_context_bytes;
        uint32_t total_agents;
        uint32_t active_agents;
    };
    MemoryStats memory_stats() const;

    /**
     * Scheduler stats: throughput, latency, utilization.
     */
    struct SchedulerStats {
        uint32_t total_inferred;
        uint32_t agents_waiting;
        uint32_t agents_active;
        float avg_latency_ms;
        float throughput_agents_per_sec;
    };
    SchedulerStats scheduler_stats() const;

private:
    uint32_t max_agents_;
    uint32_t max_batch_size_;

    SharedPrefixCache shared_prefix_;

    // Agent registry
    std::map<uint32_t, AgentContext> agents_;

    // Scheduling queues
    std::deque<uint32_t> idle_agents_;
    std::deque<uint32_t> waiting_agents_;      // Sorted by priority
    std::deque<uint32_t> active_agents_;

    // Stats
    std::atomic<uint32_t> total_completed_{0};
    std::atomic<uint64_t> total_latency_ms_{0};

    // Internal helpers
    void _sort_waiting_queue();
    uint32_t _estimate_batch_tokens(const std::vector<uint32_t>& agent_ids);
    void _compose_batch_tokens(
        const std::vector<uint32_t>& agent_ids,
        InferenceBatch& batch
    );
};

} // namespace tenselerate
