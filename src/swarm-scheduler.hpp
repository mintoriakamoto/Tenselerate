// swarm-scheduler.hpp – Multi-agent KV pooling orchestrator
// Manages 1000+ concurrent agents via shared prefix cache

#pragma once

#include <cstdint>
#include <vector>
#include <deque>
#include <map>
#include <chrono>
#include <atomic>
#include <memory>

namespace tenselerate {

enum class AgentState {
    IDLE,
    WAITING,
    ACTIVE,
    DONE
};

struct SharedPrefix {
    std::vector<int32_t> tokens;
    uint32_t n_tokens = 0;
    size_t cache_size_bytes = 0;
    void* kv_cache_k = nullptr;
    void* kv_cache_v = nullptr;
};

struct AgentContext {
    uint32_t agent_id;
    AgentState state = AgentState::IDLE;
    uint32_t priority = 0;
    uint32_t timeout_ms = 5000;
    std::vector<int32_t> tokens;
    std::vector<int32_t> output_tokens;
    float confidence = 0.0f;
    std::chrono::system_clock::time_point enqueued_at;
    std::chrono::system_clock::time_point completed_at;
};

struct InferenceBatch {
    uint32_t batch_id = 0;
    std::vector<uint32_t> agent_ids;
    std::vector<int32_t> composite_tokens;
    std::vector<uint32_t> token_offsets;
    std::vector<uint32_t> output_start_idx;
    uint32_t total_tokens = 0;
};

class SwarmScheduler {
public:
    struct MemoryStats {
        uint32_t shared_prefix_bytes = 0;
        uint32_t per_agent_context_bytes = 0;
        uint32_t total_agents = 0;
        uint32_t active_agents = 0;
    };

    struct SchedulerStats {
        uint64_t total_inferred = 0;
        uint32_t agents_waiting = 0;
        uint32_t agents_active = 0;
        float avg_latency_ms = 0.0f;
        float throughput_agents_per_sec = 0.0f;
    };

    SwarmScheduler(uint32_t max_agents, uint32_t max_batch_size, uint32_t prefix_ctx_size);

    void init_shared_prefix(const std::vector<int32_t>& prefix_tokens);
    void register_agent(uint32_t agent_id, uint32_t priority);
    void enqueue_agent(uint32_t agent_id, const std::vector<int32_t>& decision_tokens, uint32_t timeout_ms);

    std::unique_ptr<InferenceBatch> dequeue_batch();
    void complete_batch(const InferenceBatch& batch, const std::vector<std::vector<float>>& logits_per_agent);

    std::vector<int32_t> get_agent_output(uint32_t agent_id, float& confidence);

    MemoryStats memory_stats() const;
    SchedulerStats scheduler_stats() const;

private:
    uint32_t max_agents_;
    uint32_t max_batch_size_;
    SharedPrefix shared_prefix_;
    std::map<uint32_t, AgentContext> agents_;
    std::deque<uint32_t> idle_agents_;
    std::deque<uint32_t> waiting_agents_;
    std::vector<uint32_t> active_agents_;

    std::atomic<uint64_t> total_completed_{0};
    std::atomic<uint64_t> total_latency_ms_{0};

    void _sort_waiting_queue();
    uint32_t _estimate_batch_tokens(const std::vector<uint32_t>& agent_ids);
    void _compose_batch_tokens(const std::vector<uint32_t>& agent_ids, InferenceBatch& batch);
};

} // namespace tenselerate
