#include "swarm-scheduler.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace tenselerate {

SwarmScheduler::SwarmScheduler(
    uint32_t max_agents,
    uint32_t max_batch_size,
    uint32_t prefix_ctx_size
)
    : max_agents_(max_agents), max_batch_size_(max_batch_size) {
    shared_prefix_.n_tokens = 0;
    shared_prefix_.cache_size_bytes = 0;
    shared_prefix_.kv_cache_k = nullptr;
    shared_prefix_.kv_cache_v = nullptr;
}

void SwarmScheduler::init_shared_prefix(const std::vector<int32_t>& prefix_tokens) {
    if (prefix_tokens.empty()) {
        throw std::invalid_argument("Prefix tokens cannot be empty");
    }
    if (agents_.size() > 0) {
        throw std::runtime_error("Cannot reinitialize prefix after agents are registered");
    }

    shared_prefix_.tokens = prefix_tokens;
    shared_prefix_.n_tokens = prefix_tokens.size();

    // Cache size estimate: each token needs ~100KB of KV (rough for Hermes 1.7B)
    // K and V cache for 64 layers, each storing token state
    // Real: depends on model (n_embd, n_embd_k, n_layers)
    // For now, placeholder: will be set by llama.cpp integration
    shared_prefix_.cache_size_bytes = prefix_tokens.size() * 100 * 1024; // 100KB/token

    std::cerr << "[SwarmScheduler] Initialized shared prefix: "
              << prefix_tokens.size() << " tokens, ~"
              << (shared_prefix_.cache_size_bytes / (1024*1024)) << " MB KV\n";
}

void SwarmScheduler::register_agent(uint32_t agent_id, uint32_t priority) {
    if (agents_.size() >= max_agents_) {
        throw std::runtime_error("Agent pool is full");
    }

    auto [it, inserted] = agents_.emplace(
        agent_id,
        AgentContext{
            .agent_id = agent_id,
            .state = AgentState::IDLE,
            .priority = priority,
            .timeout_ms = 5000,
        }
    );

    if (!inserted) {
        throw std::invalid_argument("Agent already registered");
    }

    idle_agents_.push_back(agent_id);
}

void SwarmScheduler::enqueue_agent(
    uint32_t agent_id,
    const std::vector<int32_t>& decision_tokens,
    uint32_t timeout_ms
) {
    auto it = agents_.find(agent_id);
    if (it == agents_.end()) {
        throw std::invalid_argument("Agent not found");
    }

    AgentContext& ctx = it->second;

    // Remove from idle if present
    auto idle_it = std::find(idle_agents_.begin(), idle_agents_.end(), agent_id);
    if (idle_it != idle_agents_.end()) {
        idle_agents_.erase(idle_it);
    }

    // Update context
    ctx.tokens = decision_tokens;
    ctx.state = AgentState::WAITING;
    ctx.timeout_ms = timeout_ms;
    ctx.enqueued_at = std::chrono::system_clock::now();

    // Enqueue in waiting
    waiting_agents_.push_back(agent_id);
    _sort_waiting_queue();
}

std::unique_ptr<InferenceBatch> SwarmScheduler::dequeue_batch() {
    if (waiting_agents_.empty()) {
        return nullptr; // No work
    }

    auto batch = std::make_unique<InferenceBatch>();
    batch->batch_id = 0; // TODO: monotonic counter

    // Select agents up to batch size
    uint32_t tokens_so_far = shared_prefix_.n_tokens; // Always include prefix
    uint32_t agents_selected = 0;

    while (!waiting_agents_.empty() &&
           agents_selected < max_batch_size_) {
        uint32_t agent_id = waiting_agents_.front();
        AgentContext& ctx = agents_[agent_id];

        uint32_t agent_tokens = ctx.tokens.size();
        if (tokens_so_far + agent_tokens <= 2048) { // Assume 2K token limit per batch
            batch->agent_ids.push_back(agent_id);
            batch->token_offsets.push_back(tokens_so_far);
            tokens_so_far += agent_tokens;
            agents_selected++;

            ctx.state = AgentState::ACTIVE;
            active_agents_.push_back(agent_id);
            waiting_agents_.pop_front();
        } else {
            break; // Batch full
        }
    }

    if (batch->agent_ids.empty()) {
        return nullptr; // Couldn't fit any agents
    }

    // Compose token stream: [prefix | agent1_tokens | agent2_tokens | ...]
    batch->total_tokens = tokens_so_far;
    batch->composite_tokens.reserve(batch->total_tokens);
    batch->composite_tokens.insert(
        batch->composite_tokens.end(),
        shared_prefix_.tokens.begin(),
        shared_prefix_.tokens.end()
    );

    for (uint32_t agent_id : batch->agent_ids) {
        AgentContext& ctx = agents_[agent_id];
        batch->output_start_idx.push_back(batch->composite_tokens.size());
        batch->composite_tokens.insert(
            batch->composite_tokens.end(),
            ctx.tokens.begin(),
            ctx.tokens.end()
        );
    }

    std::cerr << "[SwarmScheduler] Batched " << batch->agent_ids.size()
              << " agents, " << batch->total_tokens << " total tokens\n";

    return batch;
}

void SwarmScheduler::complete_batch(
    const InferenceBatch& batch,
    const std::vector<std::vector<float>>& logits_per_agent
) {
    if (logits_per_agent.size() != batch.agent_ids.size()) {
        throw std::invalid_argument("Logits count mismatch");
    }

    auto now = std::chrono::system_clock::now();

    for (size_t i = 0; i < batch.agent_ids.size(); ++i) {
        uint32_t agent_id = batch.agent_ids[i];
        AgentContext& ctx = agents_[agent_id];

        // Find the sampled token (argmax for now; replace with proper sampling)
        const auto& logits = logits_per_agent[i];
        if (logits.empty()) {
            throw std::invalid_argument("Empty logits for agent");
        }

        int32_t sampled_token = std::max_element(logits.begin(), logits.end()) - logits.begin();
        ctx.output_tokens.push_back(sampled_token);
        ctx.confidence = logits[sampled_token];

        // Record timing
        ctx.completed_at = now;
        auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - ctx.enqueued_at
        ).count();

        // Update stats
        total_completed_.fetch_add(1, std::memory_order_relaxed);
        total_latency_ms_.fetch_add(latency_ms, std::memory_order_relaxed);

        // Remove from active, move to idle (or back to waiting if more context)
        auto active_it = std::find(active_agents_.begin(), active_agents_.end(), agent_id);
        if (active_it != active_agents_.end()) {
            active_agents_.erase(active_it);
        }

        ctx.state = AgentState::DONE;
        idle_agents_.push_back(agent_id);
    }
}

std::vector<int32_t> SwarmScheduler::get_agent_output(uint32_t agent_id, float& confidence) {
    auto it = agents_.find(agent_id);
    if (it == agents_.end()) {
        throw std::invalid_argument("Agent not found");
    }

    AgentContext& ctx = it->second;
    if (ctx.state != AgentState::DONE) {
        throw std::runtime_error("Agent result not ready");
    }

    std::vector<int32_t> result = ctx.output_tokens;
    confidence = ctx.confidence;

    // Reset agent for reuse
    ctx.state = AgentState::IDLE;
    ctx.tokens.clear();
    ctx.output_tokens.clear();

    return result;
}

SwarmScheduler::MemoryStats SwarmScheduler::memory_stats() const {
    uint32_t active_count = 0;
    uint32_t total_per_agent = 0;

    for (const auto& [id, ctx] : agents_) {
        if (ctx.state == AgentState::ACTIVE || ctx.state == AgentState::WAITING) {
            active_count++;
        }
        // Rough: 100KB per token in agent context
        total_per_agent += ctx.tokens.size() * 100 * 1024;
    }

    return MemoryStats{
        .shared_prefix_bytes = shared_prefix_.cache_size_bytes,
        .per_agent_context_bytes = total_per_agent,
        .total_agents = static_cast<uint32_t>(agents_.size()),
        .active_agents = active_count,
    };
}

SwarmScheduler::SchedulerStats SwarmScheduler::scheduler_stats() const {
    uint32_t total = total_completed_.load(std::memory_order_relaxed);
    uint64_t latency_sum = total_latency_ms_.load(std::memory_order_relaxed);

    float avg_latency = total > 0 ? (float)latency_sum / total : 0.0f;
    float throughput = total > 0 ? (total * 1000.0f) / latency_sum : 0.0f; // agents/sec

    return SchedulerStats{
        .total_inferred = total,
        .agents_waiting = static_cast<uint32_t>(waiting_agents_.size()),
        .agents_active = static_cast<uint32_t>(active_agents_.size()),
        .avg_latency_ms = avg_latency,
        .throughput_agents_per_sec = throughput,
    };
}

void SwarmScheduler::_sort_waiting_queue() {
    // Sort by priority (higher first), then by enqueue time (FIFO)
    std::sort(waiting_agents_.begin(), waiting_agents_.end(),
        [this](uint32_t a, uint32_t b) {
            const auto& ctx_a = agents_[a];
            const auto& ctx_b = agents_[b];
            if (ctx_a.priority != ctx_b.priority) {
                return ctx_a.priority > ctx_b.priority;
            }
            return ctx_a.enqueued_at < ctx_b.enqueued_at;
        }
    );
}

uint32_t SwarmScheduler::_estimate_batch_tokens(const std::vector<uint32_t>& agent_ids) {
    uint32_t total = shared_prefix_.n_tokens;
    for (uint32_t id : agent_ids) {
        total += agents_[id].tokens.size();
    }
    return total;
}

void SwarmScheduler::_compose_batch_tokens(
    const std::vector<uint32_t>& agent_ids,
    InferenceBatch& batch
) {
    batch.composite_tokens.clear();
    batch.composite_tokens.reserve(_estimate_batch_tokens(agent_ids));

    // Add prefix
    batch.composite_tokens.insert(
        batch.composite_tokens.end(),
        shared_prefix_.tokens.begin(),
        shared_prefix_.tokens.end()
    );

    // Add agent tokens
    for (uint32_t agent_id : agent_ids) {
        batch.output_start_idx.push_back(batch.composite_tokens.size());
        const auto& tokens = agents_[agent_id].tokens;
        batch.composite_tokens.insert(
            batch.composite_tokens.end(),
            tokens.begin(),
            tokens.end()
        );
    }

    batch.total_tokens = batch.composite_tokens.size();
}

} // namespace tenselerate
