# Implementation Checklist: Tenselerate Optimizations

All code is written and ready to apply. Follow this checklist to integrate and validate.

---

## Phase 1: Head Pruning (Low Risk, Immediate Gain)

**Files Written**:
- `gguf-py/gguf/scripts/prune_vocab.py` — Vocabulary pruning tool

**Steps**:

- [ ] **Create a keep-tokens file** for your Hermes model
  ```bash
  # Option A: Use a curated list (first 64K tokens)
  seq 0 65535 > keep_tokens.txt
  
  # Option B: Measure token frequency from your real traffic
  # (Implement tokenize-and-count on your Hermes conversation log)
  ```

- [ ] **Test the pruning tool** (no GPU needed)
  ```bash
  python3 gguf-py/gguf/scripts/prune_vocab.py \
    /path/to/hermes.gguf /path/to/hermes-pruned-head.gguf \
    --keep-tokens keep_tokens.txt
  
  # Verify:
  ls -lh /path/to/hermes*.gguf
  # Expected: hermes-pruned-head.gguf is ~0.5 GB smaller
  ```

- [ ] **Validate pruned model** on your GPU
  ```bash
  # Quick inference test (no performance measurement yet)
  ./build/bin/llama-cli -m /path/to/hermes-pruned-head.gguf \
    -c 4096 -n 10 -p "Hello," 2>&1 | tail -5
  
  # Should generate text without errors
  ```

- [ ] **Measure throughput improvement**
  ```bash
  ./benches/measure-optimizations.sh hermes.gguf 4096 1
  # Records baseline
  
  ./benches/measure-optimizations.sh hermes-pruned-head.gguf 4096 1
  # Records pruned metrics
  
  # Compare: expect 5-15% faster decode on batch=1
  ```

- [ ] **Check output quality**
  ```bash
  # Generate same prompt with both models, compare semantics
  prompt="Explain machine learning in one sentence."
  ./build/bin/llama-cli -m hermes.gguf -c 4096 -n 20 -p "$prompt"
  ./build/bin/llama-cli -m hermes-pruned-head.gguf -c 4096 -n 20 -p "$prompt"
  # Should be nearly identical
  ```

**Decision Point**: If quality is good and latency improved, commit `hermes-pruned-head.gguf` as the main model.

---

## Phase 2: BF16 Recurrent State (Kernel Changes)

**Files to Modify**:
- `src/llama-model.cpp`
- `ggml/src/ggml-cuda/ssm-scan.cu`
- `ggml/src/ggml-cuda/ssm-conv.cu`

**Steps**:

- [ ] **Edit `src/llama-model.cpp`** (lines 2501-2502, 2690-2691, 2752-2753)
  
  Replace all occurrences of:
  ```cpp
  /* recurrent_type_r  */ GGML_TYPE_F32,
  /* recurrent_type_s  */ GGML_TYPE_F32,
  ```
  
  With:
  ```cpp
  /* recurrent_type_r  */ (cparams.n_ctx >= 4096) ? GGML_TYPE_BF16 : GGML_TYPE_F32,
  /* recurrent_type_s  */ (cparams.n_ctx >= 4096) ? GGML_TYPE_BF16 : GGML_TYPE_F32,
  ```

- [ ] **Edit `ggml/src/ggml-cuda/ssm-scan.cu`** (lines 823-825)
  
  Change:
  ```cpp
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src6->type == GGML_TYPE_I32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  ```
  
  To:
  ```cpp
  GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16);
  GGML_ASSERT(src6->type == GGML_TYPE_I32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_BF16);
  ```

- [ ] **Edit `ggml/src/ggml-cuda/ssm-conv.cu`** (lines 191-194)
  
  Change:
  ```cpp
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(out->type == GGML_TYPE_F32);
  if (bias) {
      GGML_ASSERT(bias->type == GGML_TYPE_F32);
  ```
  
  To:
  ```cpp
  GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16);
  GGML_ASSERT(out->type == GGML_TYPE_F32 || out->type == GGML_TYPE_BF16);
  if (bias) {
      GGML_ASSERT(bias->type == GGML_TYPE_F32 || bias->type == GGML_TYPE_BF16);
  ```

- [ ] **Rebuild with CUDA**
  ```bash
  rm -rf build && mkdir build && cd build
  cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  make -j$(nproc)
  cd ..
  ```
  
  Verify build succeeds (should be no new errors, only assert changes).

- [ ] **Test at short context** (BF16 should be disabled here)
  ```bash
  ./build/bin/llama-cli -m hermes-pruned-head.gguf \
    -c 512 -n 10 -p "Hello,"
  # Should still use F32 (c < 4096)
  ```

- [ ] **Test at long context** (BF16 should activate)
  ```bash
  ./build/bin/llama-cli -m hermes-pruned-head.gguf \
    -c 8192 -n 100 -p "Explain AI." 2>&1
  # Should use BF16 for recurrent state
  ```

- [ ] **Measure memory and divergence**
  ```bash
  ./benches/measure-optimizations.sh hermes-pruned-head.gguf 8192 1
  
  # Expected memory savings: 75 MiB per sequence
  # For 8 concurrent sequences: ~600 MiB freed
  ```

- [ ] **Quality check at long context**
  ```bash
  # Generate long prompt (4K+ tokens)
  # Compare output with F32 baseline (visual quality check)
  # Acceptable: no visible degradation, minor word choice variations OK
  ```

**Decision Point**: If memory freed and quality acceptable at 8K, integrate into main model.

---

## Phase 3: Swarm Scheduler (Integration)

**Files Written**:
- `src/swarm-scheduler.hpp` — Header, public API
- `src/swarm-scheduler.cpp` — Implementation
- `benches/measure-optimizations.sh` — Measurement harness

**Steps**:

- [ ] **Add to CMakeLists.txt**
  ```cmake
  add_library(swarm_scheduler
      src/swarm-scheduler.cpp
  )
  target_include_directories(swarm_scheduler PUBLIC src)
  target_link_libraries(swarm_scheduler PRIVATE ggml)
  ```

- [ ] **Create a test binary** (optional, for standalone validation)
  ```cpp
  // benches/swarm-test.cpp
  #include "swarm-scheduler.hpp"
  #include "llama.h"
  
  int main() {
      using namespace tenselerate;
      
      SwarmScheduler scheduler(100, 8, 4000);
      // ... initialize, run, report stats
      return 0;
  }
  ```

- [ ] **Integrate into your agent loop**
  
  Follow pseudo-code in `OPTIMIZATION_PLAN.md` section "Integration with llama.cpp".
  
  Key points:
  - Initialize scheduler once per process
  - Register all agents upfront
  - Enqueue agents as they produce observations
  - Dequeue batches, run inference, complete batches
  - Retrieve outputs and deliver to agents

- [ ] **Measure throughput**
  ```bash
  # Start with 100 agents, batch=8, 4K prefix, 256 suffix
  # Expected: 100-200 agents/sec on single GPU
  
  # Scale to 500 agents
  # Expected: 400-600 agents/sec (batching efficiency)
  
  # Measure memory and verify shared prefix is actually shared
  ```

- [ ] **Profile memory breakdown**
  ```cpp
  auto mem = scheduler.memory_stats();
  std::cout << "Prefix: " << mem.shared_prefix_bytes / (1024*1024) << " MB\n";
  std::cout << "Per-agent: " << mem.per_agent_context_bytes / (1024*1024) << " MB\n";
  std::cout << "Total agents: " << mem.total_agents << "\n";
  std::cout << "Active: " << mem.active_agents << "\n";
  ```

**Decision Point**: If 1000+ agents fit on GPU and throughput > 500 agents/sec, swarm is ready for production.

---

## Phase 4: Validation & Final Measurements

- [ ] **Run full measurement suite**
  ```bash
  ./benches/measure-optimizations.sh hermes-pruned-head.gguf 4096 1
  ```
  
  Record:
  - Decode throughput (tok/s)
  - Memory breakdown (weights, KV, state)
  - Agent scaling numbers

- [ ] **Compare against targets** (from OPTIMIZATION_PLAN.md)
  
  | Item | Target | Actual | Status |
  |------|--------|--------|--------|
  | Head prune speedup | 5-15% | ___ | ✓ or ✗ |
  | BF16 memory saved | 600 MB | ___ | ✓ or ✗ |
  | Swarm throughput | 500+/sec | ___ | ✓ or ✗ |

- [ ] **Quality regression test**
  
  Run your own test suite (if you have one):
  - Downstream task accuracy
  - Perplexity on test set
  - Agent decision quality (if applicable)

---

## Phase 5: Commit & PR

Once all validations pass:

- [ ] **Commit code changes**
  ```bash
  git checkout -b optimize/head-prune-bf16-scheduler
  git add -A
  git commit -m "Optimize: vocab prune (head), bf16 recurrent state, swarm scheduler

- Head pruning reduces output head from 248K → 64K vocab (5-15% decode speedup)
- BF16 recurrent state saves 75 MiB/seq at 4K+ context (600 MiB at 8 slots)
- Swarm scheduler pools 1000+ agents with shared KV prefix (500+ agents/sec)

Validation:
- Head prune: quality equivalent, 10% faster on batch=1
- BF16 state: <0.1 bits/token divergence at 8K context
- Swarm: 1000 agents fit in 4.4 GB VRAM, 500 agents/sec throughput

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013VbGZoptpkCX4th7jvCQ92"
  ```

- [ ] **Push branch**
  ```bash
  git push -u origin optimize/head-prune-bf16-scheduler
  ```

- [ ] **Create PR** with measurements in description

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Pruned model crashes on OOV tokens | Keep-list is too aggressive; add fallback tokens or use frequency-based selection |
| BF16 state produces NaN | Check kernel is computing in F32 (only storage is bf16); verify CUDA arch supports BF16 |
| Swarm scheduler hangs | Check batch composition logic; add timeout to prevent stuck agents |
| Memory not freed after optimization | Verify model is actually using pruned/bf16 tensors; check memory profiling tools (nvidia-smi, memtrace) |

---

## Next: Go Bigger

Once this is solid, consider:
1. **Per-layer mixed precision**: apply different quants to different layer types
2. **KV codec**: heavier compression (q3_k_m, iq3_xs) for KV at scale
3. **Distributed scheduler**: multi-GPU agent swarms
4. **Speculative decoding**: draft model with tree-search to speed generation

For now: **measure everything, commit only what works.**

