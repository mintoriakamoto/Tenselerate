# Tenselerate Optimization Roadmap

## Mission
Maximize Hermes 1.7B inference throughput on RTX 3060 Ti (12GB VRAM) via aggressive optimization across kernel, memory, and architecture layers.

## Optimization Phases

### Phase 1: Foundational Compression ✅ IN PROGRESS
**Goal:** Reduce model size and memory footprint without kernel changes
- [x] Vocabulary pruning: 248K → 64K tokens (−0.5 GB/token read)
- [x] BF16 recurrent state quantization (−75 MiB/seq for GDN states)
- [x] Swarm scheduler: multi-agent KV pooling (1000+ concurrent agents)
- [x] C++ implementation + test suite
**Status:** Code merged to `optimize/vocab-bf16-swarm`
**Blocked on:** Hardware execution (need RTX 3060 Ti measurements)

### Phase 2: Kernel Fusion (NEXT)
**Goal:** Reduce kernel launch overhead and improve cache locality
- [ ] Fused FFN kernel: `hidden → FFN_up → GELU → FFN_down → hidden` (1 kernel instead of 3)
  - CUDA: ~450 lines
  - Target: 2-3× throughput vs. separate kernels
  - Timeline: 3-4 days
- [ ] Fused sparse attention: local + strided patterns
  - CUDA: ~200 lines
  - Target: 40-50% reduction in attention compute
  - Timeline: 2-3 days
**Branch:** `optimize/phase2-kernels`

### Phase 3: Speculative Decoding
**Goal:** Reduce latency via draft-target verification
- [ ] Hermes-1B draft + 1.7B target model orchestration
- [ ] Token verification state machine
- [ ] Batch composition for mixed draft/target
**Branch:** `optimize/phase3-speculative`

### Phase 4: Advanced Quantization
**Goal:** Aggressive weight + activation quantization with minimal quality loss
- [ ] Ternary quantization: weights as {−α, 0, +α}
- [ ] Mixed-precision calibration by layer sensitivity
- [ ] Low-rank correction for critical layers
**Branch:** `optimize/phase4-quantization`

### Phase 5: Multi-Agent Batching
**Goal:** Maximize GPU utilization for swarm workloads
- [ ] Prefix cache optimization
- [ ] KV cache sharing strategies
- [ ] Batch composition heuristics
**Branch:** `optimize/phase5-swarm`

---

## Success Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Model size | ~3.5 GB | ~1.5 GB |
| Decode throughput | ~12 tok/s | ~40+ tok/s |
| Memory per token | 0.5 GB | 0.2 GB |
| Multi-agent capacity | 10–50 | 1000+ |

---

## Development Workflow

1. **Branch naming:** `optimize/phase<N>-<feature>`
2. **Commits:** Atomic changes with clear messages
3. **CI/CD:** Build-cuda-ubuntu.yml enforces CUDA compliance
4. **Review:** Mandatory for all PRs (use PR template)
5. **Benchmarks:** Measure before/after with benches/measure-optimizations.sh

---

## Critical Files

- `src/llama.cpp` – inference loop
- `ggml-cuda.cu` – CUDA kernels
- `src/swarm-scheduler.{hpp,cpp}` – multi-agent orchestration
- `gguf-py/gguf/scripts/prune_vocab.py` – vocabulary pruning
- `benches/measure-optimizations.sh` – benchmarking harness

---

## Known Constraints

- **VRAM:** 12 GB (hard limit on RTX 3060 Ti)
- **Model:** Hermes 1.7B (hybrid recurrent transformer with GDN layers)
- **Tokenizer:** 248K vocabulary (base)
- **Batch size:** ~32 agents for swarm (limited by prefix cache)

---

## References

- [OPTIMIZATION_PLAN.md](./OPTIMIZATION_PLAN.md) – Technical specs
- [IMPLEMENTATION_CHECKLIST.md](./IMPLEMENTATION_CHECKLIST.md) – Validation steps
- [AGENTS.md](./AGENTS.md) – Project governance (AI/agent usage)
