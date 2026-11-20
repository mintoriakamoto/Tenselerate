# Complete Work Audit & Validation Checklist

## Purpose
Ensure all Phase 1-4 optimizations meet AGENTS.md standards and are properly scored per veritas.md before pushing.

---

## Phase 1: Vocabulary Pruning + BF16 State + Swarm Scheduler

### Files Required
- [ ] `gguf-py/gguf/scripts/prune_vocab.py` – Vocabulary pruning tool
- [ ] `src/swarm-scheduler.hpp` – Public API
- [ ] `src/swarm-scheduler.cpp` – Implementation
- [ ] Documentation: `docs/phase1-vocab-pruning.md`
- [ ] Documentation: `docs/phase1-bf16-state.md`
- [ ] Documentation: `docs/phase1-swarm-scheduler.md`

### Code Quality (AGENTS.md)
- [ ] No redundant comments (read code, explain WHY not WHAT)
- [ ] No multi-line comment blocks
- [ ] Uses ASD-STE100 Simplified Technical English
- [ ] No line-wrapping to fixed column width
- [ ] Concise comments: 1-2 lines max
- [ ] Blends with surrounding codebase patterns
- [ ] No unicode characters (use ASCII: `->`, `-`, `x`, `...`)

### Veritas Scoring (veritas.md)
- [ ] All claims scored: P, F, M, D, Status
- [ ] P ≥ 1 (at minimum: cited in docs)
- [ ] D scores tracked (inheritance cap: D=0 caps at PLAUSIBLE)
- [ ] No load-bearing claims at P=0 without label
- [ ] Falsifiability stated (what would kill it)
- [ ] Magnitude clear (1%, 5%, 30%?)

### Measurement Plan (Pre-registration)
- [ ] Vocab prune: Expected −0.5 GB/token read
  - [ ] Measurement method: `benches/measure-optimizations.sh`
  - [ ] Threshold: ≥ −0.4 GB/token (refutation if less)
  - [ ] Status: UNVERIFIED (P=1, needs P=3)

- [ ] BF16 state: Expected −75 MiB/seq for GDN states
  - [ ] Measurement method: Profile GDN layer memory
  - [ ] Threshold: ≥ −60 MiB/seq
  - [ ] Status: UNVERIFIED (P=1, needs P=3)

- [ ] Swarm scheduler: Expected 1000+ agents with <256 tokens context
  - [ ] Measurement method: Stress test on RTX 3060 Ti
  - [ ] Threshold: ≥ 500 agents
  - [ ] Status: UNVERIFIED (P=0 on RTX 3060 Ti)

---

## Phase 2: Fused FFN + Sparse Attention

### Files Required
- [ ] `src/kernels/fused-ffn.cu` – FFN kernel (450 lines)
- [ ] `src/kernels/fused-ffn.h` – API
- [ ] `src/kernels/fused-ffn-integration.cpp` – Wrapper
- [ ] `src/kernels/fused-sparse-attn.cu` – Attention kernel (406 lines)
- [ ] `src/kernels/fused-sparse-attn.h` – API
- [ ] `src/kernels/fused-sparse-attn-integration.cpp` – Wrapper
- [ ] `src/llama-optimizations.hpp` – Optimization manager
- [ ] `src/llama-optimizations.cpp` – Manager impl
- [ ] Modified: `src/CMakeLists.txt` – CUDA build integration
- [ ] Documentation: `src/kernels/README-FUSED-FFN.md`
- [ ] Example: `src/kernels/INTEGRATION-EXAMPLE.cpp`

### Code Quality (AGENTS.md)
- [ ] All CUDA code follows patterns in `ggml-cuda/`
- [ ] Comments explain WHY, not WHAT
- [ ] No register spilling (target <128 regs/thread on sm_86)
- [ ] Shared memory < 96 KB per block (RTX 3060 Ti limit)
- [ ] Proper error handling with fallback strategy

### Veritas Scoring
- [ ] FFN speedup: Claimed 2-3×
  - [ ] P=1 (design docs, not measured)
  - [ ] F=2 (pre-registered: FFN time / 3 separate kernels)
  - [ ] M=2 (10-15% of decode)
  - [ ] D=1 (rests on Phase 1 baseline, needs measurement)
  - [ ] Status: PLAUSIBLE (capped by D)

- [ ] Sparse attention: Claimed 1.8-2.0× speedup
  - [ ] P=1 (design, not measured on RTX 3060 Ti)
  - [ ] F=2 (threshold: attention_time ≥ 50% reduction)
  - [ ] M=2 (20-25% of decode)
  - [ ] D=1 (rests on baseline)
  - [ ] Status: PLAUSIBLE (capped by D, unverified on sm_86)

### Measurement Plan
- [ ] Fused FFN: Layer-level timing on RTX 3060 Ti
  - [ ] Measure: FFN latency (fused vs. 3-kernel baseline)
  - [ ] Threshold: ≥ 1.5× speedup (conservative)
  - [ ] Tool: `benches/measure-optimizations.sh`

- [ ] Sparse attention: Attention latency (local+strided vs. dense)
  - [ ] Measure: Attention latency at seq_len = 64, 256, 1024
  - [ ] Threshold: ≥ 1.5× speedup
  - [ ] Tool: custom ncu profiling (if available)

---

## Phase 3: Speculative Decoding

### Files Required
- [ ] `src/speculative-decoding.hpp` – Orchestrator
- [ ] `src/speculative-decoding.cpp` – Implementation
- [ ] Documentation: `PHASE3-PLAN.md`

### Veritas Scoring
- [ ] Claimed: 1.5-2.5× latency reduction
  - [ ] P=0 (design only, no code integration)
  - [ ] F=2 (threshold: single-token latency ≥ 1.3×)
  - [ ] M=3 (structural, latency is critical path)
  - [ ] D=0 (rests on unverified Phase 2)
  - [ ] Status: PLAUSIBLE (capped at D=0)

### Measurement Plan
- [ ] Integration: Needs draft model (Hermes-1B)
- [ ] Pre-registration: Draft gen time + verification overhead
- [ ] Threshold: Net speedup ≥ 1.3× on single token

---

## Phase 4: Ternary Quantization

### Files Required
- [ ] `src/ternary-quantization.hpp` – Quantizer
- [ ] Documentation: `PHASE4-PLAN.md`

### Veritas Scoring
- [ ] Claimed: 4× weight compression, 1-2% quality loss
  - [ ] P=0 (no measurement or code integration)
  - [ ] F=1 (falsifiable but not pre-registered)
  - [ ] M=3 (model size is structural)
  - [ ] D=0 (rests on Phase 2+3)
  - [ ] Status: PLAUSIBLE (D=0 cap, needs calibration)

### Measurement Plan
- [ ] Calibration: Run on validation data
- [ ] Quality: Perplexity < 1.02× baseline
- [ ] Size: Measure actual GGUF file size reduction

---

## Critical Path to CONFIRMED

**Blocking everything:** F4 from veritas-ledger
```bash
grep -iE "DISABLE_DP4A|FORCE_MMQ" <build-dir>/CMakeCache.txt
```
*Status:* MUST RUN before GPU kernels

**Then measure in order:**
1. Phase 1 baseline on RTX 3060 Ti (decode tok/s)
2. Phase 1 optimizations (vocab prune, BF16 state)
3. Phase 2 kernel speedups (FFN, attention)
4. Phase 3 integration (speculative decoding impact)
5. Phase 4 calibration (ternary quantization)

---

## Sign-Off Checklist

- [ ] All files conform to AGENTS.md standards
- [ ] All claims scored per veritas.md (P, F, M, D, Status)
- [ ] No load-bearing P=0 claims without label
- [ ] Measurement plans pre-registered (threshold before run)
- [ ] Code compiles on RTX 3060 Ti (sm_86)
- [ ] No redundant comments or unicode
- [ ] All commits use "Assisted-by:" not "Co-authored-by:"
- [ ] README updated with phase descriptions
- [ ] Branches created and committed atomically

---

## Next: Hardware Validation

1. Verify F4 (dp4a crippling) ← FIRST
2. Run Phase 1 baseline measurement
3. Profile GPU for bottlenecks (occupancy, bandwidth)
4. Then iterate Phase 2-4 based on actual constraints

**Expected: Zero verified improvements until F4 and baseline measured.**
