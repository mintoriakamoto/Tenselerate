# Phase 3: Speculative Decoding

## Overview

**Speculative decoding** parallelizes inference via:
1. **Draft model** (Hermes-1B): Generate K tokens speculatively in parallel
2. **Verification**: Check draft tokens against target model (Hermes-1.7B)
3. **Acceptance**: Keep verified tokens, fallback on rejection

**Expected impact:** 1.5-2.5× latency reduction for single-token generation

---

## How It Works

```
Timeline (Baseline):
┌─────────────────────────────────────────┐
│ Target Model → Generate token 1         │ 85 ms
└─────────────────────────────────────────┘
│ Target Model → Generate token 2         │ 85 ms
└─────────────────────────────────────────┘
Total: 170 ms for 2 tokens

Timeline (Speculative):
┌─────────────────────┬───────────────────┐
│ Draft → 4 tokens    │ Target → Verify   │ 50 ms
└─────────────────────┴───────────────────┘
│ Accept 2-3 tokens, generate 1 new       │ 50 ms
└─────────────────────────────────────────┘
Total: 100 ms for 5-6 tokens (but we only wanted 2)
Speedup: 170 / 50 = 3.4× for the first batch
```

---

## Architecture

### Components

1. **SpeculativeDecoder** (speculative-decoding.hpp)
   - Manages draft generation and verification
   - Tracks acceptance rates and statistics
   - Configurable: max draft tokens, temperature, sampling method

2. **SpeculativeBatch**
   - Composes: [context | draft_1 | draft_2 | ... | draft_K]
   - Allows parallel evaluation of K draft tokens
   - Records verification indices for efficient GPU computation

3. **TokenVerifier**
   - Implements acceptance sampling: P(token|target) / P(token|draft)
   - Falls back to greedy (argmax) if probabilities unavailable
   - Supports temperature-scaled verification

4. **BatchVerifier** (for multi-agent)
   - Verifies drafts from multiple agents in parallel
   - Composes combined batch for efficient GPU utilization
   - Returns verified tokens per agent

### Models

- **Draft Model:** Hermes-1B (smaller, faster)
  - Generates 4-8 speculative tokens
  - ~2-3× faster than target model
  - May use lower precision (BF16, INT8)

- **Target Model:** Hermes-1.7B (full model)
  - Verifies draft tokens
  - Generates fallback if needed
  - Maintains output quality

---

## Implementation Status

### Phase 3.1: Core Infrastructure ✅ (Today)
- [x] SpeculativeDecoder class (orchestration logic)
- [x] Batch composition (efficient GPU layout)
- [x] Token verification (acceptance sampling)
- [x] Multi-agent support (for swarm scheduler)
- [x] Statistics tracking (acceptance rate, speedup)

### Phase 3.2: Integration (Next)
- [ ] Load dual models (draft + target) in llama.cpp context
- [ ] Model initialization and vocab mapping
- [ ] GPU memory management (both models simultaneously)
- [ ] Token stream composition and batching

### Phase 3.3: Inference Loop Integration
- [ ] Replace single-token generation with speculative decoding
- [ ] Handle acceptance/rejection in sampler
- [ ] Fallback on verification failure
- [ ] Async draft generation while target verifies

### Phase 3.4: Optimization & Tuning
- [ ] Profile draft vs. target inference time
- [ ] Tune max_draft_tokens for your setup
- [ ] Optimize batch composition for GPU efficiency
- [ ] Cache draft model KV states between generations

---

## Expected Performance

### Single Iteration
| Metric | Baseline | Speculative | Speedup |
|--------|----------|-------------|---------|
| Time per token | 85 ms | 50 ms | 1.7× |
| Tokens generated | 1 | 3-5 | 3-5× |
| Tokens output | 1 | 1.5-2.5 avg | 1.5-2.5× |

### Full Sequence (128 tokens)
| Metric | Before Phase 3 | After Phase 3 | Gain |
|--------|---|---|---|
| Time | 10.8 s (12 tok/s) | 5.5-7 s | 1.5-2× faster |
| Quality | Baseline | Same | No degradation |

---

## Configuration Options

```cpp
SpeculativeConfig config;
config.max_draft_tokens = 4;           // 4 speculative tokens per round
config.temperature_draft = 0.8f;       // Draft slightly more confident
config.temperature_target = 0.7f;      // Target more confident
config.use_acceptance_sampling = true; // Probabilistic verification
config.verbose = true;                 // Print statistics
```

### Tuning for RTX 3060 Ti

- **max_draft_tokens = 3-4**: Balance between speculation and latency
- **Draft model size**: Use quantized Hermes-1B (BF16 or INT8)
- **Temperature ratio**: draft/target = 1.15 (draft explores more)

---

## Multi-Agent Speculative Decoding (Swarm Mode)

For 1000s of concurrent agents:

```
Batch:
├─ Agent 1: 4 draft tokens
├─ Agent 2: 4 draft tokens
├─ Agent 3: 4 draft tokens
└─ ... 32 agents: 128 total draft tokens

Single GPU batch:
├ Draft Model: Generate all 128 in one inference ✅ (parallelizable)
└ Target Model: Verify all 128 in one inference ✅ (parallelizable)

Result: 32-128× GPU utilization improvement
```

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| src/speculative-decoding.hpp | 320 | Main orchestrator + config |
| src/speculative-decoding.cpp | 250 | Implementation + batch logic |
| PHASE3-PLAN.md | (this file) | Technical specification |

---

## Next Steps

1. **Integration into llama.cpp**
   - Load draft model (Hermes-1B)
   - Allocate separate KV cache
   - Replace main generate() loop

2. **Hardware Testing**
   - Profile draft vs. target timing
   - Measure actual speedup on RTX 3060 Ti
   - Tune max_draft_tokens parameter

3. **Optimization**
   - Async draft generation (while target verifies)
   - KV cache sharing where possible
   - Batch verification for multi-agent

---

## References

- [Accelerating LLM Inference with Speculative Decoding](https://arxiv.org/abs/2302.01318)
- [Parallel Speculative Decoding](https://arxiv.org/abs/2402.08831)
- Phase 1: Vocabulary pruning (reduces draft model overhead)
- Phase 2: Fused kernels (faster verification step)

---

## Status

**Phase 3.1 Complete:** Core infrastructure ready.

**Next:** Integration into inference loop (Phase 3.2-3.4).
