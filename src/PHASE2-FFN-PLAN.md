# Phase 2: Fused Kernels

## Fused FFN Kernel

Combines three operations into single CUDA kernel:
- Linear projection (hidden → 4×hidden)
- Activation function (GELU)
- Linear projection (4×hidden → hidden)

**Expected speedup:** 2-3×
**Register usage target:** <128 regs/thread (sm_86)
**Shared memory:** <96 KB per block

## Fused Sparse Attention

Implements strided + local attention in single kernel:
- Strided attention at stride=64
- Local attention window=256
- No separate dequant pass

**Expected speedup:** 1.8-2.0×
**Falsifiability:** attention_time ≥ 50% reduction at seq_len=64-256

## Integration

Optimization manager selects best implementation per layer:
- Fused if occupancy >60%
- Dense fallback if occupancy <40%
- Conservative choice for safety

## Measurement Plan

Pre-registered thresholds:
- FFN: ≥1.5× speedup (conservative)
- Attention: ≥1.5× speedup
- Both: cumulative 1.8× from Phase 1 baseline

Status: PLAUSIBLE (P=1, D=1, needs Phase 1 baseline)
