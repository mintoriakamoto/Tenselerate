# Phase 4: Ternary Quantization

## Overview

**Ternary quantization** encodes weights as **{-α, 0, +α}** per-layer, achieving **4× compression** over float32.

Combined with LoRA fine-tuning for minimal quality loss.

---

## How It Works

### Standard Float32
```
Weight: 0.453    → 4 bytes
Weight: -0.891   → 4 bytes
Weight: 0.012    → 4 bytes
Storage: 3 weights = 12 bytes
```

### Ternary Encoding
```
Weight: 0.453    → +α (code 2)  → 2 bits
Weight: -0.891   → -α (code 1)  → 2 bits
Weight: 0.012    → 0   (code 0)  → 2 bits
Storage: 3 weights = 6 bits = 1 byte (4× compression)
```

### LoRA Correction
```
Output = Ternary(W) @ X + LoRA_A @ LoRA_B^T @ X
         └─ Fast, quantized        └─ Low-rank correction
           
Result: Near-original accuracy with 4× compression + small correction overhead
```

---

## Compression Summary

| Component | Size (Float32) | Size (Ternary) | Compression |
|-----------|----------|----------|---|
| **Weights** | 3.5 GB | 0.875 GB | 4× |
| **Scales (α)** | — | 50 MB | — |
| **LoRA (rank 8)** | — | 200 MB | — |
| **Total (Phase 1+4)** | 3.5 GB | **1.1 GB** | **3.2×** |

---

## Architecture

### Components

1. **TernaryQuantizer**
   - Calibrate: Find optimal α per layer
   - Encode: Map weights to {-α, 0, +α}
   - Decode: Reconstruct for reference
   - Measure: Compute quantization error (RMSE)

2. **LoRACorrection**
   - Low-rank decomposition: [n_in, rank] × [rank, n_out]
   - Rank: 8-16 (tunable)
   - Applied per-layer to recover precision

3. **LayerQuantizer**
   - Sensitivity-aware: Important layers get less compression
   - Layer scores: Gradient-based sensitivity analysis
   - Per-layer α adjustment

### Layer Sensitivity

Critical layers (first few transformer blocks, output head):
- Higher sensitivity → Larger α → Less compression
- Lower sensitivity → Smaller α → More compression

---

## Implementation Status

### Phase 4.1: Core Infrastructure ✅ (Today)
- [x] Ternary encoding/decoding (2 bits per weight)
- [x] Scale calibration (95th percentile for α)
- [x] LoRA correction framework
- [x] Error measurement (RMSE)
- [x] Multi-layer orchestrator
- [x] Sensitivity-aware quantization

### Phase 4.2: Calibration (Next)
- [ ] Sensitivity analysis (gradient-based)
- [ ] Calibration dataset selection
- [ ] Per-layer α optimization
- [ ] Quality validation

### Phase 4.3: LoRA Fine-tuning
- [ ] SVD-based low-rank decomposition
- [ ] Training loop for LoRA parameters
- [ ] Quality-vs-compression tuning
- [ ] Rank selection heuristics

### Phase 4.4: Deployment
- [ ] GGUF export (quantized weights + α + LoRA)
- [ ] Inference integration
- [ ] Speed/accuracy benchmarking
- [ ] Hardware validation

---

## Expected Performance

### Model Size Reduction

| Phase | Model Size | Reduction |
|-------|----------|---|
| Baseline | 3.5 GB | — |
| + Phase 1 (vocab prune) | 3.0 GB | -14% |
| + Phase 1 (BF16 state) | 2.2 GB | -37% |
| + Phase 4 (ternary) | **1.1 GB** | **-69%** |

### Quality Impact

| Metric | Baseline | Ternary | Loss |
|--------|----------|--------|------|
| PERPLEXITY | 8.5 | 8.7 | +2.4% |
| MMLU (5-shot) | 62% | 61% | -1% |
| AVERAGE | 100% | 98-99% | -1-2% |

Quality maintained via LoRA (rank 8-16).

---

## Calibration Strategy

### Step 1: Sensitivity Analysis
```python
# Compute gradient-based sensitivity per layer
for layer in model.layers:
    for weight in layer.weights:
        sensitivity = ||∇loss/∂weight|| @ std_dev(weight)
```

### Step 2: Per-Layer α Calibration
```python
# More sensitive → higher α (less compression)
alpha_layer = base_alpha * (1 + sensitivity_scale)

# Quantize with sensitivity-aware scaling
quantized = quantizer.quantize(weights, alpha_layer)
```

### Step 3: LoRA Fine-tuning
```python
# Decompose residual = original - quantized
residual_matrix = original_weights - dequantized

# Low-rank: U @ S[:r] @ V^T ≈ residual
lora_A, lora_B = svd_decompose(residual_matrix, rank=8)
```

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| src/ternary-quantization.hpp | 350 | Quantizer + orchestrator |
| PHASE4-PLAN.md | (this file) | Technical specification |

---

## Configuration

```cpp
QuantizationConfig config;
config.use_ternary = true;
config.use_lora_correction = true;
config.lora_rank = 8;
config.calibration_quantile = 0.95f;  // Use top 5% for scale
config.symmetric = true;               // {-α, 0, +α}

// Per-layer sensitivity (from analysis)
config.layer_scales = {1.5, 1.3, 1.2, 1.0, ... };
```

---

## Inference

### Quantized Forward Pass
```cpp
// For each layer:
output = ternary_matmul(x, W_ternary, alpha)  // Fast, quantized
output += lora_matmul(x, lora_A, lora_B)      // Correction, small rank

// Result: High quality with small overhead
```

### Speed Impact
- Ternary GEMM: ~2× faster than float32 (reduced bandwidth)
- LoRA: ~5-10% overhead (small rank)
- **Net:** +1.5-1.8× throughput gain (if ternary kernel optimized)

---

## Next Steps

1. **Calibration**
   - Compute layer sensitivities on validation data
   - Optimize α per layer
   - Select LoRA rank

2. **Integration**
   - Load quantized weights in inference
   - Implement ternary matrix multiply kernel
   - Add LoRA forward pass

3. **Fine-tuning**
   - Optimize LoRA parameters
   - Validate quality on downstream tasks
   - Measure actual speedup

4. **Deployment**
   - Export GGUF with quantized weights
   - Test on RTX 3060 Ti
   - Compare vs. Phase 1+2 alone

---

## References

- [Ternary Weight Networks](https://arxiv.org/abs/1605.04711)
- [XNOR-Net](https://arxiv.org/abs/1603.05279)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

---

## Status

**Phase 4.1 Complete:** Core infrastructure (encoding, LoRA framework, orchestration).

**Next:** Phase 4.2-4.4 (calibration, fine-tuning, deployment).
