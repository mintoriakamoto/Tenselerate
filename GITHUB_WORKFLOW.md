# Tenselerate GitHub Workflow

## Current Project Status

### Active Branches

| Branch | Status | Focus | Last Commit |
|--------|--------|-------|------------|
| `master` | Stable | Base llama.cpp | baseline |
| `optimize/vocab-bf16-swarm` | ✅ Complete | Phase 1: Vocab prune, BF16 state, swarm scheduler | 6f56eba |
| `optimize/phase2-kernels` | 🔄 In Progress | Phase 2: Fused FFN, sparse attention | 224e0b3 |
| `optimize/phase3-speculative` | ⏳ Pending | Phase 3: Draft-target verification | — |
| `optimize/phase4-quantization` | ⏳ Pending | Phase 4: Ternary weights, mixed-precision | — |
| `optimize/phase5-swarm` | ⏳ Pending | Phase 5: Advanced multi-agent batching | — |

---

## Development Workflow

### Creating a Feature Branch

```bash
# Create from main (or from a parent optimization branch)
git checkout -b optimize/phase<N>-<feature>
git push -u origin optimize/phase<N>-<feature>
```

### Committing Changes

```bash
# Atomic commits with clear messages
git add <files>
git commit -m "feat/fix: <summary>

<body with details>

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

### Pull Request Workflow

1. **Push to branch** (auto-triggers CI/CD)
2. **GitHub checks:**
   - `build-cuda-ubuntu` – CUDA compilation
   - `python-lint` – Python style (flake8)
   - `code-style` – C++ formatting
3. **Request review** (in PR description)
4. **Address feedback** → commit → CI re-runs
5. **Merge** when green ✅

---

## CI/CD Pipeline

### Automated Checks

- ✅ **CUDA Build** (`build-cuda-ubuntu.yml`)
  - Compiles with `-DGGML_CUDA_F16 -O3`
  - Runs on Ubuntu 22.04 with CUDA 12.1
  - Timeout: 30 minutes

- ✅ **Python Linting** (`python-lint.yml`)
  - flake8 on `gguf-py/**/*.py`
  - No print() statements (use logging)
  - Max line length: 120 chars

- ✅ **Code Style** (`code-style.yml`)
  - clang-format on src/
  - EditorConfig validation

### Manual Benchmarking

For performance-critical changes:

```bash
# Run optimization benchmark suite
./benches/measure-optimizations.sh \
    --config hermes-1.7b-base \
    --metric decode-throughput \
    --iterations 100
```

---

## PR Template Requirements

```markdown
## Overview
[What this PR does and why]

## Changes
- [x] Fused FFN kernel
- [x] Integration layer
- [x] Documentation

## Performance Impact
- Memory I/O: -60% (compared to 3 separate kernels)
- Throughput: +1.5-2.5× on RTX 3060 Ti

## Testing
- [x] Unit tests in benches/test-fused-ffn.cu
- [x] Benchmark: benches/measure-optimizations.sh
- [ ] Integration test on real model (blocked on hardware)

## AI Disclosure
YES – All code generated with Claude AI assistant.
Human oversight: Code review, test validation, optimization strategy.
```

---

## Issue Tracking

### Creating an Issue

```bash
gh issue create \
  --title "Phase 2: MMVQ width-32 optimization" \
  --body "Expand MMVQ dispatch width from 8 to 32 for RTX 3060 Ti" \
  --label "phase-2,cuda-optimization"
```

### Issue Labels

- `phase-1` / `phase-2` / `phase-3` / `phase-4` / `phase-5`
- `cuda-optimization` – kernel work
- `memory-optimization` – storage/layout
- `integration` – linking components
- `blocked` – waiting on something
- `ready-for-review` – PR ready

---

## Review Checklist

Before merging:

- [ ] Passes all CI checks (green build, linting)
- [ ] Code review approved (1+ maintainer)
- [ ] Benchmark results attached (if performance-critical)
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] AI disclosure included in PR body

---

## Next Steps (Immediate)

1. **Phase 2.1 – Sparse Attention Kernel**
   - Location: `src/kernels/fused-sparse-attn.cu`
   - Scope: ~200 lines CUDA
   - Timeline: 2-3 days
   - Target: 40-50% reduction in attention compute

2. **Phase 2.2 – CMake Integration**
   - Add CUDA kernel compilation to CMakeLists.txt
   - Link against libfused_ffn.so
   - Test with `make test`

3. **Phase 2.3 – Integration into Inference Loop**
   - Modify `src/llama.cpp` to call fused FFN
   - Replace 3 separate `ggml_mul_mat` calls
   - Verify numerical correctness (golden tests)

4. **Hardware Validation (blocking Phase 3)**
   - Run on RTX 3060 Ti
   - Measure actual throughput
   - Compare vs. Phase 1 predictions

---

## Tools & Commands

### Local Development

```bash
# Build (with CUDA)
cmake -B build -DGGML_CUDA=ON
make -C build -j$(nproc)

# Run tests
./bin/test-llama-archs --model hermes-1.7b

# Benchmark
./benches/measure-optimizations.sh
```

### GitHub Interactions (gh CLI)

```bash
# List PRs
gh pr list --state open

# Check PR status
gh pr checks <pr-number>

# Request review
gh pr review <pr-number> --request-review @reviewer

# Merge PR
gh pr merge <pr-number> --squash
```

### Monitoring CI

- GitHub Web: https://github.com/mintoriakamoto/Tenselerate/actions
- Filter: branch=`optimize/phase2-kernels` to see Phase 2 builds

---

## Troubleshooting

### CUDA Compilation Fails

**Error:** `error: no default CUDA compute capability`

**Fix:** Specify architecture:
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
```
(RTX 3060 Ti = compute capability 86)

### Python Import Errors

**Error:** `attempted relative import with no known parent package`

**Fix:** Use absolute imports:
```python
from gguf.gguf_reader import GGUFReader  # ✅ Correct
# from gguf_reader import GGUFReader    # ❌ Wrong
```

### Shared Memory Exceeds Limit

**Error:** `__syncthreads() is not valid in thread divergent code`

**Fix:** Reduce shared memory or increase block size:
```cuda
__shared__ float smem[1024];  // 4KB, not 96KB
```

---

## References

- [Tenselerate Roadmap](./TENSELERATE_ROADMAP.md)
- [Optimization Plan](./OPTIMIZATION_PLAN.md)
- [Fused FFN Documentation](./src/kernels/README-FUSED-FFN.md)
- [CONTRIBUTING.md](./CONTRIBUTING.md) – General contribution guidelines
