# The rig, and the topology its arithmetic forces

Everything in this repository is designed for one machine. This page states it, does the
arithmetic, and derives the topology. Scored per `docs/veritas.md`.

## The machine

| part | spec | link |
| --- | --- | --- |
| board | B650E (AM5) | - |
| CPU | Ryzen 9 9950X, 16C/32T | - |
| RAM | 32 GB DDR5 (Crucial, top bin) | - |
| GPU 0 | **CMP 170HX**, GA100 sm_80, 10 GB stock -> **unlocked to 40 GB**, 70 SMs | PCIe Gen2 x4, ~2 GB/s |
| GPU 1 | **RTX 3060 12 GB**, GA106 sm_86 | PCIe Gen4 x4, ~8 GB/s |
| GPU 2 | **CMP 100-210**, GV100 sm_70, 16 GB HBM2, unlocked | Gen3 x1 |
| GPU 3 | **CMP 100-210**, GV100 sm_70, 16 GB HBM2, unlocked | Gen3 x1 |

**Total VRAM: 84 GB across four devices and three architectures** (sm_70, sm_80, sm_86).

> **Resolved, from measurement.** An earlier revision of this page flagged "Gen3 x1,
> 128 GB/s" as unreconcilable and assumed 830 GB/s HBM2. The fleet has since been
> measured, and the numbers live in `rust/profiles/fleet.json` of the `deepinfer` repo.
> Every bandwidth and tensor figure below is now that measurement, and three of the old
> assumptions were wrong in ways that changed the conclusions. The corrections are in
> their own section.

## The three numbers that decide the topology

### 1. Splitting makes single-stream latency *worse*, not better

With a layer split, one token must traverse every stage, so its latency is the **sum**:

    latency = SUM_i ( f_i * W / B_i )

Minimising that subject to `SUM f_i = 1` puts **everything on the fastest card**. There is
no arrangement of four GPUs that beats one 170HX on single-token latency for a model that
fits in 40 GB. This is the opposite of the intuition that more GPUs are faster, and it is
why the 27B should stay on GPU 0 alone whenever latency is what matters.

### 2. Splitting multiplies throughput, if work is in flight

Balance the split by bandwidth (`f_i ∝ B_i`) and every stage takes the same time
`t = W / SUM(B)`. Steady-state pipelined throughput is then `SUM(B) / W`:

All figures measured (`rust/profiles/fleet.json`):

| device | DRAM GB/s | tensor TFLOPS | link GB/s | share of model | GB held |
| --- | --- | --- | --- | --- | --- |
| 170HX 40 GiB | **1227** | 191.38 | 1.6 | 41.5% | 6.39 |
| CMP 100-210 #1 | **700** | 87.0 | 0.81 | 23.7% | 3.65 |
| CMP 100-210 #2 | **700** | 86.6 | 0.85 | 23.7% | 3.65 |
| RTX 3060 12 GB | **328** | 25.0 (spec est.) | - | 11.1% | 1.71 |
| **total** | **2955** | - | - | 100% | 15.40 |

    stage time t          = 5.21 ms
    pipelined throughput  = 192 tok/s   (170HX alone: 80)      = 2.41x
    single-token latency  = 4t = 20.8 ms (170HX alone: 12.6)   = 1.66x worse

**Trade 66% of latency for 2.4x of throughput.** That is a worse trade than the earlier
estimate, because the 170HX alone is faster than that estimate assumed - 1227 GB/s
achieved, not 890. For a swarm it is still the right side; for one interactive stream it
is now clearly wrong.

### 3. The links are a non-issue for layer split and fatal for anything else

A layer boundary carries **activations**, not weights: `n_embd x 2 bytes = 10.2 KB` per
token.

| what crosses | per token | over the slowest link (Gen3 x1) |
| --- | --- | --- |
| activations (layer split) | 10.2 KB | 31 us over 3 hops = **0.15% of the step** |
| weights (row / tensor split) | 15.4 GB | **15.4 s per token** |

So `LLAMA_SPLIT_MODE_LAYER` is the only viable topology on this rig, and
`LLAMA_SPLIT_MODE_ROW` / `_TENSOR` are off the table by five orders of magnitude. The
crippled PCIe links everyone treats as this hardware's fatal flaw simply do not matter
once only activations cross them.

## Measured: which card to unlock, and it is not the one I said

An earlier revision of this page predicted that the two CMP 100-210s would destroy
prefill, on the strength of the lost fork's `svmi-gpucheck` finding a Volta CMP card's
tensor cores firmware-gimped at FP16 ~5.6 TFLOPS. **The measurement says otherwise.**

| | assumed here | measured | |
| --- | --- | --- | --- |
| 170HX DRAM | 890 GB/s | **1227 GB/s** | +38% |
| CMP 100-210 DRAM | 830 GB/s | **700 GB/s** | -16% |
| **CMP 100-210 tensor** | **5.6 TFLOPS** | **87.0 TFLOPS** | **15.5x better - not gimped** |
| 3060 DRAM | 360 GB/s | 328 GB/s | - |

87 TFLOPS against a healthy GV100's ~125 is roughly 70% of full rate. Whatever unlock
those cards are running, it worked. The 12.6x prefill swing that section warned about
resolves to the good case, and **the conclusion it led to was wrong.**

### Prefill at the bandwidth-weighted split, with measured tensor rates

| stage | share of layers | TFLOPS | stage time | share of pipeline |
| --- | --- | --- | --- | --- |
| 170HX | 41.5% | 191.38 | 0.480 s | 18.0% |
| CMP #1 | 23.7% | 87.0 | 0.602 s | 22.6% |
| CMP #2 | 23.7% | 86.6 | 0.605 s | 22.7% |
| **3060** | **11.1%** | **25.0** | **0.982 s** | **36.8%** |

Pipeline 2.669 s against 1.156 s for the 170HX alone: **2.31x slower, so a measured 900
tok/s becomes about 390.** Prefill still regresses under a bandwidth-weighted split - but
**the bottleneck is the 3060**, not the Voltas. It holds 11% of the layers and eats 37%
of the pipeline.

So the correction, stated plainly: **the previous revision said `gpu-unlock` on the 3060
was worth ~2% and the wrong card to unlock. With measured numbers it is the right card,
and it is worth about +23%** (prefill 2.669 s -> 2.178 s, 390 -> 478 tok/s), because the
3060 is the only capped card *and* now the slowest stage. The reasoning was sound; the
input was wrong, and it was wrong because it came from a document rather than a
measurement. `D=0` propagating, exactly as `docs/veritas.md` rule 1 describes.

### The split is still the bigger lever

Bandwidth-weighting over-loads the 3060 for prefill, because its bandwidth share exceeds
its compute share:

| | bandwidth-weighted | compute-weighted |
| --- | --- | --- |
| 170HX | 41.5% | **49.1%** |
| CMP #1 | 23.7% | 22.3% |
| CMP #2 | 23.7% | 22.2% |
| **3060** | **11.1%** | **6.4%** |

Moving the 3060 from 11.1% to 6.4% of the layers costs decode a little and buys prefill
a lot - and no unlock is needed for it, just `--tensor-split`. Because stage time is
`max(bytes/B_i, flops/T_i)` and one split must serve both phases, **the honest answer on
this fleet is to under-weight the 3060 relative to its bandwidth** and let the 170HX
carry more, then unlock the 3060 on top.

Three further facts from `deepinfer-rs`, all build-relevant:

- **CUDA 13 dropped sm_70**, so the CMP 100-210s require **CUDA 12.x**. That constrains
  the whole toolchain for the fleet, not just those cards.
- Link bandwidths are **0.81 and 0.85 GB/s** for the CMPs and 1.6 GB/s for the 170HX -
  measured, and consistent with Gen3 x1 and Gen2 x4. The activation traffic conclusion
  holds with room to spare: 10.2 KB per token over 0.81 GB/s is 13 us.
- The 3060's 25.0 TFLOPS is a **spec estimate for FP16 with FP32 accumulate** - i.e. the
  *capped* figure - and is the one number in the fleet profile that is not measured.
  `probe` on that card is what confirms the +23% above.

## What 84 GB changes

| bpw | dense params that fit |
| --- | --- |
| 4.5 (Q4_K_M) | **~149 B** |
| 2.5 | **~269 B** |

The "fit a bigger model" problem is over as a capacity question. What remains is that a
dense model reads all its weights per token, so a 149B dense model decodes at
`2910/84 = 35 tok/s` pipelined. An **MoE** of the same resident size reads only its active
experts, so it is both bigger and faster - which is the same conclusion reached for one
card, now with four times the room to use it.

## Two traps, both read from source

**1. `-ot` silently disables pipeline parallelism.** `src/llama-context.cpp:432-438`
enables it only when all of these hold:

```cpp
bool pipeline_parallel =
    model.n_devices() > 1 &&
    model.n_gpu_layers() > model.hparams.n_layer_all &&
    model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
    cparams.offload_kqv &&
    !model.has_tensor_overrides();
```

Any `-ot` / `--override-tensor` sets `has_tensor_overrides()` and **turns off the 3.3x**,
with no error - only the absence of the `pipeline parallelism enabled` log line. The lost
fork's documentation leaned on `-ot` for placement. On this rig that advice is now
actively harmful, and the check is the log line.

It also requires every device to report `async` and `events`
(`llama-context.cpp:440-456`); a device missing either silently drops the whole feature.
**Worth confirming the Volta cards report both** before counting on any of this.

**2. Three architectures, and CUDA 12.x is forced.** `CMAKE_CUDA_ARCHITECTURES=70;80;86`,
and **CUDA 13 dropped sm_70**, so the whole fleet is pinned to CUDA 12.x by the two CMP
100-210s. sm_70 also has no bf16 at all, so any bf16 path is 170HX/3060-only - which
matters because the 170HX's headline tensor figure (191.38 TFLOPS) *is* bf16. A bf16 fast
path would silently exclude half the fleet.

The lost fork's `svmi-gpucheck.py` recorded a Volta CMP card's tensor cores as
firmware-gimped at FP16 ~5.6 TFLOPS, and an earlier revision of this page carried that
forward. **It is superseded: the fleet measures 87.0 and 86.6 TFLOPS on these two
cards.** Either they are different units or the unlock worked. Treat any surviving
"Volta is gimped" claim in this repo as retracted.

## Order

1. **`probe` the 3060** (`deepinfer-rs probe --device cuda:3`). It is the one unmeasured
   number in the fleet profile - 25.0 TFLOPS is a spec estimate for the capped FP16/FP32
   path - and it is the slowest prefill stage, so it decides whether `gpu-unlock` on that
   card is worth the +23% predicted above.
2. **Confirm the five pipeline-parallel conditions** hold for your launch, by the
   `pipeline parallelism enabled` log line. If `-ot` is in the command, remove it first.
3. **Measure the pipelined aggregate** at `-np 8` against the **192 tok/s** prediction,
   and single-stream latency against **20.8 ms**. Both are on record above; a miss either
   way is informative.
4. **Try the compute-weighted split** (`--tensor-split` putting the 3060 at ~6% rather
   than ~11%). It needs no unlock and no rebuild, and it is the cheapest prefill win on
   the list.
5. Only then consider a larger model, and prefer MoE for the reason in the table.

## The Volta cards are first-class in this tree, and the build now refuses to lose them

A claim that the two CMP 100-210s are "depreciating in software" was made from another
fork's README (`P=1`, and about a different codebase). Read against this tree it is wrong:

- Flash attention has a dedicated Volta tensor-core path: `fattn.cu:224` and `fattn.cu:673`
  select between the vector, tile and MMA-F16 kernels specifically for `cc == 700`.
- `mmf.cuh` carries `VOLTA_MMA_AVAILABLE` branches with their own tile geometry.
- None of the `BEST_FATTN_KERNEL_NONE` returns in `fattn.cu:576-631` depends on the
  architecture, so nothing falls back to the CPU because a card is Volta.

What the source does show is two ways to lose the cards **silently**, both in the build:

1. `ggml/src/ggml-cuda/CMakeLists.txt:43` drops `70-virtual` from the default list when
   the toolkit is CUDA 13 or newer. No warning; the binary simply cannot serve sm_70.
2. `volta_mma_available()` (`common.cuh:360`) is true only when the highest compiled
   architecture not above the card is **exactly** 700. Build with `61;80;86` and a Volta
   card resolves to 610: it keeps running, off its tensor-core kernels, with no log line.

Both are closed by `GGML_CUDA_REQUIRE_ARCHS` (`ggml/cmake/cuda-require-archs.cmake`):
configuration fails unless every listed architecture is compiled, fails on `native`
because that cannot be verified, and fails outright on a toolkit that cannot target the
architecture. The `tenselerate-fleet` preset sets it for this machine:

    cmake --preset tenselerate-fleet     # 70-real;80-real;86-real, required: 70;80;86

Still outside the build's reach, and `P=0` until checked on the box: the **driver**.
Volta support ends with a legacy driver branch, so the machine as a whole has to stay on
the last branch that still drives sm_70. Record the installed branch here once read from
`nvidia-smi`, and pin it in the package manager.
