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

> **One number is unreconciled and it changes the arithmetic below.** The rig notes say
> "Gen3 x1, 128 GB/s" for the CMP 100-210s. GV100 HBM2 is ~830 GB/s nominal and Gen3 x1
> is ~1 GB/s; 128 GB/s matches neither. The tables below assume **830 GB/s HBM2 per
> card**. If the real achieved figure is 128 GB/s, the pipelined throughput below drops
> from 189 to about 97 tok/s and the balanced split shifts hard toward the 170HX.
> `P=0` until measured, which is one `llama-bench` per card.

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

| device | GB/s | share of model | GB held |
| --- | --- | --- | --- |
| 170HX (achieved, measured) | 890 | 30.6% | 4.71 |
| 3060 12 GB | 360 | 12.4% | 1.91 |
| CMP 100-210 #1 | 830 | 28.5% | 4.39 |
| CMP 100-210 #2 | 830 | 28.5% | 4.39 |
| **total** | **2910** | 100% | 15.40 |

    stage time t          = 5.29 ms
    pipelined throughput  = 189 tok/s   (170HX alone: 58)      = 3.3x
    single-token latency  = 4t = 21.2 ms (170HX alone: 17.3)   = 1.22x worse

**Trade 22% of latency for 3.3x of throughput.** For a swarm of agents that is the right
side of the trade; for one interactive stream it is not. The rig can do both, on
different models, at once.

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

**2. Three architectures means a three-arch build.** `CMAKE_CUDA_ARCHITECTURES=70;80;86`.
sm_70 has no bf16 at all, so any bf16 path is 170HX/3060-only. And the lost fork's own
`svmi-gpucheck.py` recorded a Volta CMP card's tensor cores as firmware-gimped - **FP16
~5.6 TFLOPS against FP32 ~10.6, i.e. FP16 slower than FP32**, where a healthy Volta is
~8x the other way. If a newer unlock fixes that, it is a different card and the numbers
above improve; that is a claim to measure, not to assume. `P=0`.

## Order

1. **`llama-bench` each card alone**, same model and flags. Four runs. It replaces every
   `B_i` above with a measured number, settles the 128 GB/s question, and tells you
   whether the Volta tensor cores are still gimped.
2. **Confirm the five pipeline-parallel conditions** hold for your launch, by the
   `pipeline parallelism enabled` log line. If `-ot` is in the command, remove it first.
3. **Measure the pipelined aggregate** at `-np 8` against the 189 tok/s prediction, and
   single-stream latency against 21.2 ms. Both predictions are on record above; a miss
   either way is informative.
4. Only then consider a larger model, and prefer MoE for the reason in the table.
