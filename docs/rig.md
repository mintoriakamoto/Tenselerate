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

## The split I derived above destroys prefill, and the unlocks decide by how much

Sections 1-3 balanced the layer split by **bandwidth**, which is right for decode because
decode is a weight read. Prefill is not: it is compute, and the two optima are nowhere
near each other on these cards.

`gpu-unlock` (GPL-2.0, boot-time kernel module, no VBIOS flash) removes NVIDIA's
GeForce cap on **FP16/BF16/TF32 tensor math with FP32 accumulation** - half rate on
GeForce, full rate on the same silicon sold as professional. The project claims **up to
2x tensor throughput and +25-55% LLM prefill**, with RTX 30 marked stable and 100+ user
reports.

It is live for this codebase: `ggml/src/ggml-cuda/mma.cuh` issues `f16.f32` and
`bf16.f32` MMA - FP16/BF16 in, FP32 accumulate - which is exactly the capped class.
`s8.s32` (what MMQ uses) is not capped, and decode at batch 1 is bandwidth-bound, so
**the unlock cannot touch decode at all.** It is a prefill and wide-batch lever only.

Scope on this rig is narrower than it looks:

| card | capped? | why |
| --- | --- | --- |
| 170HX | **no** | GA100 datacenter die, not GeForce; measured BF16 tensor 191 TFLOPS = full rate for 70 SMs |
| 3060 | **yes** | Ampere GeForce, on the stable list |
| 2x CMP 100-210 | **not by this** | GV100/Volta; gpu-unlock lists Turing, Ampere, Ada only. Their problem is a different and worse one |

### Prefill, 4096 tokens of the 27B, at the bandwidth-weighted split

| scenario | 170HX | 3060 | each CMP | total | tok/s |
| --- | --- | --- | --- | --- | --- |
| as the old `svmi-gpucheck` found them | 0.35 s | 1.07 s | **11.27 s** | 23.96 s | **171** |
| 3060 unlocked | 0.35 s | 0.54 s | **11.27 s** | 23.42 s | **175** |
| both unlocked (healthy GV100) | 0.35 s | 0.54 s | 0.50 s | 1.90 s | **2156** |

`170HX alone: 900 tok/s.` Tensor rates are nominal/derived except the 170HX's, which is
measured - `P=1`, and one `llama-bench` per card replaces the lot.

Three things fall out, and the second is a trap:

1. **A bandwidth-weighted split cuts prefill from 900 to 171 tok/s - a 5.3x
   regression** - if the Volta tensor cores are still as the lost fork measured them
   (FP16 ~5.6 TFLOPS against FP32 ~10.6, i.e. FP16 *slower* than FP32). The two CMPs
   hold **57% of the layers between them** and contribute ~2.5% of the tensor throughput
   each. They become the pipeline.
2. **`gpu-unlock` on the 3060 is worth ~2% here, not 25-55%.** The unlock works; the
   3060 just is not the bottleneck. The project's figure is for a single-GPU machine, and
   it does not survive a pipeline where another stage is 20x slower. Unlocking the wrong
   card is the classic way to spend a weekend on nothing.
3. **The Volta tensor state is the single highest-magnitude unknown on this rig: a
   12.6x swing in prefill** (171 -> 2156 tok/s) from one unmeasured fact. Nothing else
   here, measured or unmeasured, is worth that much.

### And one `--tensor-split` cannot serve both phases

Stage time is `max(bytes/B_i, flops/T_i)`, and the weighting that balances the first does
not balance the second. On these cards the two optima differ by roughly 11x in relative
weight for the CMPs. llama.cpp has one split, so **choosing it is choosing which phase
you are optimising.** For an agent system with cached prefixes, decode dominates the
wall clock; for cold 35K prompts, prefill does.

If the Voltas come back gimped, the better topology is **not** one model pipelined across
four cards. It is **independent model instances** - the 27B on the 170HX, and separate
servers on the other cards - because an agent swarm has no cross-stage dependency to pay
for, every card prefills at its own rate, and no stage can poison the others. Capacity
without the pipeline's coupling.

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

1. **`llama-bench` each card alone**, same model and flags, `-p 4096 -n 64`. Four runs,
   and it is now the highest-magnitude action on the rig: it replaces every `B_i` above
   with a measured number, settles the 128 GB/s question, and - from the `pp` figure -
   tells you which of the three prefill rows above is true. That one fact is worth 12.6x.
   Do this before any unlock work, so you unlock the card that is actually the
   bottleneck.
2. **Confirm the five pipeline-parallel conditions** hold for your launch, by the
   `pipeline parallelism enabled` log line. If `-ot` is in the command, remove it first.
3. **Measure the pipelined aggregate** at `-np 8` against the 189 tok/s prediction, and
   single-stream latency against 21.2 ms. Both predictions are on record above; a miss
   either way is informative.
4. Only then consider a larger model, and prefer MoE for the reason in the table.
