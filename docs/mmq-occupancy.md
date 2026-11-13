# MMQ at decode width is occupancy-starved

The ledger's largest outstanding prize is `T1`: MMQ achieves ~280 GB/s where MMVQ
achieves 832 on the same card, against a measured 1227 GB/s of DRAM bandwidth. That
is 23% of the card, and the ledger has carried it as `P=1 PLAUSIBLE` with the wrong
mechanism attached ("MMQ moves 3x the bytes"). This document replaces the mechanism
with one read out of the source, and the arithmetic says the fix may be a single
integer in a config table rather than the new kernel the earlier estimate assumed.

Nothing here has been run. Everything is read from source or derived from it, and the
refutations are pre-registered at the bottom. Read that section before acting.

## What the config asks for

Q4_K on Ampere, `ggml/src/ggml-cuda/mmq-config-ampere.cuh:157`:

    CASE(GGML_TYPE_Q4_K, 256, 1, 128, 8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);

The field order is given by the `CASE` macro at `ggml/src/ggml-cuda/mmq.cuh:206`:
`type, nthreads, occupancy, I, J, sram_layout, K_vram, stream_k, fallback`. So this row
is **nthreads 256, occupancy 1, I 128, J 8**.

`occupancy` is not a hardware property. It is the second argument to
`__launch_bounds__` at `ggml/src/ggml-cuda/mmq.cuh:962`:

    __launch_bounds__(ggml_cuda_mmq_get_nthreads(...), ggml_cuda_mmq_get_occupancy(...))

That second argument is `minBlocksPerMultiprocessor`: a promise to the compiler about
how many blocks need to be co-resident, which the compiler spends as a register
budget. `occupancy = 1` tells nvcc that one block per SM is enough, so it may use up
to the full 255 registers per thread. The register count that results is then itself
the thing that holds occupancy at 1. The constraint is self-fulfilling, and its origin
is a number in a table.

The macro's own `static_assert` permits up to 8.

## The launch geometry at decode width

At decode width `ncols_max` is small, the `J` search at `mmq.cuh:1490-1510` picks the
`J=8` bucket, and `ntx = 1`. Stream-k is on, so the grid comes from
`mmq.cuh:1449-1452`:

    const int ntiles_dst = ntx * nty * ntzw;
    const int tiles_nwaves = (ntiles_dst + nsm - 1) / nsm;
    const int tiles_efficiency_percent = 100 * ntiles_dst / (nsm*tiles_nwaves);
    const dim3 block_nums_stream_k(... tiles_efficiency_percent >= 90 ? ntiles_dst : nsm, 1, 1);

With `I = 128`, the row counts in this model give:

| matmul | rows | ntiles | waves | efficiency | grid | blocks/SM |
| --- | --- | --- | --- | --- | --- | --- |
| o_proj / ffn_down | 5120 | 40 | 1 | 57% | 70 | 1.00 |
| q_proj | 3072 | 24 | 1 | 34% | 70 | 1.00 |
| k_proj / v_proj | 512 | 4 | 1 | 5% | 70 | 1.00 |

Every one of them falls below the 90% threshold, so the grid is `nsm` = 70 blocks:
exactly one block per SM. One block of 256 threads is 8 warps. GA100 holds 64.

**The whole GPU runs at 12.5% occupancy.**

## Why that is the cost

At 1227 GB/s the card needs a quantity of memory in flight to cover HBM latency. One
128-bit load per thread is 16 bytes:

| occupancy | warps/SM | occ. | threads | in flight | smem | regs/thread | vs. 500 ns need |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **1 (shipped)** | **8** | **12.5%** | **17,920** | **280 KB** | 40.0 K | 256 | **47%** |
| 2 | 16 | 25.0% | 35,840 | 560 KB | 80.1 K | 128 | 93% |
| 3 | 24 | 37.5% | 53,760 | 840 KB | 120.1 K | 85 | 140% |
| 4 | 32 | 50.0% | 71,680 | 1120 KB | 160.1 K | 64 | 187% |

The requirement scales with the latency assumed, which is the weakest number here and
is **`P=0`, assumed, not measured on this unit**:

| HBM latency | bytes required in flight |
| --- | --- |
| 300 ns | 359 KB |
| 400 ns | 479 KB |
| 500 ns | 599 KB |
| 600 ns | 719 KB |

The conclusion does not turn on picking the right one. At the most generous 300 ns the
shipped configuration still supplies 78% of the requirement; at 500 ns it supplies
47%. MMQ measures 23% of the card. So occupancy alone predicts a shortfall of the
observed kind and roughly half of its observed size, with the dequantization pipeline
accounting for the rest.

It also retires the mechanism the ledger had been carrying. **No part of this requires
MMQ to move more bytes than MMVQ.** The "3x the bytes" story in `T1` is withdrawn.

## Shared memory does not force it

From `mmq_get_nbytes_shared` (`mmq.cuh:1396`) and
`ggml_cuda_mmq_get_nbytes_shared_x` (`mmq.cuh:427`), with `turing_mma_available` true
on sm_80 so the MMA layout is taken:

    sram_stride(Q8_1) = 2*32 + 2*32/QI8_1 + 4          = 76        (mmq.cuh:136)
    nbs_x             = I * stride * 4 = 128 * 76 * 4  = 38,912 B
    nbs_y             = J * sizeof(block_q8_1_mmq) = 8 * 144 -> padded to 2,048 B
    nbs_ids           = J * 4                          = 32 B
    total                                              = 40,992 B = 40.0 KiB

GA100 offers 164 KiB of shared memory per SM, so **shared memory permits four blocks
per SM.** It is not the binding constraint at any occupancy this document proposes.
Registers are, and registers are what the `occupancy` field hands to the compiler.

## The pattern across the tables

Counting the `occupancy` field over every `CASE` row in each architecture table:

| table | occupancy values used |
| --- | --- |
| ampere | **1** (all 352 rows) |
| blackwell | **1** |
| cdna | **1** |
| pascal-dp4a | 2 |
| pascal-older | 1, 2 |
| gcn | 2, 3 |
| rdna2 | 2 |
| rdna3 | 1, 2, 4 |
| rdna3-5 | 2 |
| rdna4 | 2 |

The three tables that take the MMA data layout are uniformly 1. Every table on the
dp4a path uses 2 or more, and RDNA3 goes to 4. That is not a per-shape tuning result
that happened to land on 1 three hundred and fifty-two times; it is a property of the
code path, and for the regime these kernels were tuned in it is very likely correct.
At prefill width the grid is many waves deep, one block per SM with the maximum
register budget is the right call, and occupancy costs nothing because the waves
themselves supply the parallelism.

Decode width is the case where that reasoning inverts, because the grid is exactly
`nsm` blocks and there is no second wave to hide anything behind.

## The change this argues for

Not "set occupancy to 2". Raising it on the large-`J` rows would spend registers in
the regime where the current value is right. The `J` bucket already separates the two
regimes: `J=8` is decode, `J=128` is prefill.

So: **raise `occupancy` on the small-`J` Q4_K Ampere rows only** — lines 157, 158 and
164 (`J` = 8, 16, 24) — and leave every other row alone. Three integers.

A caveat that bounds the prize, stated here rather than left for the benchmark to
discover: at `N=1` the decode path is MMVQ, not MMQ. MMQ carries the `N>=2` region,
which is the MTP / draft / multi-sequence path. So this cannot move single-stream
`tg`, and the 4.4x headroom in `T1` lives in the flat 55 ms `N=2..16` region — the
same region the "agents are columns" idea depends on.

## Refutations, pre-registered

In order of cost. The first two need no rebuild and the third needs no GPU.

1. **`ncu` on a batched decode.** `sm__warps_active.avg.pct_of_peak_sustained_active`
   should read close to **12.5%**, and `launch__waves_per_multiprocessor` close to
   **1.00**. *If warps_active comes back above 25%, this document is wrong*: the
   kernel is getting more blocks than the config asks for and the 280 GB/s has
   another cause entirely. This is the cheap test that can close the whole idea.
2. **`launch__occupancy_limit_registers`** should be the binding limiter. If shared
   memory or block size is reported as the limiter instead, the smem arithmetic above
   is wrong.
3. **`-Xptxas -v` after changing the three integers, no GPU required.** At
   `occupancy = 2` the kernel must fit in 128 registers per thread. If the register
   count exceeds that, or the build reports spill stores, the change is dead before
   anything is benchmarked and the remaining 4.4x belongs to the read-once kernel
   after all.
4. **`llama-bench` with `-n 64` and draft/MTP enabled, or `-ub` in 2..16.** The
   prediction is movement in the flat region. Single-stream `tg` is explicitly *not*
   the test and is predicted not to move.
5. **Prefill regression check.** `pp4096` must not drop. If it does, the `J` split is
   not clean and the change must be narrowed further.

## Scores

| # | claim | P | F | M | D | status |
| --- | --- | --- | --- | --- | --- | --- |
| K1 | Q4_K/Ampere ships nthreads 256, occupancy 1, I 128 | 2 | 1 | 1 | 2 | CONFIRMED |
| K2 | `occupancy` is `__launch_bounds__`'s minBlocksPerMultiprocessor, a compiler hint, not a hardware limit | 2 | 1 | 2 | 2 | CONFIRMED |
| K3 | At decode width the grid is `nsm` blocks, 1 per SM, 8 of 64 warps = 12.5% | 2 | 2 | 2 | 1 | CONFIRMED (code) |
| K4 | Shared memory is 40.0 KiB/block and permits 4 blocks/SM on GA100 | 2 | 2 | 1 | 2 | CONFIRMED |
| K5 | All MMA-layout tables use occupancy 1; every dp4a-path table uses >= 2 | 2 | 1 | 1 | 2 | CONFIRMED |
| K6 | MMQ's 280 GB/s is occupancy-starvation, not extra bytes | 1 | 2 | 3 | **0** | PLAUSIBLE (capped) |
| K7 | Raising occupancy on the small-J rows recovers part of the 4.4x | 1 | 2 | 3 | **0** | PLAUSIBLE (capped) |

`K6` and `K7` are capped at `PLAUSIBLE` by the inheritance rule: both rest on the
280 GB/s figure, which is `T1` at `P=1`, and on an assumed HBM latency at `P=0`.
Refutation 1 resolves both. Writing more about them will not raise either score.
