# The ledger

Every claim from this session's work that other work stands on, scored by
`AGENTS.veritas.md`. `P` provenance, `F` falsifiability, `M` magnitude, `D`
dependence; 11 maximum. `D=0` caps status at `PLAUSIBLE`.

**The headline, stated before the table so it cannot be buried: this session produced
zero verified performance improvements on the box.** It produced one measured negative
result, one reproduced-and-fixed defect, and the removal of four unverified
load-bearing claims. Scoring exists to make that visible instead of letting a long
list of documents imply progress.

## Foundations - what everything else rests on

| # | claim | P | F | M | D | Σ | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 70 SMs / 4480 cores; the card is a GA100 harvest, not a 108-SM A100 | 3 | 1 | 1 | 2 | 7 | CONFIRMED |
| F2 | Tensor cores uncrippled: TF32 94.78, BF16 191.38 TFLOPS, INT8 389 TOPS, all within 2% of the 0.648 SM ratio | 3 | 1 | 2 | 2 | **8** | CONFIRMED |
| F3 | BAR1 64 GiB > 40 GiB VRAM, so all device memory is host-mappable | 3 | 1 | 1 | 2 | 7 | CONFIRMED |
| F4 | **CUDA-core dp4a: crippled, or not?** `rig-cmp170hx-3060.md` said uncrippled; every deploy preset compiles `DISABLE_DP4A=ON` | 2 | 2 | 3 | 2 | **9** | **CONTRADICTED** |
| F5 | Host RAM does not bound model size; that is an mmap artifact (4x64 MiB staging buffers, `llama-model-loader.cpp:1504-1515`; AUTO→mmap at `:561`) | 2 | 1 | 2 | 2 | 7 | CONFIRMED (code), UNVERIFIED (measurement) |

**F4 is the most valuable item in this ledger and it is not a discovery — it is a
contradiction.** Both sources were read, so it is `CONFIRMED` as a contradiction while
remaining unresolved as to fact. By rule 3 it outranks every theory below that depends
on it. **The resolver first stated here was wrong** - see the retraction table. `--version`
cannot report this flag: the CUDA compile flags a binary exposes come from a hardcoded
list in `ggml_backend_cuda_get_features` (`ggml/src/ggml-cuda/ggml-cuda.cu`), surfaced
by `llama_print_system_info` (`src/llama.cpp:598-619`), and `GGML_CUDA_DISABLE_DP4A`
is not in it - it was a fork-only flag that never appeared in upstream at all. The
command would print `FORCE_MMQ` and silently never print `DISABLE_DP4A`, and an absent
line reads as "off".

Ground truth for the build that produced the binary:

    grep -iE "DISABLE_DP4A|FORCE_MMQ|CUDA_ARCH" <build-dir>/CMakeCache.txt

And the runtime feature list, which is trustworthy for what it does cover:

    ./build/bin/llama-cli -m "$GGUF" --list-devices 2>&1 | head -20

## Measured results

| # | claim | P | F | M | D | Σ | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | e4m3 dot-product error 0.008855 vs q8_0's 0.000092 — 96x worse at the same byte cost | 3 | 2 | 3 | 2 | **10** | CONFIRMED |
| M2 | `test-quantize-fns` overflowed the heap: `tmp_q2` sized `2*test_size` holds the *vec_dot_type*, and e4m3's is f32 | 3 | 3 | 1 | 2 | 9 | CONFIRMED, FIXED |
| M3 | e4m3 format is optimal: worst relative error 5.88% against the 6.25% half-ulp bound, exhaustive sweep | 3 | 3 | 1 | 2 | 8 | CONFIRMED |
| M4 | e4m3 correctness: 28/28 FLASH_ATTN_EXT cases and every CPY path pass on CPU, incl. GQA ratio 6 | 3 | 3 | 1 | 2 | 9 | CONFIRMED (CPU only — CUDA kernels have never executed) |

**M1 is the highest-scoring claim in the session and it argues against my own
feature.** That is the rubric working: a measured negative outranks an unmeasured
positive, and the e4m3 gate gained a mandatory quality arm because of it.

## Theories - unmeasured, and labelled

| # | claim | P | F | M | D | Σ | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | MMQ achieves 280 GB/s where MMVQ achieves 832 — neither bandwidth- nor compute-bound (4% of 389 TOPS). **Mechanism now read from source: occupancy starvation, `docs/mmq-occupancy.md`** | 1 | 2 | 3 | 1 | 7 | PLAUSIBLE |
| T2 | Decode at 50 tok/s is 49% of its own floor; the gap is MMVQ's +11.5 ms per column | 1 | 2 | 2 | 1 | 6 | PLAUSIBLE |
| T2a | **MMVQ re-fetches each weight block once per column** — confirmed from source, see below | **2** | 2 | 2 | 1 | 7 | **CONFIRMED (mechanism)** |
| T3 | The +11.5 ms slope is dp4a→dp2a *emulation* cost paid per column | 1 | 2 | 3 | **0** | 6 | PLAUSIBLE (capped — rests on F4) |
| T4 | e4m3's rationale: quantized KQ dots all end in `ggml_cuda_dp4a`; e4m3 is the first byte-dense dp4a-free cache type | 2 | 2 | 2 | **0** | 6 | PLAUSIBLE (capped — rests on F4) |
| T5 | "Agents are columns": N agents cost one weight read inside the flat region, so cost is a step function | 1 | 2 | 3 | **0** | 6 | PLAUSIBLE (capped — rests on T1) |
| T6 | `GGML_CUDA_MMVQ_MAX=32` with `-np 8` costs ~30% of aggregate | 1 | 2 | 2 | **0** | 5 | PLAUSIBLE (capped) |
| T7 | Vocabulary pruning: ceiling +4.6% at 64K, break-even inflation 1.046 | 2 | 3 | 1 | 1 | 7 | PLAUSIBLE (gate built and tested) |
| T8 | GDN state fp32→fp16 is worth ~0.46% of decode, not a "pays twice" item | 2 | 1 | 0 | 1 | 4 | CONFIRMED as arithmetic; argues against the proposal |

Note what the `D` column does to T3-T6: **four of the session's most exciting ideas are
capped at PLAUSIBLE because they inherit from one unrun command.** No amount of further
writing raises them. That is the ordering the rubric is for.

## The per-column cost, read from source

T2a raises T2's central mechanism from `P=1` (inferred from a fitted step time) to `P=2`.
The kernel says it plainly.

`ggml/src/ggml-cuda/mmvq.cu:747-752` — the column loop sits **inside** the K-block loop:

```cpp
for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    for (int j = 0; j < ncols_dst; ++j) {
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp[j][i] += vec_dot_q_cuda(
                vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
```

The weight index `kbx_offset + i*stride_row_x + kbx` **does not depend on `j`**. And the
callee takes the raw pointer and re-derives the address every call
(`vecdotq.cuh:918-921`):

```cpp
static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;
```

Nothing stages x into registers or shared memory before the `j` loop. The only x-side
amortization in the kernel is an L2 prefetch hint guarded by
`#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK`, which does not compile on sm_80.

### But the mechanism is not what the proposal assumed, and that matters

`kernel-readonce.md` read the slope as *"it re-streams most of the weights for every
column"* - i.e. DRAM traffic scaling with N. The code does not support that. A block
fetched at `j=0` is fetched again at `j=1` microseconds later, so the repeat almost
certainly **hits L1**. What provably scales with N is:

- **global load requests** (one per column per block),
- **the entire dequantize pipeline** - the shifts, masks, scale lookups and float
  combines in `vec_dot_q4_K_q8_1_impl_vmmq`, re-run per column,
- **the dp4a chains**, two per `QR4_K` iteration, re-run per column.

**DRAM bytes most likely stay roughly flat.** The cost is instruction issue, L1
bandwidth and repeated arithmetic - not re-read bytes.

### This makes the project's own pre-registered refutation able to fire falsely

`kernel-readonce.md` pre-registers: *"MMQ at N=8 reads ~1x the weight bytes - the
traffic is already minimal... this document should be closed as refuted"*, and
`width-bytes.sh` measures DRAM bytes.

But **~1x DRAM bytes is also exactly what a per-column L1 re-read looks like.** The
instrument cannot tell "properly amortized" from "re-read out of cache", so the
refutation branch can trigger on a problem that is entirely real. This is the same
class of error as the `--version` resolver in the retraction table: a pre-registered
test whose instrument cannot see the thing it tests for. Doctrine rule 4 applies -
score it `F=1` until the metrics discriminate.

**The metrics that do discriminate**, one `ncu` run:

    ncu --metrics \
      dram__bytes_read.sum,\
      l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
      smsp__inst_executed.sum,\
      sm__warps_active.avg.pct_of_peak_sustained_active \
      ./build/bin/llama-bench -m "$GGUF" -ngl 999 -fa 1 -n 16

Pre-registered, on N=1 vs N=8:

| result | reading | action |
| --- | --- | --- |
| load requests and instructions scale ~8x, DRAM bytes ~flat | per-column L1 re-read plus repeated dequant - the real shape | a read-once kernel wins, but it wins on **instructions and L1**, not DRAM; size it that way |
| DRAM bytes also scale ~8x | the L1 is not absorbing the repeats after all | the original framing was right |
| neither scales | the slope is not the `j` loop at all | re-measure the slope before touching a kernel |

## Reconciled against the measured fleet

`rust/profiles/fleet.json` in the `deepinfer` repo measures this card at **1227 GB/s
achieved** (decision kernel) and 1445 GB/s on a read benchmark. This ledger's figures
came from the lost fork and used **832 GB/s**, derived from MMVQ's 18.5 ms step
intercept.

**These do not contradict each other, and distinguishing them sharpens T1 and T2.** 832
is what *the MMVQ kernel achieves*; 1227 is what *the card achieves*. The ledger was
measuring a kernel and calling it the card.

| | GB/s | weight read | floor per emitted token (MTP 1.88) |
| --- | --- | --- | --- |
| MMVQ achieved | 832 | 18.51 ms | 9.85 ms = **102 tok/s** |
| **card achievable** | **1227** | **12.55 ms** | **6.68 ms = 150 tok/s** |
| card read benchmark | 1445 | 10.66 ms | 5.67 ms = 176 tok/s |

So T2's headline - *"decode at 50 tok/s is 49% of its own floor"* - is right about MMVQ
and understates the gap against the hardware:

- **49% of MMVQ's own floor** (the ledger's framing, still true)
- **33% of the card's achievable floor** (the framing that matters for what is on the
  table)

And the kernel-efficiency picture in T1 gets worse, which makes it a bigger prize rather
than a smaller one:

| path | achieved | share of the card's 1227 |
| --- | --- | --- |
| MMVQ N=1 | 832 GB/s | **68%** |
| MMQ flat N=2..16 | 280 GB/s | **23%** |

T1 previously said MMQ moves the bytes at 280 against MMVQ's 832 - a 3x gap between two
kernels. Against the card it is **4.4x of headroom on MMQ and still 1.5x on MMVQ**.
Neither kernel is close to the silicon.

Scores unchanged: both remain `P=1` on the step times they are fitted from, and the
corrected `ncu` run in the section above is still what settles them. What changed is the
denominator, and it moved against us.

## MMQ's 4.4x, taken apart from source

`docs/mmq-occupancy.md` has the full derivation and its pre-registered refutations. The
short version, because it changes what the biggest item in this ledger is *for*:

Q4_K on Ampere ships `occupancy = 1` (`mmq-config-ampere.cuh:157`), and that field is
`__launch_bounds__`'s `minBlocksPerMultiprocessor` (`mmq.cuh:962`) — a register budget
handed to the compiler, not a hardware limit. At decode width the stream-k grid is
exactly `nsm` blocks (`mmq.cuh:1449-1452`), so one block of 256 threads lands per SM:
**8 of GA100's 64 warps, 12.5% occupancy, across the whole GPU.**

| occupancy | occ. | threads | in flight | smem | regs/thread | vs. 500 ns need |
| --- | --- | --- | --- | --- | --- | --- |
| **1 (shipped)** | **12.5%** | **17,920** | **280 KB** | 40.0 K | 256 | **47%** |
| 2 | 25.0% | 35,840 | 560 KB | 80.1 K | 128 | 93% |

Shared memory is 40.0 KiB per block and GA100 has 164, so **four blocks would fit**;
nothing in the hardware asks for 1. Every architecture table on the dp4a path uses 2 or
more, and all three MMA-layout tables use 1 uniformly — 352 rows of it on Ampere.

Two consequences for this ledger. **The "3x the bytes" mechanism under T1 is retracted**
(below): occupancy alone predicts a shortfall of the right kind and about half the
observed size, with no extra traffic required. And the fix is no longer costed as a
kernel — it is three integers on the small-`J` rows, with `-Xptxas -v` able to kill it
before any GPU is involved.

T1 and the new `K6`/`K7` stay `PLAUSIBLE`. They rest on the 280 GB/s figure, still
`P=1`, and on an assumed HBM latency at `P=0`. One `ncu` counter settles all three, and
it is the same run this ledger has been recommending: `warps_active` near 12.5%
confirms it, above 25% kills it outright.

## Retracted and corrected

| claim | what it was | what it cost |
| --- | --- | --- |
| `n_outputs_max` wastes ~2 GB of VRAM | `P=0`. `server-context.cpp:1043-1045` sets it unconditionally; the fallback never fires | one wrong recommendation, retracted same session |
| BAR1 ⇒ `O_DIRECT` can read straight into VRAM | `P=0`. The kernel cannot pin `VM_IO\|VM_PFNMAP` pages; needs GPUDirect Storage | a whole proposed mechanism, withdrawn — and the real answer (F5) was simpler |
| "A `% 3` rung makes the budget whole" | `P=0`. The kernel already tiles and masks; the bug was the selection ladder | a wrong explanation attached to a correct fix |
| `-bs` shipped without tracing enforcement | `speculative.cpp:2494-2510` leaves `n_outputs_max_per_seq=1` for draft-mtp | a latent bug in my own change, flagged by me |
| "162-170 TFLOPS FP16" unlabelled | ambiguous between tensor and CUDA-core (44.92); relabelled against F2 | carried through several documents |
| "Tensor-core utilisation is not the lever; bytes are" | contradicted by arithmetic: bytes are 6% of the per-sequence term | shaped the order of work |
| "~108 SMs" | `P=0`, assumed; F1 corrects it | my own reasoning about filling the GPU |
| "the compute explanation needs a coincidence" | `P=0`. I argued the slope could not be compute because dp4a MACs for one column are ~1 ms, so compute would have to be ~15 ms and MMVQ at N=1 would be both at the memory roof and 81% compute-bound. That under-counted the repeated unit: the code repeats the **whole dequant pipeline** per column, not just the MACs. No coincidence is required and the compute/L1 branch is live | steered the diagnosis toward DRAM re-reads, which the source does not support |
| "MMQ moves 3x the bytes" | `P=0`. Nothing in the kernel re-reads the weights per tile, and the occupancy arithmetic in `docs/mmq-occupancy.md` reproduces a shortfall of the observed kind without any extra traffic at all. The 280-vs-832 gap is memory-level parallelism, not volume | the mechanism under the session's largest prize, and it costed the fix as a new kernel when the lever is three integers in a config table |
| `llama-cli --version` resolves F4 | `P=0`. `DISABLE_DP4A` is absent from upstream and from the CUDA feature list, so the command's silence would have read as "flag off" | **the ledger's own top-ranked item, recommended five times.** A broken instrument on a pre-registered test is worse than no test: it returns a confident wrong answer |

## Before and after

Scored at the state of the knowledge base, not the state of the hardware.

| | before | after |
| --- | --- | --- |
| load-bearing claims at `P=0` (asserted) | 4 | **0** |
| load-bearing claims with `D=0` **and not labelled as such** | 4 | **0** |
| contradictions found and marked | 0 | **1** (F4) |
| claims at `P=3` (reproduced measurement) | 2 | **7** |
| load-bearing mechanisms confirmed from source (`P=2`) | - | **+1** (T2a) |
| claims retracted | 0 | **10** |
| broken instruments found on pre-registered tests | 0 | **2** |
| **verified performance gains on the box** | **0** | **0** |

The last row is the honest one. Nothing got faster. What changed is that the four
claims holding up the most expensive planned work are now labelled, and the single
command that resolves the biggest of them is written down as step 0 instead of being
assumed away.

## What the rubric says to do next

Ranked by `P+F` for what to trust, and by `M` for what to measure — rule 2, kept
separate on purpose.

1. **F4**, seconds. `grep -iE "DISABLE_DP4A|FORCE_MMQ" <build-dir>/CMakeCache.txt`
   (not `--version` - see above). It is the foundation under T3, T4, T5 and T6: four
   `D=0` claims, combined magnitude 11. Nothing else here has a better ratio of cost
   to consequence.
2. **T6**, one A/B, no rebuild. Highest magnitude per minute after F4, and it may be
   costing throughput right now.
3. **M4 on CUDA0.** `test-backend-ops -o FLASH_ATTN_EXT -b CUDA0`. The e4m3 kernels
   have never executed; everything claimed about them is CPU-only.
4. **T1**, one `ncu` run — now with a mechanism and a threshold.
   `sm__warps_active.avg.pct_of_peak_sustained_active` near **12.5%** confirms
   occupancy starvation; **above 25% refutes it** and the 4.4x has another cause.
   Cheaper still, and needing no GPU: change the three integers named in
   `docs/mmq-occupancy.md` and read `-Xptxas -v`. If the kernel will not fit in 128
   registers per thread, the item is closed.
5. **T7**, CPU only. The gate exists and is tested.

**Before rebuilding any of the lost fork:** `docs/ports.md`. Upstream landed DFlash,
DFlash2, EAGLE3 and DSpark while the fork was away, so part of what was lost is now
redundant, and the one genuine gap (joint survival and a globally ranked draft budget,
llama.cpp issue #25096) is gated on an offline variance measurement that costs a log tap.

Not on this list: writing anything further about T5. It cannot rise above PLAUSIBLE
until F4 and T1 resolve, and more prose will not change its score.
