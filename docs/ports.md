# What to port, and what upstream already did for us

Scored per `docs/veritas.md`. Everything here was read from this tree or from a named
source; nothing is recalled.

## First, the deletions: upstream has overtaken part of the lost fork

Before rebuilding anything, note what arrived in upstream while the fork was away. These
are **in this tree right now**:

| capability | where | consequence |
| --- | --- | --- |
| DFlash, DFlash2 draft models | `src/models/dflash.cpp`, `common/speculative.cpp` (`is_dflash2`) | upstream block drafting, with a lattice selector |
| EAGLE3 | `src/models/eagle3.cpp` | no longer a fork feature |
| **DSpark** | `common/speculative.cpp:1324+` (`is_dspark`) | the confidence-truncated draft path exists |
| `--dflash` | `common/arg.cpp` | plumbed |
| lazy tensor reads | `--tensor-read-lazy` (b10679) | overlaps the fork's `--lazy-mode` |

So the honest reading: **a meaningful slice of the fork's speculation work is now
redundant.** Rebuilding it would re-implement upstream. The remaining fork-specific
value is narrower than the 19 lost commits suggest.

## The gap, read line by line

Upstream's DSpark truncates a draft on **per-position** confidence
(`common/speculative.cpp:1332-1334`):

```cpp
const float * conf = params.p_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {
    const int32_t idx = beg + i;
    if (conf && conf[(size_t) idx * n_embd_dec] < params.p_min) {
        break;
    }
```

Each position is judged **in isolation** against a fixed threshold. `common/common.h:332-333`
carries only `p_split` and `p_min` - there is no survival parameter.

DeepInfer, DeepSeek's production engine, does something different
(`deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py`):

```python
cum_rates = torch.cumprod(gathered_rates, dim=1)   # [num_seqs, nextn]
...
extra_budget = total_q - num_seqs
flat_rates = cum_rates.reshape(-1)                 # rank ALL (seq, pos) pairs
```

Two differences, and both matter:

1. **Joint survival, not per-position.** The quantity that decides whether verifying
   position `j` pays is P(positions 1..j *all* accepted) - a cumulative product. A draft
   position whose own confidence clears `p_min` can still be nearly worthless if the
   prefix in front of it is shaky. Upstream cannot see that.
2. **A global budget, ranked.** One column budget for the whole batch, handed to the
   (sequence, position) pairs with the highest cumulative survival. Fresh sequences get
   rate 0 and keep only their one guaranteed token.

llama.cpp issue **#25096** proposes exactly (1) and the load-aware half of (2), with the
same line references, so this is a wanted upstream feature and not a private detour.

## Why this is the right port for this box specifically

The fork's own depth sweep: **n-max 1 = +35%, 2 = +15%, 3 = -15%, 5 = -22%.** A fixed
depth pays for every verify column whether or not the draft was ever likely, so beyond
depth 2 the average draft is worse than free. That is a property of *uniform depth*, not
of the head.

**Fixed depth is the special case of the budget allocator in which every rate is equal.**
Given the same total column budget, a confidence-ranked allocation therefore weakly
dominates it. That is an argument, not a measurement - see the gate.

It also supplies the piece `docs/agents-are-columns.md` was missing. That document argues
idle columns inside MMQ's flat region are a free resource, and then has no principled
answer to *which* speculative work should fill them. This is the answer, from a system
running it in production.

Independent corroboration on the exact model: `jamepeng/llama-cpp-python` reports that
for **Qwen3.8 27B**, "testing so far suggests `draft_n_max=2` as the best starting"
point - matching the fork's measured sweep from a different codebase and a different
tester.

## The gate, and it needs no kernel and no GPU change

The allocator's gain is entirely a function of the **variance** of cumulative survival
across sequences and positions. If every slot's confidence profile looks alike there is
nothing to reallocate and the gain is zero, however sound the theory.

That is measurable **offline, from logs, before writing the allocator**:

1. Run the server with `--spec-type draft-mtp` at `-np 8` on real traffic.
2. Log, per step, the per-position confidence vector already available from
   `llama_get_embeddings_nextn(ctx_dft)` - the same array DSpark reads at
   `speculative.cpp:1328`. **Instrument verified by reading: the data source exists
   upstream and needs no new plumbing.**
3. Offline, for each step: compute `cumprod` per sequence, then compare the columns a
   top-k budget allocation *would* have chosen against the columns fixed depth
   *did* choose, and score both by whether those positions were actually accepted.

Pre-registered:

| result | reading | action |
| --- | --- | --- |
| the allocator picks materially different columns and its picks accept more often | the variance is real; the gain is the measured difference | build Port B, budget-ranked |
| it picks nearly the same columns | confidence is near-uniform across slots; a global budget buys nothing | build only Port A (joint survival), skip the allocator |
| its picks accept *less* often | the head's confidence is not calibrated to acceptance | calibration first; neither port is safe until that is fixed |

The third row is the one to want: DeepInfer carries a `batch_confidence_error` metric
(`tilelang_ops/sampler.py:464`, the sum of `|accept_prob - confidence|`) precisely
because calibration is the assumption the whole mechanism rests on. If the MTP head's
confidence does not track acceptance on this merge, confidence-scheduling is building on
sand, and the ledger's inheritance rule caps everything above it.

## The crash the fork reverted a feature for has a named cause

The lost fork added a bounded attention window (PR #62) and then reverted it: builds
after #49 crashed, and the two suspects on record were that bounded window and the GDN
recurrent-state rollback (PR #52, which exists because of *"the partial `seq_rm` after an
MTP verify block"*). Those were treated as two candidates. **DeepInfer says they are one
interaction, and asserts the invariant that makes it safe**
(`deepinfer/worker/model_runner.py:108-113`):

```python
# 要保证截断长度至少有 window_size + backtracking，保证回退后仍然有足够 window size 的 swa 给到 decoder
assert self.window_size + envs.DEEPINFER_MAX_BACKTRACKING_TOKENS <= envs.DEEPINFER_YOCO_PREFILL_KEEP_TOKENS
```

"The truncation length must be at least `window_size + backtracking`, so that after a
rollback there is still a full window of SWA to hand the decoder."

And the backtracking margin is **tied to the draft depth**
(`deepinfer/environ.py:139`):

```python
DEEPINFER_MAX_BACKTRACKING_TOKENS = EnvVar(lambda v: max(_int(v), envs.DEEPINFER_USE_NEXTN_DECODE + 1))
```

`DEEPINFER_USE_NEXTN_DECODE` is the maximum draft length (default 5). So the retained
suffix must be at least `window + max_draft + 1`.

**That is the mechanism.** A speculative rejection rolls the sequence back by up to the
draft depth. If the bounded window retains exactly the window and no more, the rollback
walks the sequence start *past* the beginning of what was kept, and the decoder is handed
a window that is not fully populated. The fork ran MTP at depths up to 5 with a bounded
window and nothing enforcing a margin.

This is worth more than any speedup in this document, because it un-reverts a feature.
The bounded window was the fork's capacity lever - more resident context in the same
VRAM - and it was abandoned as unsafe when the actual requirement is a one-line assert.
`P=2`, read from source; that it is *this* repo's crash is `P=1` inference from matching
symptoms, and the cheap confirmation is to re-apply the window with the margin enforced
and see whether the crash returns.

## Two scheduling mechanisms, both algorithmic

**Zero-Overhead Scheduler (ZOS).** `deepinfer/worker/worker.py:98-102` reorders a decode
step to:

    1. forward current step
    2. post-process PREVIOUS step      <- deferred by one step
    3. sample current step
    4. post-process current step       (in place for prefill; deferred for ZOS decode)

The CPU-side post-process - detokenisation, stop-string checks, sequence bookkeeping - is
deferred one step so it overlaps the next forward instead of serialising behind it. This
is a candidate for the fork's **7.6 ms of per-step residual** that the ledger records as
neither weight read nor MMVQ column, and it is pure scheduling: no kernel, no numerics.
Enabled by default there, which is a statement of confidence.

**Batch-invariant mode** (`DEEPINFER_BATCH_INVARIANT`). Pins results so they do not
change with batch composition - threaded through the attention path
(`models/v4_modules/attn.py:666,1002`) and reduction order
(`parallel_utils/userbuffer.py:79`). Its value here is **measurement soundness before
performance**: without it, every A/B at `-np 8` is confounded, because the composition of
the batch changes between runs. The ledger's whole measurement queue assumes
reproducibility it has not established.

## Ranked

| # | port | size | `P` | `M` | needs |
| --- | --- | --- | --- | --- | --- |
| 1 | **The offline gate above** | a log tap plus a script | 2 | - | nothing but traffic |
| 2 | **Port A - joint survival cutoff** | ~15 lines in the draft loop, new `--spec-draft-p-survival-min` | 2 | 1-2 | gate row 1 or 2 |
| 3 | **Port B - global ranked budget** | scheduler-level, the invasive half of #25096 | 2 | 2-3 | gate row 1 |
| 1= | **The YOCO margin assert** | one line, plus re-applying the reverted window | 2 | 3 | nothing - it un-reverts a feature rather than adding one |
| 4 | **ZOS deferred post-process** | scheduler reorder, no kernel | 2 | 1-2 | a profile showing the residual is CPU-side |
| 5 | Batch-invariant inference | large | 1 | - | **measurement soundness before performance:** without it every A/B at `-np 8` is confounded by batch composition |
| 6 | SAR draft heads (#25096 feature 2) | new weights, training | 1 | 3 | a trained head; DSpark reports 60-85% over MTP-1, unverified here |

Not portable: every DeepInfer kernel. It targets **SM100 and Ascend 950**; this card is
sm_80. The ideas port, the code does not.

## The honest ceiling

Port A is a strict improvement over a per-position threshold and is cheap, but its effect
is bounded by how often a draft's prefix is shaky while the position itself looks
confident - possibly rare. Port B is where the magnitude is, and it is gated on a variance
measurement nobody has taken. Neither is worth starting before row 1 of the gate, which
costs a log tap.
