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

## Ranked

| # | port | size | `P` | `M` | needs |
| --- | --- | --- | --- | --- | --- |
| 1 | **The offline gate above** | a log tap plus a script | 2 | - | nothing but traffic |
| 2 | **Port A - joint survival cutoff** | ~15 lines in the draft loop, new `--spec-draft-p-survival-min` | 2 | 1-2 | gate row 1 or 2 |
| 3 | **Port B - global ranked budget** | scheduler-level, the invasive half of #25096 | 2 | 2-3 | gate row 1 |
| 4 | Batch-invariant inference (`DEEPINFER_BATCH_INVARIANT`) | large | 1 | - | **worth noting for a different reason:** without it, every A/B at `-np 8` is confounded by batch composition. It is a measurement-soundness feature before it is a feature |
| 5 | SAR draft heads (#25096 feature 2) | new weights, training | 1 | 3 | a trained head; DSpark reports 60-85% over MTP-1, unverified here |

Not portable: every DeepInfer kernel. It targets **SM100 and Ascend 950**; this card is
sm_80. The ideas port, the code does not.

## The honest ceiling

Port A is a strict improvement over a per-position threshold and is cheap, but its effect
is bounded by how often a draft's prefix is shaky while the position itself looks
confident - possibly rare. Port B is where the magnitude is, and it is gated on a variance
measurement nobody has taken. Neither is worth starting before row 1 of the gate, which
costs a log tap.
