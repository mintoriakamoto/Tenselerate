# The Plinian rule

Pliny the Elder wrote the first work to index its own sources. He named who told him
what, and he separated what he had seen from what he had been told. He is also the
reason we know that is not enough: the *Naturalis Historia* preserved a great many
false things faithfully, with attribution, for fifteen centuries.

That is the whole discipline in one figure.

> **A citation transfers provenance, not truth. A cited falsehood is still a falsehood,
> and it is more dangerous than an uncited one, because it has stopped looking like a
> guess.**

This repository has already paid for that lesson. `rig-cmp170hx-3060.md` asserted
"INT8 / dp4a: uncrippled". `kernel-readonce.md` cited it correctly and made it the
central design decision of a kernel estimated at days of work. The citation was
honest; the claim had never been measured on this unit, and the build preset the box
actually ships compiles `GGML_CUDA_DISABLE_DP4A=ON` — which only exists if the
opposite is true. The reference laundered an unverified claim into a foundation.

So every claim in this repository carries two things that are not optional: **where it
came from**, and **what it rests on**.

## The score

Four axes, 0-3 each, 11 maximum. Put the vector next to any claim that other work
will stand on.

**P - Provenance.** Where it came from.

| | |
| --- | --- |
| 0 | asserted from model knowledge or recall |
| 1 | cited to a document in this repo |
| 2 | read from code or config, at a named file and line |
| 3 | reproduced by a measurement recorded in this repo |

**F - Falsifiability.** What would kill it.

| | |
| --- | --- |
| 0 | no refutation stated; unfalsifiable as written |
| 1 | falsifiable in principle, nothing specified |
| 2 | refutation pre-registered with a numeric threshold, before the run |
| 3 | pre-registered refutation has been run and survived |

**M - Magnitude.** What it is worth if true.

| | |
| --- | --- |
| 0 | under 1%; changes no decision |
| 1 | 1-5% |
| 2 | 5-30%, or removes a constraint believed to be binding |
| 3 | over 30%, structural, or invalidates other work |

**D - Dependence.** What it stands on.

| | |
| --- | --- |
| 0 | rests on at least one claim that is not CONFIRMED |
| 1 | rests only on CONFIRMED claims |
| 2 | self-contained: the measurement is the claim |

**Status** is separate from the score and is one of: `CONFIRMED`, `PLAUSIBLE`,
`UNVERIFIED`, `CONTRADICTED`, `REFUTED`, `RETRACTED`.

## Three rules that follow, and are not negotiable

**1. The inheritance cap.** `D=0` caps status at `PLAUSIBLE`, whatever P, F and M say.
Nothing is more certain than its foundation. When a foundation is resolved, every
claim above it must be re-scored — and the re-score is the work, not a formality.

**2. Magnitude is not evidence.** High M with low P is the most dangerous cell in the
table, not the most exciting one. A 3x prize resting on a doc-cited number is a 3x
prize resting on a doc-cited number. **Rank what to work on by P+F. Rank what to
measure first by M.** Those are different orderings and conflating them is how weeks
disappear.

**3. A contradiction is a finding, and it outranks a theory.** Two sources in this
repo disagreeing is `CONFIRMED` the moment both have been read, even with neither
resolved. It is a known defect in the foundation, so it ranks above any new idea built
on top of that foundation. Hunt for these first: they are cheap to find, they are
certain, and they invalidate work that has not been done yet.

**4. Verify the instrument before you count the test.** `F=2` means a refutation is
pre-registered *and the thing that would run it works*. A pre-registered test with an
unchecked instrument is `F=1`, and it is more dangerous than no test, because it comes
back with a confident wrong answer instead of nothing.

This rule is here because this repository's own ledger broke it. Its top-ranked item -
the one command said to resolve four capped claims, repeated five times - was
`llama-cli --version | grep DISABLE_DP4A`. The flag is absent from upstream and from
the CUDA feature list the binary exposes, so the command could only ever print nothing
for it, and nothing would have been read as "the flag is off". The test was
pre-registered. The instrument did not exist. Check that a command prints what you
think it prints, on the machine you think it runs on, before you spend a ranking on it.

## Verify, then refute

For any claim that other work will stand on, in this order:

1. **Source it.** Name the file and line, or the measurement. If neither exists, the
   claim is `P=0` and must say so in the same sentence it is made.
2. **State what would kill it**, with a number, *before* looking. A refutation written
   after the result is not a refutation.
3. **Check the instrument.** Does the command print what you think it prints? Does the
   flag exist in the build you will run it against? Confirm this before step 4, not
   after a surprising result.
4. **Try to kill it.** Prefer the cheap test that could close the whole idea over the
   expensive one that would confirm it. Ten minutes that can end a week is the best
   trade available.
5. **Record the refutation that did not fire.** A survived prediction is evidence; an
   unrecorded one is a story.
6. **Re-score what stands on it.**

## Retraction

Retracting costs one sentence. Carrying a false claim costs whatever gets built on it.

State the correction plainly, in the place the claim lives, and fix the places that
cite it in the same change. Do not leave a corrected doc pointing at an uncorrected
one. Do not soften a retraction into a caveat — if it is wrong, it is wrong, and the
next reader needs to know that without reading the history.

Claims this repository has retracted are listed in `docs/veritas-ledger.md` with what
they cost. That list is an asset, not an embarrassment: it is the only record of which
kinds of mistake this project actually makes.

## What this is not

It is not a reason to stop proposing things. Speculation scored honestly as
speculation is useful and cheap. A document that says "`P=0, F=2, M=3`, here is the
ten-minute test that closes it" is doing its job perfectly.

The failure mode is not having ideas. It is an idea that has quietly stopped being
labelled as one.
