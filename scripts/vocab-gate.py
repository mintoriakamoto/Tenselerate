#!/usr/bin/env python3
"""
vocab-gate.py - decide whether pruning the 248,320-token vocabulary is worth it,
                using the box's own traffic rather than an assumption.

THE QUESTION THIS ANSWERS
-------------------------
Trimming the output head saves bytes on every decode step, because the head is the
one vocabulary-sized matrix that is read in full per token. The pitch for it is:

    248,320 x 5120 = 1.27 B params. Trim to 64K and the per-token read drops.

What that pitch does not price is the other side. Removing tokens does not only
shrink the head - it changes the tokenizer, so text that used the removed tokens
re-encodes into MORE tokens. You save x% per token and then emit y% more tokens.
If y > x the change is a regression, and no weight-size calculation will show it.

This script measures both sides on real traffic and reports the net.

    net speedup = 1 / (inflation x (1 - saving_fraction)) - 1

EXACT vs ESTIMATED
------------------
  * Coverage and the byte/latency savings are EXACT arithmetic on your numbers.
  * Re-tokenization inflation is an ESTIMATE. Simulating a pruned BPE exactly would
    mean rebuilding the merge table; this uses greedy longest-match over the
    retained pieces instead. Greedy is what a pruned BPE approximates and it is
    biased slightly PESSIMISTIC (real BPE merges can do better), so a cut that
    looks net-positive here should stay positive in practice. A cut that looks
    marginal is marginal.

INPUTS
------
  --vocab     TSV of `id<TAB>piece`, one per line, for the whole vocabulary.
              From a GGUF:  python gguf-py/scripts/gguf_dump.py --json MODEL.gguf
              or any dump that gives you id and surface piece.
              Use --piece-format hex if pieces are hex-encoded bytes.

  --emitted   Token IDs the model GENERATED, whitespace-separated, from real
              traffic. This is the side that decides coverage: the head is read
              once per decode step, so what matters is what Hermes actually emits.
                  llama-tokenize -m MODEL.gguf -f responses.txt --ids > emitted.txt

  --prompt    (optional) Token IDs from the PROMPT side. Inflation here costs
              prefill time, not decode time, and is weighted accordingly.

Everything else has a default taken from the measured record for this box; pass
your own if they have moved.

USAGE
-----
    python3 -I vocab-gate.py --vocab vocab.tsv --emitted emitted.txt
    python3 -I vocab-gate.py --vocab vocab.tsv --emitted emitted.txt \
        --prompt prompt.txt --cuts 32768,65536,131072
"""

import argparse
import collections
import sys


# ---------------------------------------------------------------- loading


def load_vocab(path, piece_format):
    """id -> surface piece. Lines are `id<TAB>piece`; a blank piece is kept as ''."""
    vocab = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            try:
                tid = int(parts[0])
            except ValueError:
                continue
            piece = parts[1]
            if piece_format == "hex":
                try:
                    piece = bytes.fromhex(piece).decode("utf-8", errors="replace")
                except ValueError:
                    pass
            vocab[tid] = piece
    if not vocab:
        sys.exit(f"error: no `id<TAB>piece` lines parsed from {path}")
    return vocab


def load_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for chunk in fh.read().replace(",", " ").split():
            try:
                ids.append(int(chunk))
            except ValueError:
                continue
    if not ids:
        sys.exit(f"error: no token IDs parsed from {path}")
    return ids


# ---------------------------------------------------------------- inflation


def greedy_reencode_len(text, kept_pieces, max_piece_len):
    """
    Tokens a string costs using ONLY the retained pieces, by greedy longest match.

    Any character with no retained piece at all costs 1 token - the byte-fallback
    assumption. Keeping byte fallback is a hard requirement of any real pruning,
    so assuming it here is correct rather than generous; without it the vocabulary
    could not encode arbitrary input and the question would be moot.
    """
    n = len(text)
    i = 0
    count = 0
    while i < n:
        hi = min(max_piece_len, n - i)
        step = 0
        while hi > 0:
            if text[i:i + hi] in kept_pieces:
                step = hi
                break
            hi -= 1
        if step == 0:
            step = 1  # byte fallback
        count += 1
        i += step
    return count


def inflation_for_cut(ids, vocab, kept_ids, kept_pieces, max_piece_len, cache):
    """
    Token count of the same text after pruning, / token count before.

    Only tokens being REMOVED need re-encoding; retained tokens still cost 1 each,
    and a greedy pass over an unchanged region cannot do better than the original
    tokenizer did. Re-encoding only the removed runs is therefore the right unit
    and makes this tractable on a large corpus.
    """
    before = len(ids)
    after = 0
    run = []  # consecutive removed pieces re-encode together, not one at a time

    def flush():
        nonlocal after
        if not run:
            return
        text = "".join(run)
        hit = cache.get(text)
        if hit is None:
            hit = greedy_reencode_len(text, kept_pieces, max_piece_len)
            cache[text] = hit
        after += hit
        run.clear()

    for tid in ids:
        if tid in kept_ids:
            flush()
            after += 1
        else:
            run.append(vocab.get(tid, ""))
    flush()

    return after / before if before else 1.0, before, after


# ---------------------------------------------------------------- report


def main():
    ap = argparse.ArgumentParser(
        description="Measure whether vocabulary pruning is a net win on real traffic.")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--emitted", required=True,
                    help="token IDs the model generated (decides coverage)")
    ap.add_argument("--prompt", help="token IDs from the prompt side (optional)")
    ap.add_argument("--piece-format", choices=("raw", "hex"), default="raw")
    ap.add_argument("--cuts", default="32768,49152,65536,98304,131072")

    # measured constants for this box - override if they have moved
    ap.add_argument("--n-embd", type=int, default=5120)
    ap.add_argument("--head-bpw", type=float, default=6.5625,
                    help="bits/weight of output.weight (Q6_K inside Q4_K_M); "
                         "check with gguf-dump, it swings the answer ~30%%")
    ap.add_argument("--total-weight-gb", type=float, default=15.4)
    ap.add_argument("--weight-read-ms", type=float, default=18.5)
    ap.add_argument("--step-ms", type=float, default=37.6,
                    help="decode step at MTP depth 1 (1.88 accepted/step at 50 tok/s)")
    ap.add_argument("--accepted-per-step", type=float, default=1.88)
    ap.add_argument("--logit-rows-per-step", type=float, default=2.0,
                    help="logit rows crossing PCIe per step (MTP depth 1 = draft+verify)")
    ap.add_argument("--pcie-gbs", type=float, default=2.0)
    ap.add_argument("--prefill-tok-s", type=float, default=900.0)
    args = ap.parse_args()

    vocab = load_vocab(args.vocab, args.piece_format)
    emitted = load_ids(args.emitted)
    prompt = load_ids(args.prompt) if args.prompt else []

    v_orig = max(vocab) + 1
    freq = collections.Counter(emitted)
    if prompt:
        freq.update(prompt)
    ranked = [tid for tid, _ in freq.most_common()]

    print(f"vocabulary        : {v_orig:,} ids ({len(vocab):,} with pieces)")  # noqa: NP100
    print(f"emitted tokens    : {len(emitted):,}")  # noqa: NP100
    if prompt:
        print(f"prompt tokens     : {len(prompt):,}")  # noqa: NP100
    print(f"distinct ids used : {len(freq):,}  "  # noqa: NP100
          f"({len(freq) / v_orig * 100:.1f}% of the vocabulary)")

    ms_per_emitted = args.step_ms / args.accepted_per_step
    ms_per_prompt = 1000.0 / args.prefill_tok_s
    print(f"\ncost model: {ms_per_emitted:.1f} ms per emitted token, "  # noqa: NP100
          f"{ms_per_prompt:.2f} ms per prompt token\n")

    header = (f"{'cut':>8} {'cover':>7} {'head GB':>8} {'saved':>7} "
              f"{'infl-out':>9} {'infl-in':>8} {'net':>8}  verdict")
    print(header)  # noqa: NP100
    print("-" * len(header))  # noqa: NP100

    cache = {}
    best = None
    for cut in sorted(int(c) for c in args.cuts.split(",")):
        if cut >= v_orig:
            continue
        kept_ids = set(ranked[:cut])
        # anything unused in the corpus but still inside the cut budget is free to keep;
        # fill the remainder by id order so the kept set is exactly `cut` entries.
        if len(kept_ids) < cut:
            for tid in range(v_orig):
                if len(kept_ids) >= cut:
                    break
                kept_ids.add(tid)
        kept_pieces = {vocab[t] for t in kept_ids if vocab.get(t)}
        max_piece_len = max((len(p) for p in kept_pieces), default=1)

        coverage = sum(c for t, c in freq.items() if t in kept_ids) / sum(freq.values())

        infl_out, _, _ = inflation_for_cut(
            emitted, vocab, kept_ids, kept_pieces, max_piece_len, cache)
        infl_in = 1.0
        if prompt:
            infl_in, _, _ = inflation_for_cut(
                prompt, vocab, kept_ids, kept_pieces, max_piece_len, cache)

        head_gb = (v_orig - cut) * args.n_embd * args.head_bpw / 8 / 1e9
        saving_frac = head_gb / args.total_weight_gb
        ms_saved_weights = args.weight_read_ms * saving_frac
        logit_mb = (v_orig - cut) * 4 / 1e6
        ms_saved_pcie = logit_mb * args.logit_rows_per_step / (args.pcie_gbs * 1e3) * 1e3

        # per emitted token: the step shrinks by what we saved, then inflates by
        # however many more tokens the same output now costs.
        step_after = args.step_ms - ms_saved_weights - ms_saved_pcie
        ms_per_emitted_after = (step_after / args.accepted_per_step) * infl_out
        net = ms_per_emitted / ms_per_emitted_after - 1

        if prompt:
            # prefill inflation is real wall-clock too, just not per-token
            extra_prefill_ms = (infl_in - 1) * len(prompt) * ms_per_prompt
        else:
            extra_prefill_ms = 0.0

        if net > 0.02:
            verdict = "WORTH IT"
        elif net > 0:
            verdict = "marginal"
        else:
            verdict = "REGRESSION"
        if best is None or net > best[1]:
            best = (cut, net, coverage, infl_out, extra_prefill_ms)

        print(f"{cut:>8,} {coverage*100:>6.2f}% {head_gb:>7.2f} "  # noqa: NP100
              f"{saving_frac*100:>6.2f}% {infl_out:>8.4f} {infl_in:>7.4f} "
              f"{net*100:>+7.2f}%  {verdict}")

    print()  # noqa: NP100
    if best is None:
        sys.exit("no cut below the vocabulary size was evaluated")

    cut, net, coverage, infl_out, extra_prefill_ms = best
    if net <= 0:
        print(f"VERDICT: no cut tested is a net win. The best, {cut:,}, is "  # noqa: NP100
              f"{net*100:+.2f}%.\nRe-tokenization costs more than the head saves. "
              f"Close the item; the head is not where the bytes are.")
    else:
        print(f"VERDICT: best cut is {cut:,} at {net*100:+.2f}% decode "  # noqa: NP100
              f"({coverage*100:.2f}% coverage, {(infl_out-1)*100:+.2f}% more "
              f"output tokens).")
        if extra_prefill_ms:
            print(f"         prefill pays {extra_prefill_ms/1000:.1f} s extra over "  # noqa: NP100
                  f"this corpus - check that against how often you prefill cold.")
        if net < 0.02:
            print("         Under 2%. Pruning means a conversion, a re-quantization "  # noqa: NP100
                  "and a tokenizer that can no longer round-trip arbitrary text.\n"
                  "         That is a lot of risk for this much; I would not.")

    print("\nReminder: inflation is a greedy-longest-match estimate and is biased\n"  # noqa: NP100
          "pessimistic. Coverage and the byte savings are exact.")


if __name__ == "__main__":
    main()
