#!/usr/bin/env python3
"""
Vocabulary pruning for output head optimization.

Keeps the input embedding intact (safety: handles arbitrary input tokens).
Prunes the output head to a curated keep-set, trading decoder latency for 0.5 GB/token read.

Usage:
  python prune_vocab.py input.gguf output.gguf --keep-tokens tokens.txt [--keep-count 64000]

The keep-tokens file is line-delimited token IDs to retain.
Alternatively, --keep-count selects the top N by frequency (needs a calibration corpus).
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Set, Dict, List
import numpy as np

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gguf.gguf_reader import GGUFReader  # noqa: E402
from gguf.gguf_writer import GGUFWriter  # noqa: E402
from gguf.constants import GGUFValueType  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prune-vocab")


class VocabPruner:
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.reader = GGUFReader(input_path)
        self.hparams = self.reader.get_hparams()
        self.has_tied_embeddings = self._check_tied()
        self.n_vocab = self.hparams.get("vocab_size", 0)

    def _check_tied(self) -> bool:
        """Check if output.weight is present (untied) or missing (tied to token_embd)."""
        try:
            # If we can find output.weight tensor, embeddings are untied
            for key in self.reader.field_names("output"):
                if "output.weight" in key:
                    return False
            return True  # Tied: output.weight missing, will use token_embd
        except:
            return True

    def load_keep_tokens(self, keep_file: str) -> Set[int]:
        """Load token IDs to keep from a line-delimited file."""
        keep_tokens = set()
        with open(keep_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        keep_tokens.add(int(line))
                    except ValueError:
                        logger.warning(f"Skipping invalid token ID: {line}")
        return keep_tokens

    def build_remap(self, keep_tokens: Set[int]) -> Dict[int, int]:
        """
        Build token ID remap: pruned_logit_idx -> real_token_id.

        Returns a mapping so that:
          - Pruned models emit indices 0..len(keep_tokens)-1
          - Each is remapped to the original token ID at the sampler boundary
          - Tokens outside keep_tokens are never produced
        """
        keep_sorted = sorted(keep_tokens)
        remap = {i: tid for i, tid in enumerate(keep_sorted)}

        # Validation: check no gaps in the original token space
        if len(keep_sorted) < self.n_vocab:
            logger.info(f"Pruning {self.n_vocab} -> {len(keep_sorted)} vocab")

        return remap, set(keep_sorted)

    def prune_head_tensor(self, head_tensor: np.ndarray, keep_tokens: Set[int]) -> np.ndarray:
        """
        Prune output head weight matrix.

        Input shape: [n_embd, n_vocab]
        Output shape: [n_embd, len(keep_tokens)]

        Selects rows corresponding to keep_tokens, preserving order.
        """
        keep_sorted = sorted(keep_tokens)
        return head_tensor[:, keep_sorted].astype(np.float32)

    def write_pruned(self, output_path: str, keep_tokens: Set[int], remap: Dict[int, int]):
        """Write pruned GGUF with remapped output head."""
        keep_sorted = sorted(keep_tokens)

        if self.has_tied_embeddings:
            raise ValueError(
                "Model has tied embeddings (no output.weight). "
                "Head == input embedding; pruning would corrupt input handling. "
                "Run untie_embeddings() first to materialize a separate output.weight."
            )

        # Create writer, copy metadata
        writer = GGUFWriter(output_path, use_temp_file=True)

        # Copy all metadata from input
        for key in self.reader.fields:
            value = self.reader.get_field(key)
            writer.add_meta_value(key, value)

        # Update vocab size metadata
        writer.hparams["vocab_size"] = len(keep_tokens)

        # Copy tensors, pruning output.weight
        for name in self.reader.tensor_names:
            tensor_data = self.reader.get_tensor(name)

            if name == "output.weight":
                # Prune output head
                tensor_data = self.prune_head_tensor(tensor_data, keep_tokens)

            # Copy tensor as-is (or pruned)
            writer.add_tensor(name, tensor_data)

        # Write remap metadata for sampler
        remap_meta = {
            "pruned_vocab_size": len(keep_tokens),
            "original_vocab_size": self.n_vocab,
            "token_id_map": {str(i): keep_sorted[i] for i in range(len(keep_sorted))}
        }
        writer.add_meta_value("pruning_remap", json.dumps(remap_meta))

        writer.write_header_and_data(close=True)
        logger.info(f"Wrote pruned model to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prune GGUF vocabulary (output head only). Keeps input embedding intact."
    )
    parser.add_argument("input", help="Input GGUF model")
    parser.add_argument("output", help="Output pruned GGUF")
    parser.add_argument("--keep-tokens", required=True, help="File with token IDs to keep (one per line)")
    parser.add_argument("--keep-count", type=int, help="Alternatively: keep top N tokens by frequency")
    parser.add_argument("--calibration-corpus", help="Corpus to measure token frequency (with --keep-count)")

    args = parser.parse_args()

    pruner = VocabPruner(args.input)

    logger.info(f"Model vocab size: {pruner.n_vocab}")
    logger.info(f"Tied embeddings: {pruner.has_tied_embeddings}")

    if args.keep_tokens:
        keep_tokens = pruner.load_keep_tokens(args.keep_tokens)
    elif args.keep_count and args.calibration_corpus:
        # Future: frequency-based selection from corpus
        raise NotImplementedError("--keep-count with corpus not yet implemented")
    else:
        parser.print_help()
        sys.exit(1)

    remap, validated = pruner.build_remap(keep_tokens)
    pruner.write_pruned(args.output, keep_tokens, remap)


if __name__ == "__main__":
    main()
