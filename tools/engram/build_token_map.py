# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Build Engram's tokenizer compression map (--engram-token-map).

Every token id is mapped to a canonical id: the first id whose decoded text, after
NFKC normalisation and lowercasing, is the same (Engram, arXiv:2601.07372, which reports
a ~23% smaller effective vocabulary for a 128k tokenizer). The result is a .npy int64
array of length vocab_size; ids beyond the tokenizer's own vocabulary (padding up to
--vocab-size) map to themselves.

    python tools/engram/build_token_map.py --tokenizer <hf name or dir> --out map.npy \\
        [--vocab-size <padded vocab>]
"""

import argparse
import unicodedata

import numpy as np


def canonical(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def build(tokenizer, vocab_size: int) -> np.ndarray:
    first: dict[str, int] = {}
    mapping = np.arange(vocab_size, dtype=np.int64)
    for token_id in range(min(len(tokenizer), vocab_size)):
        key = canonical(tokenizer.decode([token_id]))
        mapping[token_id] = first.setdefault(key, token_id)
    return mapping


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--tokenizer", required=True, help="Hugging Face tokenizer name or path")
    p.add_argument("--out", required=True)
    p.add_argument("--vocab-size", type=int, default=None, help="padded model vocab size")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    mapping = build(tok, args.vocab_size or len(tok))
    np.save(args.out, mapping)
    distinct = len(np.unique(mapping[: len(tok)]))
    print(
        f"{args.out}: {len(tok)} tokens -> {distinct} canonical ids ({1 - distinct / len(tok):.1%} fewer)"
    )


if __name__ == "__main__":
    main()
