# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""GPTDatasetConfig.add_bos: every document reads as [BOS] + tokens (the tokenized data carries
only an EOD at each document end); off by default, and the cache key unchanged when off."""

import os
import tempfile

import numpy
import pytest

from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import DType, IndexedDataset, IndexedDatasetBuilder
from megatron.core.datasets.utils import Split, compile_helpers

BOS, EOD, PAD, VOCAB = 1, 2, 0, 1000
LENGTHS = [7, 13, 5, 21, 9, 17, 3, 11]  # document lengths, EOD included


class _Tok:
    bos, eod, pad, vocab_size = BOS, EOD, PAD, VOCAB
    unique_identifiers = {"class": "stub"}


@pytest.fixture(scope="module")
def indexed():
    compile_helpers()
    d = tempfile.mkdtemp()
    prefix = os.path.join(d, "docs")
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=DType.optimal_dtype(VOCAB))
    docs = []
    for i, n in enumerate(LENGTHS):
        doc = [10 + 50 * i + j for j in range(n - 1)] + [EOD]  # distinct tokens, EOD last
        docs.append(doc)
        builder.add_document(numpy.array(doc), [len(doc)])
    builder.finalize(prefix + ".idx")
    return IndexedDataset(prefix), docs, prefix, d


def _dataset(indexed, add_bos, seq=16):
    ds, docs, prefix, d = indexed
    config = GPTDatasetConfig(
        random_seed=1234, sequence_length=seq, split="100,0,0", tokenizer=_Tok(),
        reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False,
        create_attention_mask=False, inter_document_masking=True, add_bos=add_bos,
        path_to_cache=os.path.join(d, "cache_bos" if add_bos else "cache"),
    )
    return GPTDataset(ds, prefix, numpy.arange(len(LENGTHS)), None, Split.train, config)


def _pieces(sample, seq):
    tokens = sample["tokens"].numpy()
    cu = sample["cu_seqlens"].numpy()
    cu = cu[: int(numpy.argmax(cu)) + 1] if cu.max() > 0 else cu[:1]
    return [tokens[a:b] for a, b in zip(cu[:-1], cu[1:])]


@pytest.mark.parametrize("seq", [8, 16])
def test_every_document_starts_with_bos(indexed, seq):
    ds = _dataset(indexed, add_bos=True, seq=seq)
    _, docs, _, _ = indexed
    starts = {doc[0]: doc for doc in docs}
    seen_bos = 0
    for i in range(len(ds)):
        for piece in _pieces(ds[i], seq):
            assert (piece == BOS).sum() <= 1  # at most one document start per piece
            if piece[0] == BOS:
                seen_bos += 1
                if len(piece) > 1:  # BOS, then the document's own first tokens
                    doc = starts[int(piece[1])]
                    assert list(piece[1:]) == doc[: len(piece) - 1]
            else:
                assert BOS not in piece  # a BOS only ever opens a piece (a document)
    assert seen_bos > 0


def test_lengths_and_sample_count(indexed):
    ds = _dataset(indexed, add_bos=True, seq=16)
    plain = _dataset(indexed, add_bos=False, seq=16)
    # one BOS per document: sum(L + 1) tokens per epoch
    assert ds._get_num_tokens_per_epoch() == sum(LENGTHS) + len(LENGTHS)
    assert plain._get_num_tokens_per_epoch() == sum(LENGTHS)
    assert len(ds) >= len(plain)


def test_off_by_default_and_cache_key_unchanged(indexed):
    plain = _dataset(indexed, add_bos=False)
    for i in range(len(plain)):
        assert BOS not in plain[i]["tokens"].numpy()
    assert "add_bos" not in plain.unique_identifiers
    assert "add_bos" in _dataset(indexed, add_bos=True).unique_identifiers
    assert _dataset(indexed, add_bos=True).unique_description_hash != plain.unique_description_hash
