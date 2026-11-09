# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""A blend whose small dataset has an empty validation split (its 0.2 % rounds to 0 documents):
the mid-level dataset would add epochs of 0 tokens forever (the v3 phase-1 / -2 prebuild hang);
the builder drops it from that split's blend and renormalizes the other weights."""

import os
import tempfile

import numpy
import pytest

from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import DType, IndexedDatasetBuilder
from megatron.core.datasets.utils import compile_helpers

EOD, VOCAB = 2, 1000


class _Tok:
    bos, eod, pad, vocab_size = None, EOD, 0, VOCAB
    unique_identifiers = {"class": "stub"}


def _write(prefix, n_docs, first_token):
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=DType.optimal_dtype(VOCAB))
    for i in range(n_docs):
        doc = [first_token] * 30 + [EOD]
        builder.add_document(numpy.array(doc), [len(doc)])
    builder.finalize(prefix + ".idx")


@pytest.fixture(scope="module")
def blend_dir():
    compile_helpers()
    d = tempfile.mkdtemp()
    _write(os.path.join(d, "big"), 2000, 11)   # valid split: 4 documents
    _write(os.path.join(d, "tiny"), 50, 22)    # valid split: 50 - round(0.998 x 50) = 0
    return d


def test_empty_valid_split_is_dropped_from_the_blend(blend_dir):
    big, tiny = os.path.join(blend_dir, "big"), os.path.join(blend_dir, "tiny")
    config = GPTDatasetConfig(
        random_seed=1234, sequence_length=16, blend=([big, tiny], [0.999, 0.001]), split="998,2,0",
        tokenizer=_Tok(), reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False,
        create_attention_mask=False, path_to_cache=os.path.join(blend_dir, "cache"),
    )
    train, valid, test = BlendedMegatronDatasetBuilder(
        GPTDataset, [200, 40, 0], lambda: True, config
    ).build()
    assert isinstance(valid, BlendedDataset) and len(valid) >= 40
    assert len(valid.datasets) == 1  # the tiny dataset is not in the validation blend
    tokens = {int(t) for i in range(len(valid)) for t in valid[i]["tokens"]}
    assert 22 not in tokens and 11 in tokens
    assert len(train.datasets) == 2  # the training blend keeps both


class _Forgetful(dict):
    """A count map that never keeps a value: the empty-split check re-opens every index (the
    behaviour before the recorded counts)."""

    def __setitem__(self, key, value):
        pass


def _build(blend_dir, cache, forget, monkeypatch):
    big, tiny = os.path.join(blend_dir, "big"), os.path.join(blend_dir, "tiny")
    config = GPTDatasetConfig(
        random_seed=1234, sequence_length=16, blend=([big, tiny], [0.999, 0.001]), split="998,2,0",
        tokenizer=_Tok(), reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False,
        create_attention_mask=False, path_to_cache=os.path.join(blend_dir, cache),
    )
    opened = []
    orig = GPTDataset.build_low_level_dataset

    def counting(prefix, cfg):
        opened.append(prefix)
        return orig(prefix, cfg)

    monkeypatch.setattr(GPTDataset, "build_low_level_dataset", staticmethod(counting))
    builder = BlendedMegatronDatasetBuilder(GPTDataset, [200, 40, 0], lambda: True, config)
    if forget:
        builder._num_elements = _Forgetful()
    datasets = builder.build()
    monkeypatch.undo()
    return datasets, opened


def test_empty_split_check_reuses_the_built_counts(blend_dir, monkeypatch):
    """The check reads the counts the mid-level build recorded: each index is opened once (not
    once more per split), and the blends are identical to re-opening every index."""
    (train, valid, _), opened = _build(blend_dir, "cache_new", False, monkeypatch)
    (train_ref, valid_ref, _), opened_ref = _build(blend_dir, "cache_ref", True, monkeypatch)
    assert sorted(opened) == sorted({p for p in opened})  # every prefix once
    assert len(opened) == 2 and len(opened_ref) == 2 + 2 * 2  # + both prefixes per blended split
    for new, ref in ((train, train_ref), (valid, valid_ref)):
        assert len(new.datasets) == len(ref.datasets)
        assert numpy.array_equal(new.dataset_index, ref.dataset_index)
        assert numpy.array_equal(new.dataset_sample_index, ref.dataset_sample_index)
        for i in (0, 1, len(new) - 1):
            assert numpy.array_equal(new[i]["tokens"], ref[i]["tokens"])
