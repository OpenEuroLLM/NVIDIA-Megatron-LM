# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""--packing-plan-path: PackingPlanDataset serves plan sequences for the listed sample indices (boundary
modes doc / group / none) and the wrapped dataset's samples for every other index."""

import json
import os
import tempfile

import numpy
import pytest

from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import DType, IndexedDataset, IndexedDatasetBuilder
from megatron.core.datasets.packing_plan import PackingPlanDataset
from megatron.core.datasets.utils import Split, compile_helpers

BOS, EOD, VOCAB, SEQ = 1, 2, 1000, 16
LENGTHS = [7, 13, 5, 21, 9, 17, 3, 11, 6, 8]


class _Tok:
    bos, eod, pad, vocab_size = BOS, EOD, 0, VOCAB
    unique_identifiers = {"class": "stub"}


@pytest.fixture(scope="module")
def gpt():
    compile_helpers()
    d = tempfile.mkdtemp()
    prefix = os.path.join(d, "docs")
    b = IndexedDatasetBuilder(prefix + ".bin", dtype=DType.optimal_dtype(VOCAB))
    for i, n in enumerate(LENGTHS):
        b.add_document(numpy.array([10 + 50 * i + j for j in range(n - 1)] + [EOD]), [n])
    b.finalize(prefix + ".idx")
    cfg = GPTDatasetConfig(
        random_seed=1234, sequence_length=SEQ, split="100,0,0", tokenizer=_Tok(),
        reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False,
        create_attention_mask=False, inter_document_masking=True, add_bos=True,
        path_to_cache=os.path.join(d, "cache"))
    ds = GPTDataset(IndexedDataset(prefix), prefix, numpy.arange(len(LENGTHS)), None, Split.train, cfg)
    return ds, d


def _plan(gpt_ds, d, slots):
    """One plan sequence per listed slot: doc 3 from its BOS (8 tokens, group 0), doc 1 (BOS + 4
    tokens, group 0), doc 5 (BOS + 3 tokens + the label token, group 1) = 17 tokens."""
    p = os.path.join(d, "plan")
    os.makedirs(p, exist_ok=True)
    seg = dict(ds=[0, 0, 0], doc=[3, 1, 5], off=[0, 0, 0], len=[8, 5, 4], group=[0, 0, 1])
    n = len(slots)
    numpy.save(os.path.join(p, "slot_bidx.npy"), numpy.asarray(sorted(slots), numpy.int64))
    numpy.save(os.path.join(p, "slot_seq.npy"), numpy.arange(n, dtype=numpy.int64))
    numpy.save(os.path.join(p, "seq_ptr.npy"), numpy.arange(n + 1, dtype=numpy.int64) * 3)
    for k, v in seg.items():
        numpy.save(os.path.join(p, f"seg_{k}.npy"), numpy.tile(numpy.asarray(v, numpy.int64), n))
    with open(os.path.join(p, "meta.json"), "w") as f:
        json.dump({"format": 1, "blend_md5": gpt_ds.unique_description_hash, "seq_length": SEQ}, f)
    return p


@pytest.mark.parametrize("mode,cu", [("doc", [0, 8, 13, 16]), ("group", [0, 13, 16]), ("none", [0, 16])])
def test_plan_sequences_and_fallback(gpt, mode, cu):
    ds, d = gpt
    slots = [1, 3]
    wrapped = PackingPlanDataset(ds, [_plan(ds, d, slots)], mode)
    assert len(wrapped) == len(ds)
    expect = numpy.concatenate([ds._get_document(3, 0, 8), ds._get_document(1, 0, 5), ds._get_document(5, 0, 4)])
    for i in range(len(ds)):
        got = wrapped[i]
        if i in slots:
            assert got["tokens"].tolist() == expect[:-1].tolist()
            assert got["labels"].tolist() == expect[1:].tolist()
            c = got["cu_seqlens"].numpy()
            assert c[: len(cu)].tolist() == cu and (c[len(cu):] == SEQ).all()
            for a, b in zip(cu[:-1], cu[1:]):
                assert got["position_ids"][a:b].tolist() == list(range(b - a))
        else:
            ref = ds[i]
            assert all(numpy.array_equal(numpy.asarray(got[k]), numpy.asarray(ref[k])) for k in ref)


def test_plan_for_another_blend_is_refused(gpt):
    ds, d = gpt
    p = _plan(ds, d, [0])
    with open(os.path.join(p, "meta.json"), "w") as f:
        json.dump({"format": 1, "blend_md5": "other", "seq_length": SEQ}, f)
    with pytest.raises(AssertionError, match="plan built for blend"):
        PackingPlanDataset(ds, [p], "none")
