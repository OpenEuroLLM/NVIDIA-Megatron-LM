# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""--dataloader-index-bundle: the cached indices of a blend, bundled into one file, give the same
datasets (lengths, blend indices, every sample) as the cache files; an index the bundle lacks is
loaded from its file."""

import glob
import os
import shutil
import tempfile

import numpy
import pytest

from megatron.core.datasets import index_bundle
from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import DType, IndexedDatasetBuilder
from megatron.core.datasets.utils import compile_helpers

BOS, EOD, VOCAB, SEQ = 1, 2, 1000, 16
NUM_SAMPLES = [200, 20, 10]  # as the training: every split gets a size


class _Tok:
    bos, eod, pad, vocab_size = BOS, EOD, 0, VOCAB
    unique_identifiers = {"class": "stub"}


def _config(d, bundle=None):
    prefixes = [os.path.join(d, f"docs{i}") for i in range(3)]
    return GPTDatasetConfig(
        random_seed=1234, sequence_length=SEQ, blend=(prefixes, [0.5, 0.3, 0.2]), split="90,10,0",
        multiple_validation_sets=True, tokenizer=_Tok(), reset_position_ids=False,
        reset_attention_mask=False, eod_mask_loss=False, create_attention_mask=False,
        inter_document_masking=True, add_bos=True, path_to_cache=os.path.join(d, "cache"),
        index_bundle=bundle)


def _build(d, bundle=None):
    return BlendedMegatronDatasetBuilder(GPTDataset, NUM_SAMPLES, lambda: True, _config(d, bundle)).build()


def _flat(datasets):
    out = []
    for ds in datasets:
        out += [x for x in (ds if isinstance(ds, list) else [ds]) if x is not None]
    return out


def _assert_same(ref, got):
    ref, got = _flat(ref), _flat(got)
    assert [type(x) for x in ref] == [type(x) for x in got]
    for a, b in zip(ref, got):
        assert len(a) == len(b)
        if isinstance(a, BlendedDataset):
            assert numpy.array_equal(a.dataset_index, b.dataset_index)
            assert numpy.array_equal(a.dataset_sample_index, b.dataset_sample_index)
        for i in range(len(a)):
            x, y = a[i], b[i]
            assert x.keys() == y.keys()
            assert all(numpy.array_equal(numpy.asarray(x[k]), numpy.asarray(y[k])) for k in x)


@pytest.fixture()
def blend_dir():
    compile_helpers()
    d = tempfile.mkdtemp()
    rng = numpy.random.default_rng(0)
    for i in range(3):
        b = IndexedDatasetBuilder(os.path.join(d, f"docs{i}.bin"), dtype=DType.optimal_dtype(VOCAB))
        for _ in range(40):
            n = int(rng.integers(3, 30))
            b.add_document(numpy.r_[rng.integers(10, VOCAB, n - 1), EOD], [n])
        b.finalize(os.path.join(d, f"docs{i}.idx"))
    yield d
    shutil.rmtree(d)


def test_bundle_serves_the_same_datasets(blend_dir):
    d = blend_dir
    ref = _build(d)  # builds the cache
    index_bundle.start_recording()
    _build(d)  # all cache hits: every index the blend loads
    paths = index_bundle.recorded_paths()
    assert len(paths) == 3 * 3 * 2 + 2 * 2  # 3 indices x 3 datasets x (train, valid) + 2 blends x 2
    prefix = os.path.join(d, "bundle", "v1")
    index_bundle.write_bundle(prefix, paths, os.path.join(d, "cache"))

    for f in glob.glob(os.path.join(d, "cache", "*.npy")):  # from the bundle alone
        os.remove(f)
    before = index_bundle.load_counts()
    got = _build(d, bundle=prefix)
    after = index_bundle.load_counts()
    assert after["file"] == before["file"]
    assert after["bundle"] - before["bundle"] == len(paths)
    _assert_same(ref, got)


def test_index_missing_from_the_bundle_comes_from_its_file(blend_dir):
    d = blend_dir
    ref = _build(d)
    index_bundle.start_recording()
    _build(d)
    paths = index_bundle.recorded_paths()
    prefix = os.path.join(d, "bundle", "part")
    index_bundle.write_bundle(prefix, paths[1:], os.path.join(d, "cache"))
    before = index_bundle.load_counts()
    got = _build(d, bundle=prefix)
    assert index_bundle.load_counts()["file"] - before["file"] == 1  # paths[0] alone
    _assert_same(ref, got)


def test_bundle_bytes_equal_the_files(blend_dir):
    d = blend_dir
    _build(d)
    index_bundle.start_recording()
    _build(d)
    paths = index_bundle.recorded_paths()
    prefix = os.path.join(d, "bundle", "bytes")
    meta = index_bundle.write_bundle(prefix, paths, os.path.join(d, "cache"))
    b = index_bundle.IndexBundle(prefix)
    for p in paths:
        e = meta["entries"][os.path.basename(p)]
        assert e["offset"] % 4096 == 0
        v = b.get(p)
        assert not v.flags.writeable
        assert numpy.array_equal(v, numpy.load(p))
    assert b.get(os.path.join(d, "elsewhere", os.path.basename(paths[0]))) is None
