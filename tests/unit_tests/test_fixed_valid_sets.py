# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""--fixed-valid-data-path / --valid-seq-length: fixed held-out validation sets (one unweighted prefix each,
evaluated completely, at their own sequence length) next to a training blend with --split."""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.training import training
from megatron.training.arguments import apply_fixed_valid_args
from megatron.training.global_vars import set_args
from tests.unit_tests.data.test_prepare_cache import _build_null_tokenizer, _create_file_prefixes


def _args(**kw):
    a = SimpleNamespace(fixed_valid_data_path=None, valid_seq_length=None, validation_set_names=None,
                        multiple_validation_sets=False, full_validation=False, seq_length=65536,
                        micro_batch_size=1, global_batch_size=1024, eval_micro_batch_size=None,
                        eval_global_batch_size=None)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_apply_fixed_valid_args():
    a = _args(fixed_valid_data_path=["/x/all", "/x/code"], validation_set_names=["all", "code"],
              valid_seq_length=4096)
    apply_fixed_valid_args(a)
    assert a.multiple_validation_sets and a.full_validation
    assert (a.eval_micro_batch_size, a.eval_global_batch_size) == (16, 16384)   # 2^26 tokens per batch
    a = _args(seq_length=4096, global_batch_size=16384, micro_batch_size=16, valid_seq_length=4096)
    apply_fixed_valid_args(a)                       # the same length: nothing changes
    assert a.eval_micro_batch_size is None and not a.full_validation
    with pytest.raises(AssertionError):
        apply_fixed_valid_args(_args(fixed_valid_data_path=["/x/a"], validation_set_names=["a", "b"]))
    with pytest.raises(AssertionError):
        apply_fixed_valid_args(_args(seq_length=6000, valid_seq_length=4096))


def test_evaluate_runs_at_valid_seq_length(monkeypatch):
    seen = []
    monkeypatch.setattr(training, "_evaluate", lambda *a, **k: seen.append(training.get_args().seq_length) or "r")
    args = SimpleNamespace(seq_length=65536, valid_seq_length=4096)
    set_args(args)
    assert training.evaluate() == "r" and seen == [4096] and args.seq_length == 65536
    monkeypatch.setattr(training, "_evaluate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        training.evaluate()
    assert args.seq_length == 65536                 # restored on an error too
    args.valid_seq_length = None
    monkeypatch.setattr(training, "_evaluate", lambda *a, **k: seen.append(training.get_args().seq_length))
    training.evaluate()
    assert seen[-1] == 65536


def test_build_with_fixed_valid_sets():
    """Single process, no torch.distributed (the builder then runs everything on this rank)."""
    import pretrain_gpt
    from megatron.core.datasets.utils import compile_helpers

    compile_helpers()
    tokenizer = _build_null_tokenizer()
    with tempfile.TemporaryDirectory() as d:
        train = _create_file_prefixes(tokenizer, os.path.join(d, "train"), 3)
        valid = _create_file_prefixes(tokenizer, os.path.join(d, "valid"), 2)

        def cfg():
            return GPTDatasetConfig(random_seed=1234, sequence_length=32, blend=(train, [0.5, 0.3, 0.2]),
                                    split="90,10,0", tokenizer=tokenizer, reset_position_ids=False,
                                    reset_attention_mask=False, eod_mask_loss=False,
                                    path_to_cache=os.path.join(d, "cache"),
                                    multiple_validation_sets=True, full_validation=True)
        args = SimpleNamespace(fixed_valid_data_path=valid, valid_seq_length=8)
        train_cfg, valid_cfg = pretrain_gpt.fixed_valid_dataset_configs(cfg(), args)
        for attr in GPTDataset._key_config_attributes():          # the training cache key is unchanged
            assert getattr(train_cfg, attr) == getattr(cfg(), attr), attr
        train_ds, valid_ds, _ = pretrain_gpt.build_with_fixed_valid_sets(
            GPTDataset, [100, None, None], lambda: True, cfg(), args)
        assert len(train_ds) >= 100
        from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
        plain_train, _, _ = BlendedMegatronDatasetBuilder(GPTDataset, [100, 1, 1], lambda: True, cfg()).build()
        assert train_ds.unique_description_hash == plain_train.unique_description_hash   # the same cache
        assert isinstance(valid_ds, list) and len(valid_ds) == 2
        for v, prefix in zip(valid_ds, valid):
            assert v.config.sequence_length == 8
            n_tokens = 32 * 64                            # every document of the set, one epoch
            assert len(v) == (n_tokens - 1) // 8
            assert v.dataset.path_prefix == prefix
