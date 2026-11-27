# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""The fsdp_dtensor save-path cache (megatron/core/transformer/fsdp_dtensor_checkpoint.py cached_split): a split of
a source is reused while the source is the same storage / layout, only on the save path, never for copies, and a
source that keeps moving falls back to the uncached path."""
import pytest
import torch

from megatron.core.transformer import fsdp_dtensor_checkpoint as fdc


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setattr(fdc, "SAVE_CACHE", True)
    fdc._SPLIT_CACHE.clear()
    fdc._SPLIT_UNCACHEABLE.clear()
    fdc._SPLIT_MOVES.clear()
    yield
    fdc._SPLIT_CACHE.clear()
    fdc._SPLIT_UNCACHEABLE.clear()
    fdc._SPLIT_MOVES.clear()


def _views(t):
    flat = t.view(-1)
    n = flat.numel() // 2
    return flat[:n], flat[n:]


def test_reused_while_the_source_is_the_same_storage_and_shows_current_values():
    src, calls = torch.arange(8.0), []
    compute = lambda: calls.append(1) or _views(src)  # noqa: E731
    with fdc.save_cache_active():
        w1, v1 = fdc.cached_split(("model", "k"), src, None, compute)
        w2, v2 = fdc.cached_split(("model", "k"), src, None, compute)
    assert len(calls) == 1 and w1 is w2 and v1 is v2
    src.add_(100.0)  # the next save: the views show the trained values
    assert torch.equal(w2, torch.arange(4.0) + 100.0)


def test_inactive_outside_the_save_path():
    src, calls = torch.arange(8.0), []
    compute = lambda: calls.append(1) or _views(src)  # noqa: E731
    fdc.cached_split(("model", "k"), src, None, compute)
    fdc.cached_split(("model", "k"), src, None, compute)
    assert len(calls) == 2 and not fdc._SPLIT_CACHE


def test_a_moved_source_is_recomputed_once_then_never_cached():
    calls = []
    with fdc.save_cache_active():
        for i in range(4):  # a fresh tensor every "save"
            src = torch.arange(8.0) + i
            w, _ = fdc.cached_split(("opt", "k", "param"), src, None, lambda: calls.append(1) or _views(src))
            assert torch.equal(w, torch.arange(4.0) + i)
    assert len(calls) == 4
    assert ("opt", "k", "param") in fdc._SPLIT_UNCACHEABLE and not fdc._SPLIT_CACHE


def test_one_move_after_the_load_is_tolerated():
    first, second, calls = torch.arange(8.0), torch.arange(8.0) * 2, []
    with fdc.save_cache_active():
        fdc.cached_split(("opt", "k", "param"), first, None, lambda: calls.append(1) or _views(first))
        fdc.cached_split(("opt", "k", "param"), second, None, lambda: calls.append(1) or _views(second))
        w, _ = fdc.cached_split(("opt", "k", "param"), second, None, lambda: calls.append(1) or _views(second))
    assert len(calls) == 2 and torch.equal(w, torch.arange(4.0) * 2)


def test_copies_are_never_cached():
    src = torch.arange(8.0)
    with fdc.save_cache_active():
        fdc.cached_split(("opt", "k", "rows"), src, None, lambda: tuple(p.clone() for p in _views(src)))
    assert ("opt", "k", "rows") in fdc._SPLIT_UNCACHEABLE and not fdc._SPLIT_CACHE


def test_swiglu_key_patterns():
    assert fdc._is_swiglu_key("decoder.layers.3.mlp.experts.linear_fc1.weight17")
    assert fdc._is_swiglu_key("decoder.layers.3.mlp.shared_experts.linear_fc1.weight")
    assert not fdc._is_swiglu_key("decoder.layers.3.mlp.experts.linear_fc2.weight17")
    assert not fdc._is_swiglu_key("decoder.layers.3.self_attention.linear_qkv.weight")
