# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Checkpoint layout handling of chained optimizers (no distributed setup needed)."""

from types import SimpleNamespace

import pytest

from megatron.core.optimizer import ChainedOptimizer
from megatron.training.checkpointing import _build_sharded_state_dict_metadata


def _chain(n):
    chain = ChainedOptimizer.__new__(ChainedOptimizer)
    chain.chained_optimizers = [object() for _ in range(n)]
    return chain


def _entry():
    return {"param_keyed": True, "param_groups": {"0": {"lr": 1.0}}}


def test_nested_chain_with_matching_count_loads_unchanged():
    """LayerWiseDistributedOptimizer (itself a chain) next to the Engram table
    optimizer: {0: {0: ..., 1: ...}, 1: ...} must not be re-split."""
    state = {0: {0: _entry(), 1: _entry()}, 1: _entry()}
    assert _chain(2)._match_saved_optimizer_layout(state) is state


def test_nested_chain_with_different_count_is_an_error():
    state = {0: {0: _entry(), 1: _entry()}, 1: _entry()}
    with pytest.raises(RuntimeError, match="nested chains"):
        _chain(3)._match_saved_optimizer_layout(state)


def _args(**kw):
    base = dict(
        use_distributed_optimizer=False,
        use_layer_wise_distributed_optimizer=False,
        ckpt_format="torch_dist",
        dist_ckpt_optim_fully_reshardable=False,
        distrib_optim_fully_reshardable_mem_efficient=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("distopt", [False, True])
def test_fully_reshardable_flag_reaches_the_metadata(distopt):
    meta = _build_sharded_state_dict_metadata(
        _args(use_distributed_optimizer=distopt, dist_ckpt_optim_fully_reshardable=True),
        dp_cp_group=object(),
    )
    assert meta["distrib_optim_sharding_type"] == "fully_reshardable"


def test_non_distributed_optimizer_without_flag_keeps_legacy_format():
    meta = _build_sharded_state_dict_metadata(_args(), dp_cp_group=object())
    assert "distrib_optim_sharding_type" not in meta


def test_current_layout_keys_merged_in_do_not_count():
    """Loading merges the current layout's (param-state only) entries into the saved
    common state: a saved single entry against two current optimizers is re-split."""
    state = {"param_keyed": True, "param_groups": {"0": {"lr": 1.0, "is_expert_parallel": False}},
             0: {"param_state": {}}, 1: {"param_state": {}}}
    chain = _chain(2)

    class _Opt:
        def __init__(self, ep):
            self.param_groups = [{"is_expert_parallel": ep}]

    chain.chained_optimizers = [_Opt(False), _Opt(True)]
    out = chain._match_saved_optimizer_layout(state)
    assert set(out) == {0, 1}
    assert out[1]["param_groups"]["0"]["is_expert_parallel"] is True
