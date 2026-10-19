# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""param.contiguous_group: a group of params is one back-to-back block, in group order, in one
bucket, in every buffer layout (plain DDP, distributed optimizer, layer-wise optimizer) -- what
MoonEP's per-expert weights need for their [n, ...] weight / main-grad views."""

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import _compute_default_per_buffer_param_layout
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer


def _params():
    """a, b, then a 5-expert group (registered in expert order), then c."""
    mk = lambda n: torch.nn.Parameter(torch.empty(n))  # noqa: E731
    a, b, c = mk(64 * 7), mk(64 * 3), mk(64 * 5)
    group = []
    for i in range(5):
        p = mk(64 * 10)
        p.contiguous_group = ("experts", i, 5)
        group.append(p)
    return [a, b, *group, c], group


def _check(index_map, group):
    starts = [index_map[p][0] for p in group]
    ends = [index_map[p][1] for p in group]
    assert all(e == s for e, s in zip(ends[:-1], starts[1:])), (starts, ends)  # back to back
    assert starts == sorted(starts)  # in group (expert) order
    assert len({index_map[p][2] for p in group}) == 1  # one bucket


@pytest.mark.parametrize("bucket_size", [None, 64 * 4, 64 * 25])
def test_default_layout(bucket_size):
    params, group = _params()
    layout = _compute_default_per_buffer_param_layout(params, bucket_size)
    _check(layout.param_index_map, group)


@pytest.mark.parametrize("bucket_size", [None, 64 * 4, 64 * 25])
def test_distributed_optimizer_layout(bucket_size):
    params, group = _params()
    cfg = DistributedDataParallelConfig(use_distributed_optimizer=True)
    layout = DistributedOptimizer._compute_per_buffer_param_layout(params, bucket_size, 2, cfg)
    _check(layout.param_index_map, group)


@pytest.mark.parametrize("bucket_size", [None, 64 * 4, 64 * 25])
@pytest.mark.parametrize("dp", [2, 4])
def test_layer_wise_layout(bucket_size, dp):
    params, group = _params()
    cfg = DistributedDataParallelConfig(use_distributed_optimizer=True)
    layout = LayerWiseDistributedOptimizer._compute_per_buffer_param_layout(
        params, bucket_size, dp, cfg
    )
    _check(layout.param_index_map, group)
    # and every param still has a place, none overlapping
    spans = sorted(layout.param_index_map[p][:2] for p in params)
    assert all(e <= s for (_, e), (s, _) in zip(spans[:-1], spans[1:]))
