# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""moe_router_lr_warmup_iters / adam_lr_warmup_iters: a per-group linear lr warmup (param_group
'lr_warmup_steps') on top of the schedule -- the router alone, or every parameter a Muon-family
optimizer hands to Adam, while the Muon-family matrices keep the schedule's lr."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.optimizer import (
    OptimizerConfig,
    _get_param_groups,
    _prepare_emerging_optimizer_params,
    get_standard_config_overrides,
)
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler


def _scheduler(groups, warmup=0):
    opt = MagicMock()
    opt.param_groups = groups
    return OptimizerParamScheduler(
        optimizer=opt, init_lr=0.0, max_lr=1e-3, min_lr=1e-3, lr_warmup_steps=warmup,
        lr_decay_steps=1000, lr_decay_style='constant', start_wd=0.1, end_wd=0.1,
        wd_incr_steps=1000, wd_incr_style='constant',
    )


@pytest.mark.parametrize("global_warmup", [0, 4])
def test_group_warmup_factor(global_warmup):
    plain, warm = {'lr': 0.0}, {'lr': 0.0, 'lr_warmup_steps': 10}
    sched = _scheduler([plain, warm], global_warmup)
    for step in range(1, 15):
        sched.step(1)
        base = 1e-3 * min(1.0, step / global_warmup) if global_warmup else 1e-3
        assert plain['lr'] == pytest.approx(base)
        assert warm['lr'] == pytest.approx(base * min(1.0, step / 10))


class _Router(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 8))


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = _Router()
        self.fc = torch.nn.Linear(8, 8, bias=False)
        self.norm = torch.nn.LayerNorm(8)


def _warmups(groups, net):
    names = {id(p): n for n, p in net.named_parameters()}
    return {names[id(p)]: g.get('lr_warmup_steps') for g in groups for p in g['params']}


@patch('torch.distributed.get_world_size', return_value=1)
@patch('torch.distributed.all_gather_object', lambda out, obj: out.__setitem__(0, obj))
def test_router_only(_ws):
    net = _Net()
    config = OptimizerConfig(optimizer='adam', lr=1e-3, moe_router_lr_warmup_iters=50)
    groups = _get_param_groups([net], config, get_standard_config_overrides(config=config))
    w = _warmups(groups, net)
    assert w['router.weight'] == 50
    assert w['fc.weight'] is None and w['norm.weight'] is None


@patch('torch.distributed.get_world_size', return_value=1)
@patch('torch.distributed.all_gather_object', lambda out, obj: out.__setitem__(0, obj))
@pytest.mark.parametrize("router", [0, 100])
def test_adam_params_under_muown(_ws, router):
    net = _Net()
    config = OptimizerConfig(optimizer='muown_classical', lr=1e-3, adam_lr_warmup_iters=100,
                             moe_router_lr_warmup_iters=router)
    overrides = get_standard_config_overrides(config=config)
    _prepare_emerging_optimizer_params(config, 'muown_classical', [net], overrides)
    groups = _get_param_groups([net], config, overrides)
    w = _warmups(groups, net)
    # the matrix stays on Muown at the schedule's lr; router and norm are Adam's and warm up
    assert w['fc.weight'] is None
    assert w['router.weight'] == 100 and w['norm.weight'] == 100 and w['norm.bias'] == 100


def test_unequal_warmups_rejected():
    config = OptimizerConfig(optimizer='muown_classical', lr=1e-3, adam_lr_warmup_iters=100,
                             moe_router_lr_warmup_iters=50)
    with pytest.raises(ValueError):
        get_standard_config_overrides(config=config)
