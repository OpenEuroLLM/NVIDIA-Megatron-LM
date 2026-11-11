# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""--override-opt_param-scheduler resets the per-group lr bounds an optimizer load restores."""

from types import SimpleNamespace

import torch

import megatron.training.checkpointing as ckpt
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler


def _setup(min_lr):
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([{'params': [p], 'default_config': True, 'max_lr': 8.5e-4, 'min_lr': min_lr}], lr=8.5e-4)
    g = 8
    sched = OptimizerParamScheduler(
        opt, init_lr=0.0, max_lr=8.5e-4, min_lr=0.0, lr_warmup_steps=0, lr_decay_steps=200 * g,
        lr_decay_style='WSD', start_wd=0.0, end_wd=0.0, wd_incr_steps=1000 * g, wd_incr_style='constant',
        use_checkpoint_opt_param_scheduler=False, override_opt_param_scheduler=True,
        wsd_decay_steps=100 * g, lr_wsd_decay_style='linear')
    return opt, sched, g


def _fake_load(opt, sched, g):
    """What a resume does: the optimizer state restores the checkpoint's group bounds (constant lr).

    A real ``torch.optim.Optimizer.load_state_dict`` (what every Megatron optimizer's load ends in)
    REPLACES each param-group dict, so the reset must address the groups the optimizer holds after
    the load, not the pre-load dicts (the c9c04235f version mutated those and was a no-op)."""
    saved = opt.state_dict()
    saved['param_groups'][0]['min_lr'] = 8.5e-4

    def impl(model, optimizer, scheduler, *a, **k):
        before = optimizer.param_groups[0]
        optimizer.load_state_dict(saved)
        assert optimizer.param_groups[0] is not before
        scheduler.num_steps = 100 * g
        scheduler.step(0)
        return 100, 0
    return impl


def test_override_resets_group_bounds(monkeypatch):
    opt, sched, g = _setup(0.0)
    monkeypatch.setattr(ckpt, '_load_checkpoint_impl', _fake_load(opt, sched, g))
    monkeypatch.setattr(ckpt, 'get_args', lambda: SimpleNamespace(override_opt_param_scheduler=True))
    assert ckpt.load_checkpoint(None, opt, sched) == (100, 0)
    assert opt.param_groups[0]['min_lr'] == 0.0
    sched.step(50 * g)
    assert abs(opt.param_groups[0]['lr'] - 8.5e-4 * 0.5) < 1e-12


def test_without_override_keeps_checkpoint_bounds(monkeypatch):
    opt, sched, g = _setup(0.0)
    monkeypatch.setattr(ckpt, '_load_checkpoint_impl', _fake_load(opt, sched, g))
    monkeypatch.setattr(ckpt, 'get_args', lambda: SimpleNamespace(override_opt_param_scheduler=False))
    ckpt.load_checkpoint(None, opt, sched)
    assert opt.param_groups[0]['min_lr'] == 8.5e-4
