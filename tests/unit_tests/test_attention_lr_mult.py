# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""attention_lr_mult: the softmax-attention parameters (SelfAttention modules, ckda_branch attention
branches + their gamma) get their own lr group, and a resume that changes the multiplier loads on
the Megatron-FSDP path with the arguments' bounds (both directions)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.optimizer import OptimizerConfig, _get_param_groups, get_standard_config_overrides
from megatron.core.optimizer.distrib_optimizer import (
    DistributedOptimizer,
    reconcile_attention_lr_group_meta,
)
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.ssm.ckda_branch import BranchMix
from megatron.core.transformer.attention import mark_softmax_attention_params

LR, MIN_LR = 8.5e-4, 8.5e-4


class _Attn(torch.nn.Module):
    """The tensors of a gated attention layer: fused qkv with its input norm, proj, q / k norm."""

    def __init__(self, h=16, d=8):
        super().__init__()
        self.linear_qkv = torch.nn.Linear(h, 4 * d, bias=False)
        self.linear_qkv.layer_norm_weight = torch.nn.Parameter(torch.ones(h))
        self.linear_proj = torch.nn.Linear(2 * d, h, bias=False)
        self.q_layernorm = torch.nn.LayerNorm(d, bias=False)
        self.k_layernorm = torch.nn.LayerNorm(d, bias=False)
        mark_softmax_attention_params(self)


class _Net(torch.nn.Module):
    def __init__(self, h=16):
        super().__init__()
        self.attn = _Attn(h)
        self.mixer = torch.nn.Linear(h, h, bias=False)  # a complex_kda projection
        self.mixer_norm = torch.nn.Parameter(torch.ones(h))
        self.branch = _Attn(h)  # a ckda_branch attention branch
        cfg = SimpleNamespace(ckda_branch="attention", ckda_branch_calibrate=False)
        self.branch_mix = BranchMix(cfg, layer_number=10)
        self.fc = torch.nn.Linear(h, h, bias=False)


def _attn_params(net):
    return list(net.attn.parameters()) + list(net.branch.parameters()) + [net.branch_mix.gamma]


def _config(mult):
    return OptimizerConfig(
        optimizer='adam', lr=LR, min_lr=MIN_LR, attention_lr_mult=mult, qk_layernorm_wd_mult=0.0,
        residual_norm_wd_mult=0.0, weight_decay=0.0,
    )


def _groups(net, mult):
    with patch('torch.distributed.get_world_size', return_value=1), patch(
        'torch.distributed.all_gather_object', lambda out, obj: out.__setitem__(0, obj)
    ):
        cfg = _config(mult)
        return _get_param_groups([net], cfg, get_standard_config_overrides(config=cfg))


def test_tags():
    net = _Net()
    assert all(getattr(p, "is_softmax_attention", False) for p in _attn_params(net))
    for p in (net.mixer.weight, net.mixer_norm, net.fc.weight):
        assert not getattr(p, "is_softmax_attention", False)
    # a KronLSTM branch's gamma is not attention
    kron = BranchMix(SimpleNamespace(ckda_branch="kronlstm", ckda_branch_calibrate=False), 10)
    assert not getattr(kron.gamma, "is_softmax_attention", False)


@pytest.mark.parametrize("mult", [1.0, 4.0])
def test_groups(mult):
    net = _Net()
    groups = _groups(net, mult)
    attn = {id(p) for p in _attn_params(net)}
    covered = set()
    for g in groups:
        ids = {id(p) for p in g['params']}
        covered |= ids
        if mult == 1.0 or not ids & attn:
            assert 'attention_lr_mult' not in g
            assert g['max_lr'] == LR and g['min_lr'] == MIN_LR
        else:
            assert ids <= attn  # a group is all-attention or no-attention
            assert g['attention_lr_mult'] == mult and not g['default_config']
            assert g['max_lr'] == pytest.approx(LR * mult)
            assert g['min_lr'] == pytest.approx(MIN_LR * mult)
    assert covered == {id(p) for p in net.parameters()}
    if mult == 1.0:  # no extra group when off
        assert not any('attention_lr_mult' in g for g in groups)


def test_rejects_nonpositive():
    with pytest.raises(ValueError):
        get_standard_config_overrides(config=_config(0.0))


# ------------------------------------------------------------------------------------------------
# resume with a changed multiplier: the Megatron-FSDP branch of DistributedOptimizer.load_state_dict
# ------------------------------------------------------------------------------------------------


class _FakeFsdpOpt:
    """DistributedOptimizer's Megatron-FSDP load / save path around a torch optimizer."""

    load_state_dict = DistributedOptimizer.load_state_dict
    _param_groups_to_param2group_meta = DistributedOptimizer._param_groups_to_param2group_meta
    _param2group_meta_to_param_groups = DistributedOptimizer._param2group_meta_to_param_groups

    def __init__(self, net, mult):
        self.net = net
        self.ddp_config = SimpleNamespace(use_megatron_fsdp=True)
        self.optimizer = torch.optim.Adam(_groups(net, mult), lr=LR)
        # the scheduler sets every group's lr from its bounds at construction (before any load)
        self.scheduler = OptimizerParamScheduler(
            self.optimizer, init_lr=0.0, max_lr=LR, min_lr=MIN_LR, lr_warmup_steps=0,
            lr_decay_steps=1000, lr_decay_style='constant', start_wd=0.0, end_wd=0.0,
            wd_incr_steps=1000, wd_incr_style='constant', use_checkpoint_opt_param_scheduler=False,
            override_opt_param_scheduler=False)
        self._names = {id(p): n for n, p in net.named_parameters()}

    def _param_name(self, p):
        return self._names[id(p)]

    def saved(self):
        """What fsdp_dtensor saves: per-parameter group metadata (and an empty state here)."""
        meta = self._param_groups_to_param2group_meta(self.optimizer.param_groups)
        return {"state": {}, "param_to_group_meta": meta}

    def bounds(self):
        """(max_lr, min_lr, lr) of every parameter by name, as the optimizer holds them."""
        return {self._param_name(p): (g['max_lr'], g['min_lr'], g['lr'])
                for g in self.optimizer.param_groups for p in g['params']}


def _load(src_mult, dst_mult):
    net = _Net()
    saved = _FakeFsdpOpt(net, src_mult).saved()
    dst = _FakeFsdpOpt(net, dst_mult)
    before = dst.bounds()
    dst.load_state_dict(saved)
    return net, dst, before


@pytest.mark.parametrize("src,dst", [(1.0, 4.0), (4.0, 1.0), (4.0, 2.0), (4.0, 4.0), (1.0, 1.0)])
def test_resume_changed_multiplier(src, dst):
    net, opt, before = _load(src, dst)
    # the group structure is the current construction's, the bounds the arguments'
    assert opt.bounds() == before
    attn = {opt._param_name(p) for p in _attn_params(net)}
    for name, (max_lr, min_lr, lr) in opt.bounds().items():
        m = dst if name in attn else 1.0
        assert max_lr == pytest.approx(LR * m) and min_lr == pytest.approx(MIN_LR * m)
        assert lr == pytest.approx(LR * m)


def test_unchanged_multiplier_keeps_checkpoint_metadata():
    """Nothing is replaced when the multiplier did not change: the checkpoint's group metadata (here
    a changed min_lr, the constant-lr case of the override test) is loaded as before."""
    net = _Net()
    saved = _FakeFsdpOpt(net, 4.0).saved()
    saved["param_to_group_meta"] = {n: dict(m, min_lr=0.5 * m['min_lr'], marker=1)
                                    for n, m in saved["param_to_group_meta"].items()}
    dst = _FakeFsdpOpt(net, 4.0)
    out, changed = reconcile_attention_lr_group_meta(
        saved["param_to_group_meta"], dst.optimizer.param_groups, dst._param_name
    )
    assert changed == 0 and out == saved["param_to_group_meta"]
    dst.load_state_dict(saved)
    assert all(g.get('marker') == 1 for g in dst.optimizer.param_groups if g['params'])


def test_without_reconcile_the_load_fails_or_ignores(monkeypatch):
    """The failure modes the reconciliation removes (the plain Megatron-FSDP load path)."""
    import megatron.core.optimizer.distrib_optimizer as dopt

    monkeypatch.setattr(dopt, "reconcile_attention_lr_group_meta", lambda meta, groups, name: (meta, 0))
    # m -> 1: the attention parameters rejoin groups whose checkpoint metadata differs: strict error
    with pytest.raises(ValueError):
        _load(4.0, 1.0)
    # 1 -> m: loads, but the checkpoint's x1 bounds silently win
    net, opt, _ = _load(1.0, 4.0)
    name = opt._param_name(net.attn.linear_qkv.weight)
    assert opt.bounds()[name][0] == pytest.approx(LR)


def test_checkpoint_metadata_dicts_not_mutated():
    net = _Net()
    saved = _FakeFsdpOpt(net, 1.0).saved()
    snapshot = {n: dict(m) for n, m in saved["param_to_group_meta"].items()}
    dst = _FakeFsdpOpt(net, 4.0)
    out, changed = reconcile_attention_lr_group_meta(
        saved["param_to_group_meta"], dst.optimizer.param_groups, dst._param_name
    )
    assert changed == len(_attn_params(net))
    assert saved["param_to_group_meta"] == snapshot  # shared per-group dicts left alone
