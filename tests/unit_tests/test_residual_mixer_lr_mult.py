# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""residual_mixer_lr_mult: the residual-stream mixers' parameters get their own LR group."""

from unittest.mock import patch

import pytest
import torch

from megatron.core.optimizer import (
    OptimizerConfig,
    _get_param_groups,
    get_standard_config_overrides,
)
from megatron.core.transformer.attn_res import BlockAttnRes
from megatron.core.transformer.mhc import MHCMixer
from megatron.core.transformer.routed_residual import RoutedStreamMixer


class _Net(torch.nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.fc = torch.nn.Linear(hidden, hidden, bias=False)
        self.mhc = MHCMixer(hidden, expansion_rate=4)
        self.routed = RoutedStreamMixer(hidden, expansion_rate=4, topk=2, key_dim=4)
        self.attn_res = BlockAttnRes(hidden)


def _mixer_params(net):
    return [p for m in (net.mhc, net.routed, net.attn_res) for p in m.parameters(recurse=False)]


def test_all_mixer_params_tagged():
    net = _Net()
    assert all(getattr(p, "is_residual_mixer", False) for p in _mixer_params(net))
    assert not getattr(net.fc.weight, "is_residual_mixer", False)


@patch('torch.distributed.get_world_size', return_value=1)
@patch(
    'torch.distributed.all_gather_object', lambda output_list, obj: output_list.__setitem__(0, obj)
)
@pytest.mark.parametrize("mult", [1.0, 0.25])
def test_mixer_lr_group(_ws, mult):
    net = _Net()
    config = OptimizerConfig(optimizer='adam', lr=4e-3, min_lr=4e-3, residual_mixer_lr_mult=mult)
    groups = _get_param_groups([net], config, get_standard_config_overrides(config=config))
    mixer = {id(p) for p in _mixer_params(net)}
    for g in groups:
        ids = {id(p) for p in g['params']}
        if mult == 1.0 or not ids & mixer:
            # untouched: the default lr (max_lr None means the scheduler's lr)
            assert g['max_lr'] in (None, 4e-3)
        else:
            # a group is all-mixer or no-mixer, and the mixer groups carry the scaled lr
            assert ids <= mixer
            assert g['max_lr'] == pytest.approx(4e-3 * mult)
            assert g['min_lr'] == pytest.approx(4e-3 * mult)
    covered = set().union(*({id(p) for p in g['params']} for g in groups))
    assert covered == {id(p) for p in net.parameters()}
