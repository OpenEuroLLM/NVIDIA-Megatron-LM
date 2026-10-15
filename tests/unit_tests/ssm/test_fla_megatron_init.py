# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""linear_fla_megatron_init: Megatron's init for the self-contained FLA mixers."""

import pytest
import torch

from megatron.core.ssm.fla_hybrid import _FLAKDA, megatron_init_fla_layer
from megatron.core.transformer import TransformerConfig

pytestmark = pytest.mark.skipif(_FLAKDA is None, reason="needs fla.layers.kda")


def _layer():
    return _FLAKDA(hidden_size=256, head_dim=64, num_heads=4, mode="chunk", use_short_conv=True)


def _config(flag):
    return TransformerConfig(
        num_layers=8,
        hidden_size=256,
        num_attention_heads=4,
        init_method_std=0.02,
        linear_fla_megatron_init=flag,
    )


def test_off_by_default_keeps_fla_init():
    torch.manual_seed(0)
    layer = _layer()
    before = {n: p.clone() for n, p in layer.named_parameters()}
    megatron_init_fla_layer(layer, _config(False))
    assert all(torch.equal(before[n], p) for n, p in layer.named_parameters())


def test_projections_take_megatron_std_and_gates_stay():
    torch.manual_seed(0)
    layer = _layer()
    a_log, dt_bias = layer.A_log.clone(), layer.dt_bias.clone()
    megatron_init_fla_layer(layer, _config(True))
    # init_method std 0.02; output_layer_init_method std 0.02 / sqrt(2 * num_layers).
    assert abs(layer.q_proj.weight.std().item() - 0.02) < 2e-3
    assert abs(layer.o_proj.weight.std().item() - 0.02 / 4) < 1e-3
    assert torch.equal(layer.A_log, a_log) and torch.equal(layer.dt_bias, dt_bias)
