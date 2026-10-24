# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""muon_qkv_headwise: per-head QKV split shapes and the tagging of linear-mixer projections."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import OptimizerConfig, _prepare_emerging_optimizer_params
from megatron.core.optimizer.emerging_optimizers import (
    TensorParallelMuownClassical,
    _get_qkv_split_shapes,
)


def _cfg(gate):
    return SimpleNamespace(num_attention_heads=6, num_query_groups=2, kv_channels=128,
                           attention_output_gate=gate)


@pytest.mark.parametrize("gate", [False, True])
def test_split_shapes(gate):
    grouped = _get_qkv_split_shapes(_cfg(gate))
    headwise = _get_qkv_split_shapes(_cfg(gate), headwise=True)
    assert sum(grouped) == sum(headwise)  # same per-group layout, finer blocks
    assert headwise == [128] * ((6 if gate else 3) + 2)


class _Mixer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(16, 16, bias=False)
        self.o_proj = torch.nn.Linear(16, 16, bias=False)
        self.q_proj.weight.headwise_dim = 4


class _Chunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=2, num_query_groups=1, kv_channels=4,
                                      attention_output_gate=False)
        self.mixer = _Mixer()
        self.linear_qkv = torch.nn.Linear(16, 16, bias=False)  # 2 q heads + k + v, 4 rows each


@pytest.mark.parametrize("headwise", [False, True])
def test_tagging(headwise):
    chunk = _Chunk()
    _prepare_emerging_optimizer_params(OptimizerConfig(optimizer="muon", lr=1e-3, muon_qkv_headwise=headwise), "muon", [chunk], {})
    q, o, qkv = chunk.mixer.q_proj.weight, chunk.mixer.o_proj.weight, chunk.linear_qkv.weight
    assert not getattr(o, "is_qkv", False)
    assert getattr(q, "is_qkv", False) == headwise
    if headwise:
        assert q.qkv_split_shapes == [4] * 4
        assert qkv.qkv_split_shapes == [4] * 4
    else:
        assert qkv.qkv_split_shapes == [8, 4, 4]


@pytest.mark.parametrize("batched", [False, True])
def test_param_split_shapes_override(batched):
    """A parameter's own qkv_split_shapes (the head-wise ones) win over the constructor's."""
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(12, 4) * 0.2)
    b = torch.nn.Parameter(a.detach().clone())
    b.qkv_split_shapes = (2, 2, 2)
    kw = dict(lr=0.01, scale_mode="spectral", extra_scale_factor=0.2, batched_step=batched,
              split_qkv=True, is_qkv_fn=lambda p: True, fp32_matmul_prec="highest")
    oa = TensorParallelMuownClassical([a], qkv_split_shapes=(2, 2, 2), **kw)
    ob = TensorParallelMuownClassical([b], qkv_split_shapes=(4, 1, 1), **kw)
    with torch._dynamo.config.patch(disable=True):
        for _ in range(3):
            g = torch.randn(12, 4)
            a.grad, b.grad = g.clone(), g.clone()
            oa.step(); ob.step()
    torch.testing.assert_close(a, b)
