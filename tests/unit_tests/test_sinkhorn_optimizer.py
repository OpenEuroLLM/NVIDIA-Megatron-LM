# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""SinkhornMomentum (DeepSeek-V4.1 Algorithm 1) and its routing as embedding_head_optimizer."""

from unittest.mock import patch

import pytest
import torch

from megatron.core.optimizer import OptimizerConfig, _get_param_groups, get_standard_config_overrides
from megatron.core.optimizer.sinkhorn import SinkhornMomentum


def _reference(w, grads, lr=0.01, beta=0.95, iters=11, tau=1e-3, eps=1e-20, gamma=0.18):
    """Algorithm 1, literally."""
    w = w.clone(); m = torch.zeros_like(w); n = w.shape[1]
    for g in grads:
        m = beta * m + (1 - beta) * g
        gh = beta * m + (1 - beta) * g
        rho = gh.norm(dim=1)
        u = gh.clone(); u[rho <= tau * rho.mean()] = 0
        for k in range(1, iters + 1):
            if k % 2:
                u = u / (u.norm(dim=1, keepdim=True) + eps)
            else:
                u = u / (u.norm(dim=0, keepdim=True) + eps)
        w = w - gamma * lr * (n ** 0.5) * u
    return w


def test_matches_algorithm_1():
    torch.manual_seed(0)
    w0 = torch.randn(50, 8, dtype=torch.float64)
    grads = [torch.randn(50, 8, dtype=torch.float64) for _ in range(4)]
    for g in grads:
        g[::7] = 0.0  # absent tokens: masked rows
    p = torch.nn.Parameter(w0.clone())
    opt = SinkhornMomentum([p], lr=0.01)
    for g in grads:
        p.grad = g.clone(); opt.step()
    torch.testing.assert_close(p.detach(), _reference(w0, grads))


def test_balanced_update():
    torch.manual_seed(1)
    g = torch.randn(400, 16) * (0.5 + torch.rand(400, 1)) * 5 * torch.rand(1, 16)
    g[:3] = 1e-9  # absent tokens: masked to a zero step
    d = SinkhornMomentum.balance(g, iters=11, tau=1e-3, eps=1e-20)
    assert d[:3].abs().max() == 0
    rows = d[3:].pow(2).mean(dim=1).sqrt()
    torch.testing.assert_close(rows, torch.ones_like(rows), atol=1e-6, rtol=1e-6)
    # columns balanced too: each column's mean square over the live rows ~ 1
    assert (d[3:].pow(2).mean(dim=0) - 1).abs().max() < 0.05


def test_rejects_even_iters():
    with pytest.raises(ValueError):
        SinkhornMomentum([torch.nn.Parameter(torch.zeros(2, 2))], iters=10)


class _Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(32, 8)
        self.word_embeddings.weight.is_embedding_or_output_parameter = True


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = _Emb()
        self.fc = torch.nn.Linear(8, 8, bias=False)
        self.norm = torch.nn.LayerNorm(8)
        self.output_layer = torch.nn.Linear(8, 32, bias=False)
        self.output_layer.weight.is_embedding_or_output_parameter = True


@patch('torch.distributed.get_world_size', return_value=1)
@patch('torch.distributed.all_gather_object', lambda out, obj: out.__setitem__(0, obj))
@pytest.mark.parametrize("head_opt", ["adam", "sinkhorn"])
def test_routing(_ws, head_opt):
    from megatron.core.optimizer import _prepare_emerging_optimizer_params

    net = _Net()
    config = OptimizerConfig(optimizer="muown_classical", lr=4e-3, embedding_head_optimizer=head_opt)
    overrides = get_standard_config_overrides(config=config)
    _prepare_emerging_optimizer_params(config, "muown_classical", [net], overrides)
    groups = _get_param_groups([net], config, overrides)
    where = {id(p): g.get('optimizer', 'muown_classical') for g in groups for p in g['params']}
    emb, out = net.embedding.word_embeddings.weight, net.output_layer.weight
    assert where[id(emb)] == where[id(out)] == head_opt
    assert where[id(net.fc.weight)] == "muown_classical"
    assert where[id(net.norm.weight)] == where[id(net.norm.bias)] == "adam"
