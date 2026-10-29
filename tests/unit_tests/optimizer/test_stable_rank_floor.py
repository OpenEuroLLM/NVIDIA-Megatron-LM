# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""The stable-rank floor of Muown's expert matrices (muown_expert_stable_rank_floor)."""

import torch

from megatron.core.optimizer.emerging_optimizers import _is_expert_param, _stable_rank_floor_


def _srank(w):
    s = torch.linalg.svdvals(w)
    return s.square().sum(-1) / s[..., 0].square()


def _collapsed(m=96, n=192, strength=3.0, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(m, n) / n**0.5
    rows = w.norm(dim=-1, keepdim=True)
    uv = torch.randn(m, 1) @ torch.randn(1, n)
    w = w + strength * uv / (uv.norm() / m**0.5)
    return w / w.norm(dim=-1, keepdim=True) * rows


def test_floor_lifts_a_collapsed_matrix_in_one_call_and_updates_g():
    healthy = torch.randn(4, 96, 192) / 192**0.5
    w = torch.cat([healthy, _collapsed()[None]], 0)
    g = -w.norm(dim=-1, keepdim=True)  # negative signs: kept
    w0 = w.clone()
    assert _srank(w)[-1] < 2 and (_srank(w)[:-1] > 25).all()
    assert _stable_rank_floor_(w, 20.0, g=g) == 1
    assert _srank(w)[-1] > 0.95 * 20.0, _srank(w)
    torch.testing.assert_close(g, -w.norm(dim=-1, keepdim=True))  # g follows the rows
    assert torch.equal(w[:-1], w0[:-1])  # healthy matrices (stable rank > floor) untouched


def test_floor_holds_under_a_persistent_rank1_push():
    """A full-size rank-1 step every iteration (as Muown's orthogonalized update of a rank-1
    gradient), then the floor: the matrix stays at the floor."""
    torch.manual_seed(1)
    w = torch.randn(1, 48, 96) * 0.02
    d = torch.randn(48, 1) @ torch.randn(1, 96)
    d = d / torch.linalg.matrix_norm(d, ord=2)
    g = w.norm(dim=-1, keepdim=True)
    for _ in range(200):
        w = w + 0.05 * d * float(torch.linalg.matrix_norm(w[0], ord=2))
        _stable_rank_floor_(w, 10.0, g=g)
    assert _srank(w)[0] > 0.95 * 10.0, _srank(w)


def test_floor_is_a_no_op_above_the_bound_and_handles_2d():
    w = torch.randn(64, 128)
    w0 = w.clone()
    assert _stable_rank_floor_(w, 5.0) == 0
    assert torch.equal(w, w0)


def test_expert_marker():
    p = torch.nn.Parameter(torch.zeros(2, 2))
    assert not _is_expert_param(p)
    p.allreduce = False
    assert _is_expert_param(p)
