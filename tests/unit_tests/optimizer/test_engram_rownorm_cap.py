# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""--engram-clip-rownorm: TableAdam rescales Engram table rows above the (ramped) cap after the update."""

import torch

from megatron.core.optimizer.engram_optimizer import TableAdam


def _table(rows=64, dim=8, big=(3, 17, 40)):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(rows, dim) * 0.1, requires_grad=False)
    with torch.no_grad():
        for r in big:
            p[r] *= 100.0  # heavy rows, norm ~ 30
    p.grad = torch.zeros_like(p)  # zero gradient: Adam moves nothing, only the cap acts
    return p


def test_cap_schedule():
    opt = TableAdam([torch.nn.Parameter(torch.zeros(2, 2))], lr=0.0, betas=(0.9, 0.95), eps=1e-8,
                    rownorm_cap=2.0, rownorm_cap_start=8.0, rownorm_cap_ramp_start=10,
                    rownorm_cap_ramp_steps=20)
    assert opt.cap_at(9) is None  # no cap before the ramp when a start value is given
    assert abs(opt.cap_at(10) - 8.0) < 1e-9
    assert abs(opt.cap_at(20) - 4.0) < 1e-9  # geometric midpoint of 8 -> 2
    assert abs(opt.cap_at(30) - 2.0) < 1e-9 and abs(opt.cap_at(1000) - 2.0) < 1e-9
    const = TableAdam([torch.nn.Parameter(torch.zeros(2, 2))], lr=0.0, betas=(0.9, 0.95), eps=1e-8,
                      rownorm_cap=3.0)
    assert const.cap_at(1) == 3.0
    off = TableAdam([torch.nn.Parameter(torch.zeros(2, 2))], lr=0.0, betas=(0.9, 0.95), eps=1e-8)
    assert off.cap_at(1) is None


def test_heavy_rows_rescaled_others_untouched():
    p = _table()
    before = p.detach().clone()
    opt = TableAdam([p], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, rowwise_v=True,
                    momentum_dtype=torch.bfloat16, rownorm_cap=2.0)
    opt.step()
    n = p.detach().norm(dim=-1)
    assert torch.all(n <= 2.0 + 1e-5)
    heavy = torch.zeros(p.shape[0], dtype=torch.bool)
    heavy[[3, 17, 40]] = True
    assert torch.allclose(n[heavy], torch.full((3,), 2.0), atol=1e-5)
    # rescaled, not reshaped: same direction as before
    cos = torch.nn.functional.cosine_similarity(p.detach()[heavy], before[heavy], dim=-1)
    assert torch.allclose(cos, torch.ones(3), atol=1e-6)
    assert torch.equal(p.detach()[~heavy], before[~heavy])
    assert int(opt.capped_rows.item()) == 3
    assert opt.max_row_norm.item() > 20.0


def test_off_and_non_engram_tables_untouched():
    p = _table()
    before = p.detach().clone()
    TableAdam([p], lr=1e-3, betas=(0.9, 0.95), eps=1e-8).step()
    assert torch.equal(p.detach(), before)  # zero grad, no cap: unchanged
    q = _table()
    q.sharded_table_kind = "pkm"
    before_q = q.detach().clone()
    TableAdam([q], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, rownorm_cap=2.0).step()
    assert torch.equal(q.detach(), before_q)  # the cap is for Engram tables only
