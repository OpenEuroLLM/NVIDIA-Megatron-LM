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


# --engram-table-optimizer sinkhorn: TableAdam's Sinkhorn mode == sinkhorn.SinkhornMomentum (unsharded)


def _pair(rows=96, dim=16, seed=1):
    torch.manual_seed(seed)
    w0 = torch.randn(rows, dim) * 0.1
    a = torch.nn.Parameter(w0.clone(), requires_grad=False)
    b = torch.nn.Parameter(w0.clone(), requires_grad=False)
    return a, b


def test_sinkhorn_matches_reference(monkeypatch):
    from megatron.core.optimizer import engram_optimizer
    from megatron.core.optimizer.sinkhorn import SinkhornMomentum

    monkeypatch.setattr(engram_optimizer, "_CHUNK_ELEMENTS", 16 * 7)  # 7 rows per chunk: many chunks
    a, b = _pair()
    ours = TableAdam([a], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, update="sinkhorn",
                     sinkhorn_momentum=0.95, sinkhorn_iters=11, sinkhorn_lr_ratio=0.18)
    ref = SinkhornMomentum([b], lr=1e-2, momentum=0.95, nesterov=True, iters=11, tau=1e-3,
                           eps=1e-20, lr_ratio=0.18)
    g = torch.Generator().manual_seed(2)
    for _ in range(4):
        grad = torch.randn(a.shape, generator=g)
        grad[::5] = 0.0  # rows absent from the batch (never touched: masked)
        a.grad, b.grad = grad.clone(), grad.clone()
        ours.step()
        ref.step()
        assert torch.allclose(a.detach(), b.detach(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(ours.state[a]['exp_avg'].float(), ref.state[b]['momentum_buffer'],
                               atol=1e-6, rtol=1e-5)


def test_sinkhorn_step_size_untouched_rows_and_cap():
    grad = torch.randn(96, 16)
    grad[:10] = 0.0
    # no cap: touched rows move by lr x lr_ratio per element (unit row RMS), untouched rows not at all
    a, _ = _pair()
    w0 = a.detach().clone()
    opt = TableAdam([a], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, update="sinkhorn")
    a.grad = grad.clone()
    opt.step()
    assert torch.equal(a.detach()[:10], w0[:10])
    step_rms = (a.detach()[10:] - w0[10:]).pow(2).mean().sqrt().item()
    assert abs(step_rms - 1e-2 * 0.18) < 1e-2 * 0.18 * 0.05
    # with the cap (rows start at norm ~0.4): every row ends at or below it
    b, _ = _pair()
    opt = TableAdam([b], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, update="sinkhorn", rownorm_cap=0.3)
    b.grad = grad.clone()
    opt.step()
    assert torch.all(b.detach().norm(dim=-1) <= 0.3 + 1e-5)


def test_bad_update_rejected():
    import pytest

    with pytest.raises(ValueError):
        TableAdam([torch.nn.Parameter(torch.zeros(2, 2))], lr=0.0, betas=(0.9, 0.95), eps=1e-8,
                  update="lion")
