# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""MHC_FUSED_READ (megatron/core/transformer/mhc_fused.py) against the compiled path and an fp32
ground truth: three chained mHC sublayers (read -> f -> write; the second and third reads also run
the previous write's backward), optionally with a second consumer of the intermediate stream
states; and a single read with every gradient."""

import pytest
import torch

from megatron.core.transformer import mhc_fused
from megatron.core.transformer.mhc import MHCMixer, _sinkhorn_knopp_impl

N = 4


def _fp32_chain(s0, mixers, ws, taps=None):
    """Ground truth: the same chain in fp32, eager."""
    s = s0
    for k, (mixer, w) in enumerate(zip(mixers, ws)):
        if taps is not None and k > 0:
            taps.append(s)
        n = s.shape[-2]
        flat = s.flatten(-2)
        r = (flat.norm(dim=-1, keepdim=True) / flat.shape[-1] ** 0.5).clamp(min=mixer.eps)
        pre, post, res = torch.split(flat @ mixer.phi / r, [n, n, n * n], dim=-1)
        h_pre = torch.sigmoid(mixer.alpha_pre * pre + mixer.b_pre)
        h_post = 2.0 * torch.sigmoid(mixer.alpha_post * post + mixer.b_post)
        h_res = _sinkhorn_knopp_impl(
            mixer.alpha_res * res.view(*res.shape[:-1], n, n) + mixer.b_res,
            mixer.num_sinkhorn_iters,
        )
        x = (h_pre.unsqueeze(-1) * s).sum(-2)
        f = torch.tanh(x @ w)
        s = (h_res.unsqueeze(-1) * s.unsqueeze(-3)).sum(-2) + h_post.unsqueeze(-1) * f.unsqueeze(-2)
    return s


def _chain(fused, s0, mixers, ws, taps=None):
    s = s0
    for k, (mixer, w) in enumerate(zip(mixers, ws)):
        if fused is None:
            return _fp32_chain(s0, mixers, ws, taps)
        if taps is not None and k > 0:
            taps.append(s)  # a second consumer of the intermediate stream state
        if fused:
            link = mhc_fused.ReadWriteLink()
            x, _, hp, hr = mhc_fused.coeffs_and_read(mixer, s, link)
        else:
            x, _, hp, hr = mixer.coeffs_and_read(s)
            link = None
        f = torch.tanh(x @ w)
        s = mixer.write(s, f, hp, hr, link=link)
    return s


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


@pytest.mark.skipif(not (torch.cuda.is_available() and mhc_fused.HAVE_TRITON), reason="needs Triton")
@pytest.mark.parametrize("T,C,tap", [(1000, 256, False), (4096, 2048, False), (1000, 256, True)])
def test_fused_matches_compiled(monkeypatch, T, C, tap):
    monkeypatch.setattr(mhc_fused, "ENABLED", True)
    torch.manual_seed(0)
    mixers = [MHCMixer(C, N).cuda() for _ in range(3)]
    for m in mixers:
        with torch.no_grad():
            m.phi.normal_(0, 0.5 / (N * C) ** 0.5)  # input-dependent coefficients of order 1
            m.alpha_pre.fill_(0.5)
            m.alpha_post.fill_(0.5)
            m.alpha_res.fill_(0.5)
    ws = [(torch.randn(C, C, device="cuda") / C**0.5).bfloat16().requires_grad_() for _ in range(3)]
    s0 = torch.randn(T, 1, N, C, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(T, 1, N, C, device="cuda", dtype=torch.bfloat16)
    assert mhc_fused.can_use(s0, N, mixers[0].phi)

    res = {}
    for fused in (None, False, True):  # fp32 ground truth, compiled bf16, fused bf16
        for p in [*ws, *(q for m in mixers for q in m.parameters())]:
            p.grad = None
        dt = torch.float32 if fused is None else torch.bfloat16
        wl = [w.detach().to(dt).requires_grad_() for w in ws]
        s = s0.to(dt).requires_grad_()
        taps = [] if tap else None
        y = _chain(fused, s, mixers, wl, taps)
        loss = (y * g.to(dt)).sum()
        for x in taps or []:
            loss = loss + (x.float() * g.float()).sum() * 0.5
        loss.backward()
        ws_grads = [w.grad for w in wl]
        res[fused] = [y.detach(), s.grad, *ws_grads,
                      *(q.grad.clone() for m in mixers for q in m.parameters())]
    names = ["out", "d_streams", "d_w0", "d_w1", "d_w2"] + [
        f"m{i}.{n}" for i, m in enumerate(mixers) for n, _ in m.named_parameters()]
    for name, fu, co, gt in zip(names, res[True], res[False], res[None]):
        assert fu is not None and co is not None, name
        # no worse than the compiled bf16 path's own error against fp32
        err_fused, err_compiled = _rel(fu, gt), _rel(co, gt)
        # scalar gates: a sum over all tokens of the upstream bf16 noise (both paths are a few %
        # off there; the single-read test below checks them tightly)
        tol = max(5e-2 if gt.numel() == 1 else 2e-2, 1.5 * err_compiled)
        assert err_fused < tol, (name, err_fused, err_compiled)


@pytest.mark.skipif(not (torch.cuda.is_available() and mhc_fused.HAVE_TRITON), reason="needs Triton")
def test_fused_read_all_grads(monkeypatch):
    """One read (coefficients, read, write coefficients out): every gradient, against fp32."""
    monkeypatch.setattr(mhc_fused, "ENABLED", True)
    torch.manual_seed(0)
    T, C = 4096, 2048
    m = MHCMixer(C, N).cuda()
    with torch.no_grad():
        m.phi.normal_(0, 0.5 / (N * C) ** 0.5)
        for a in (m.alpha_pre, m.alpha_post, m.alpha_res):
            a.fill_(0.5)
    s0 = torch.randn(T, 1, N, C, device="cuda", dtype=torch.bfloat16)
    gx = torch.randn(T, 1, C, device="cuda", dtype=torch.bfloat16).float()
    gp = torch.randn(T, 1, N, device="cuda")
    gr = torch.randn(T, 1, N, N, device="cuda")
    res = {}
    for mode in ("gt", "compiled", "fused"):
        for p in m.parameters():
            p.grad = None
        if mode == "gt":
            s = s0.float().requires_grad_()
            n = N
            flat = s.flatten(-2)
            r = (flat.norm(dim=-1, keepdim=True) / flat.shape[-1] ** 0.5).clamp(min=m.eps)
            pre, post, rs = torch.split(flat @ m.phi / r, [n, n, n * n], dim=-1)
            h_pre = torch.sigmoid(m.alpha_pre * pre + m.b_pre)
            hp = 2.0 * torch.sigmoid(m.alpha_post * post + m.b_post)
            hr = _sinkhorn_knopp_impl(m.alpha_res * rs.view(*rs.shape[:-1], n, n) + m.b_res,
                                      m.num_sinkhorn_iters)
            x = (h_pre.unsqueeze(-1) * s).sum(-2)
        elif mode == "compiled":
            s = s0.clone().requires_grad_()
            x, _, hp, hr = m.coeffs_and_read(s)
        else:
            s = s0.clone().requires_grad_()
            x, _, hp, hr = mhc_fused.coeffs_and_read(m, s, mhc_fused.ReadWriteLink())
        ((x.float() * gx).sum() + (hp * gp).sum() + (hr * gr).sum()).backward()
        res[mode] = [x.detach(), hp.detach(), hr.detach(), s.grad,
                     *(p.grad.clone() for p in m.parameters())]
    names = ["x", "h_post", "h_res", "d_streams"] + [n for n, _ in m.named_parameters()]
    for name, fu, co, gt in zip(names, res["fused"], res["compiled"], res["gt"]):
        err_fused, err_compiled = _rel(fu, gt), _rel(co, gt)
        assert err_fused < max(1e-2, 1.5 * err_compiled), (name, err_fused, err_compiled)
