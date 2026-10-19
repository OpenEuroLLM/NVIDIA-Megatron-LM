# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""MHC_FUSED_READ (megatron/core/transformer/mhc_fused.py) against the compiled path: two chained
mHC sublayers (read -> f -> write), outputs and every gradient (streams, sublayer, mixer)."""

import pytest
import torch

from megatron.core.transformer import mhc_fused
from megatron.core.transformer.mhc import MHCMixer

N = 4


def _chain(fused, s0, mixers, ws):
    s = s0
    for mixer, w in zip(mixers, ws):
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
@pytest.mark.parametrize("T,C", [(1000, 256), (4096, 2048)])
def test_fused_matches_compiled(monkeypatch, T, C):
    monkeypatch.setattr(mhc_fused, "ENABLED", True)
    torch.manual_seed(0)
    mixers = [MHCMixer(C, N).cuda() for _ in range(2)]
    for m in mixers:
        with torch.no_grad():
            m.phi.normal_(0, 0.5 / (N * C) ** 0.5)  # input-dependent coefficients of order 1
            m.alpha_pre.fill_(0.5)
            m.alpha_post.fill_(0.5)
            m.alpha_res.fill_(0.5)
    ws = [(torch.randn(C, C, device="cuda") / C**0.5).bfloat16().requires_grad_() for _ in range(2)]
    s0 = torch.randn(T, 1, N, C, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(T, 1, N, C, device="cuda", dtype=torch.bfloat16)
    assert mhc_fused.can_use(s0, N, mixers[0].phi)

    res = {}
    for fused in (False, True):
        for p in [*ws, *(q for m in mixers for q in m.parameters())]:
            p.grad = None
        s = s0.clone().requires_grad_()
        y = _chain(fused, s, mixers, ws)
        y.backward(g)
        res[fused] = [y.detach(), s.grad, *(w.grad for w in ws),
                      *(q.grad for m in mixers for q in m.parameters())]
    names = ["out", "d_streams", "d_w0", "d_w1"] + [
        f"m{i}.{n}" for i, m in enumerate(mixers) for n, _ in m.named_parameters()]
    for name, a, b in zip(names, res[True], res[False]):
        assert a is not None and b is not None, name
        assert _rel(a, b) < 2e-2, (name, _rel(a, b))
