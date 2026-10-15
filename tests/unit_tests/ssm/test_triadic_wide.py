# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Wide-key state recurrence (megatron/core/ssm/triadic_wide.py): against FLA's kernels
where both apply (K <= 256), and the triadic delta rule beyond K = 256."""

import pytest
import torch
import torch.nn.functional as F

wide = pytest.importorskip("megatron.core.ssm.triadic_wide")
if not wide.HAVE_FLA or not torch.cuda.is_available():
    pytest.skip("needs flash-linear-attention and a GPU", allow_module_level=True)

from fla.ops.common.chunk_delta_h import (  # noqa: E402
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)


def _fla_fwd_h(**kw):
    import fla.ops.common.chunk_delta_h as m

    return m.chunk_gated_delta_rule_fwd_h.__wrapped__(**kw)


def _fla_bwd_dhu(**kw):
    import fla.ops.common.chunk_delta_h as m

    return m.chunk_gated_delta_rule_bwd_dhu.__wrapped__(**kw)


def _rand(*shape, gen, dtype=torch.bfloat16, scale=1.0):
    return (torch.randn(*shape, device="cuda", generator=gen) * scale).to(dtype)


@pytest.mark.parametrize("K", [128, 256])
@pytest.mark.parametrize("use_gk", [True, False])
def test_matches_fla_kernels(K, use_gk):
    gen = torch.Generator(device="cuda").manual_seed(0)
    B, T, H, V = 2, 200, 2, 64
    k = F.normalize(_rand(B, T, H, K, gen=gen).float(), dim=-1).bfloat16()
    q = F.normalize(_rand(B, T, H, K, gen=gen).float(), dim=-1).bfloat16()
    w = _rand(B, T, H, K, gen=gen, scale=0.1)
    u = _rand(B, T, H, V, gen=gen)
    gk = None
    if use_gk:
        # chunk-local cumulative log2 decays, as KDA passes them
        step = -F.softplus(torch.randn(B, T, H, K, device="cuda", generator=gen)) * 0.05
        gk = step.view(B, -1, 64, H, K).cumsum(2).view(B, T, H, K) if T % 64 == 0 else None
        if gk is None:
            pad = (-T) % 64
            s = F.pad(step, (0, 0, 0, 0, 0, pad)).view(B, -1, 64, H, K).cumsum(2)
            gk = s.view(B, T + pad, H, K)[:, :T].contiguous()
    h0 = torch.randn(B, H, K, V, device="cuda", generator=gen) * 0.1
    ref = _fla_fwd_h(k=k, w=w, u=u, gk=gk, initial_state=h0, output_final_state=True)
    out = wide.wide_fwd_h(k, w, u, gk=gk, initial_state=h0, output_final_state=True)
    for a, b in zip(out, ref):
        torch.testing.assert_close(a.float(), b.float(), rtol=2e-2, atol=2e-2)
    do = _rand(B, T, H, V, gen=gen)
    dv = _rand(B, T, H, V, gen=gen, scale=0.1)
    dht = torch.randn(B, H, K, V, device="cuda", generator=gen) * 0.1
    ref = _fla_bwd_dhu(q=q, k=k, w=w, do=do, dv=dv, gk=gk, h0=h0, dht=dht, scale=K**-0.5)
    out = wide.wide_bwd_dhu(q, k, w, do, dv, gk=gk, h0=h0, dht=dht, scale=K**-0.5)
    for a, b in zip(out, ref):
        torch.testing.assert_close(a.float(), b.float(), rtol=2e-2, atol=2e-2)
