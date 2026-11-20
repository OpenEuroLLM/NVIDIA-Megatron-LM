# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""rowmask_attention: block-sparse attention with a per-row bit mask per (query tile, key block) equals dense attention
under the same mask (per query: the allowed key blocks, causal by positions), forward and q / k / v gradients."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton kernels)")


def _reference(q, k, v, allowed, q_pos, k_pos, scale, rep, per_group, bn):
    """Dense masked attention in float64: q [Hq, Sq, D], allowed [Gs, Sq, P]."""
    Hq, Sq, D = q.shape
    Sk = k.shape[1]
    blk = torch.arange(Sk, device=q.device) // bn
    outs = []
    for h in range(Hq):
        hk = h // rep
        a = allowed[hk if per_group else 0][:, blk]  # [Sq, Sk]
        m = a & (k_pos[None, :] <= q_pos[:, None]) & (q_pos[:, None] >= 0)
        s = (q[h].double() @ k[hk].double().T) * scale
        s = s.masked_fill(~m, float("-inf"))
        p = torch.softmax(s, -1).nan_to_num(0.0)
        outs.append(p @ v[hk].double())
    return torch.stack(outs)


@pytest.mark.parametrize("per_group", [True, False])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("dens", [0.3, 1.0])
def test_rowmask_matches_dense_masked(per_group, D, dens):
    from megatron.core.transformer.experimental_attention_variant import rowmask_attention as ra

    torch.manual_seed(0)
    Hq, Hk, bn = 4, 2, 64
    lengths = (200, 136, 70)  # spans packed at tile / block boundaries, padding rows / keys between them
    starts, tot = [], 0
    for L in lengths:
        starts.append(tot)
        tot += -(-L // 64) * 64
    Sq = Sk = tot
    q_pos = torch.full((Sq,), -1, dtype=torch.int32)
    k_pos = torch.full((Sk,), 2**30, dtype=torch.int32)
    P = Sk // bn
    Gs = Hk if per_group else 1
    allowed = torch.zeros(Gs, Sq, P, dtype=torch.bool)
    for a0, L in zip(starts, lengths):
        q_pos[a0:a0 + L] = torch.arange(L, dtype=torch.int32)
        k_pos[a0:a0 + L] = torch.arange(L, dtype=torch.int32)
        p0, p1 = a0 // bn, -(-(a0 + L) // bn)
        allowed[:, a0:a0 + L, p0:p1] = torch.rand(Gs, L, p1 - p0) < dens
        cur = p0 + torch.arange(L) // bn
        allowed[:, torch.arange(a0, a0 + L), cur] = True  # every query its own page (as the window)
    q = torch.randn(Hq, Sq, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(Hk, Sk, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(Hk, Sk, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    allowed, q_pos, k_pos = allowed.cuda(), q_pos.cuda(), k_pos.cuda()
    idx, bits, cnt = ra.lists_from_allowed(allowed)
    scale = D**-0.5
    out = ra.rowmask_block_attention(q, k, v, idx, bits, cnt, q_pos, k_pos, scale, per_group, bn)
    qd, kd, vd = (x.detach().double().requires_grad_() for x in (q, k, v))
    ref = _reference(qd, kd, vd, allowed, q_pos, k_pos, scale, Hq // Hk, per_group, bn)
    real = (q_pos >= 0)[None, :, None]
    assert float(((out.double() - ref).abs() * real).max() / ref.abs().max()) < 2e-2
    go = torch.randn_like(out) * real
    g = torch.autograd.grad((out.float() * go.float()).sum(), (q, k, v))
    gr = torch.autograd.grad((ref * go.double()).sum(), (qd, kd, vd))
    for x, y in zip(g, gr):
        assert float((x.double() - y).abs().max() / y.abs().max()) < 3e-2


@pytest.mark.parametrize("mode", ["block", "token_candidates"])
@pytest.mark.parametrize("selector", ["page_mean", "page_maxpool"])
def test_packed_path_rowmask_equals_flex(mode, selector):
    """flex_packed_page_attention with kernel="rowmask" equals the flex kernel on packed spans (documents and a chunk
    with an offset), block-wise selection and per-token choices inside block candidates, outputs and gradients."""
    from megatron.core.transformer.experimental_attention_variant import page_sparse as ps

    torch.manual_seed(1)
    hq, g, d = 4, 2, 64
    kw = dict(selector=selector, page=64, topk=3, window=128, sink_pages=1, per_group=True)
    if mode == "block":
        cfg = ps.PageSparseConfig(**kw, select_block=64)
    else:
        cfg = ps.PageSparseConfig(**kw, select_block=1, block_candidates=5, candidate_block=64)
    spans = []
    for L, off in [(700, 0), (300, 0), (512, 256)]:
        q = torch.randn(1, hq, L - off, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, g, L, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, g, L, d, device="cuda", dtype=torch.bfloat16)
        spans.append((q, k, v, off, None))

    def run(kernel):
        leaves = [[t.clone().requires_grad_() for t in sp[:3]] for sp in spans]
        outs = ps.flex_packed_page_attention([(*lv, sp[3], None) for lv, sp in zip(leaves, spans)], cfg, d**-0.5,
                                             64, kernel=kernel)
        gen = torch.Generator(device="cuda").manual_seed(5)
        loss = sum((o.float() * torch.randn(o.shape, device="cuda", generator=gen)).sum() for o in outs)
        loss.backward()
        return [o.detach().float() for o in outs], [t.grad.float() for lv in leaves for t in lv]

    oa, ga = run("flex")
    ob, gb = run("rowmask")
    for x, y in zip(oa + ga, ob + gb):
        assert float((x - y).abs().max() / y.abs().max().clamp(min=1e-6)) < 2e-2


def test_lists_from_allowed_chunking_and_bits():
    """The chunked byte packing equals the bit definition (bit i of a tile's word = row i allows the block), for any
    chunk size, rows past Sq absent."""
    from megatron.core.transformer.experimental_attention_variant import rowmask_attention as ra

    torch.manual_seed(3)
    allowed = torch.rand(2, 300, 9, device="cuda") < 0.3
    ref = None
    for chunk in (1, 2, 128):
        idx, bits, cnt = ra.lists_from_allowed(allowed, chunk=chunk)
        nq = idx.shape[1]
        dense = torch.zeros(2, nq * 64, 9, dtype=torch.bool, device="cuda")
        for g in range(2):
            for t in range(nq):
                for j in range(int(cnt[g, t])):
                    w = int(bits[g, t, j])
                    for i in range(64):
                        if (w >> i) & 1:
                            dense[g, t * 64 + i, int(idx[g, t, j])] = True
        torch.testing.assert_close(dense[:, :300], allowed)
        assert not dense[:, 300:].any()
        got = (idx[..., :1].clone(), cnt.clone())
        ref = ref or got
        torch.testing.assert_close(got[1], ref[1])
