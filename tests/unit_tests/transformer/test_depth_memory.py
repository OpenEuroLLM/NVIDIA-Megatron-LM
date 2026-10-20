# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Linear Depth Memory (megatron/core/transformer/depth_memory.py, arXiv:2610.18809)."""

import pytest
import torch

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.depth_memory import (
    DepthMemoryBoundary,
    DepthMemorySite,
    DepthMemoryState,
    rls_write,
)

G, K, H = 4, 3, 32


def _config(**kw):
    base = dict(
        num_layers=2, hidden_size=H, num_attention_heads=4, depth_memory=True,
        depth_memory_groups=G, depth_memory_key_dim=K, depth_memory_hidden_dim=8,
    )
    return TransformerConfig(**{**base, **kw})


def test_rls_writes_solve_the_weighted_ridge_regression():
    """After l writes, M is the exact minimizer of paper Eq. 6:
    sum_i (prod_{j>i} lambda_j) |v_i - M'k_i|^2 + (prod_j lambda_j) / alpha |M|^2."""
    torch.manual_seed(0)
    tokens, dv, steps, alpha = 5, 6, 7, 0.7
    memory = torch.zeros(tokens, G, K, dv, dtype=torch.float64)
    cov = (torch.eye(K, dtype=torch.float64) * alpha).expand(tokens, G, K, K)
    a = torch.eye(K, dtype=torch.float64).expand(tokens, G, K, K) / alpha  # A_{-1}
    b = torch.zeros(tokens, G, K, dv, dtype=torch.float64)  # sum k v' (memory layout)
    for _ in range(steps):
        k = torch.randn(tokens, G, K, dtype=torch.float64)
        v = torch.randn(tokens, G, dv, dtype=torch.float64)
        lam = torch.rand(G, dtype=torch.float64) * 0.3 + 0.7
        memory, cov = rls_write(memory, cov, k, v, lam)
        a = lam[:, None, None] * a + k.unsqueeze(-1) * k.unsqueeze(-2)
        b = lam[:, None, None] * b + k.unsqueeze(-1) * v.unsqueeze(-2)
    torch.testing.assert_close(memory, torch.linalg.solve(a, b))
    torch.testing.assert_close(cov, torch.linalg.inv(a))


def test_read_is_the_stream_plus_memory_readout():
    torch.manual_seed(0)
    site = DepthMemorySite(_config())
    x = torch.randn(10, 2, H)
    state = DepthMemoryState(x, G, K, 1.0)
    state.memory = torch.randn_like(state.memory)
    with torch.no_grad():
        site.read.read_gain.zero_()
    torch.testing.assert_close(site.read_input(state), x)  # gain 0: x alone
    with torch.no_grad():
        site.read.read_gain.fill_(1.0)
    query, gate = site.read.net(x)
    expected = x + torch.einsum("sbgkd,sbgk->sbgd", state.memory, query * gate[..., None]).reshape(
        x.shape
    )
    torch.testing.assert_close(site.read_input(state), expected)
    assert torch.allclose(gate, torch.ones_like(gate))  # identity-centred gate starts at 1


def test_write_keeps_the_plain_residual_and_fits_the_output():
    """x <- x + out; a single write of out under key k reads back ~out (ridge alpha=1:
    out / (lambda + 1) with unit-norm keys) through the same key."""
    torch.manual_seed(0)
    site = DepthMemorySite(_config())
    x = torch.randn(6, 1, H)
    state = DepthMemoryState(x, G, K, 1.0)
    out = torch.randn(6, 1, H)
    new = site.write_output(state, out)
    torch.testing.assert_close(new, x + out)
    assert state.stream is new
    key, weight = site.write.net(new)
    k = key * weight.sqrt()[..., None]
    lam = torch.sigmoid(site.write.forget_logit)
    recalled = torch.einsum("sbgkd,sbgk->sbgd", state.memory, k)
    target = out.reshape(6, 1, G, H // G) * weight.sqrt()[..., None]
    factor = (k * k).sum(-1, keepdim=True) / (lam[:, None] + (k * k).sum(-1, keepdim=True))
    torch.testing.assert_close(recalled, target * factor)


def test_boundary_and_gradients():
    torch.manual_seed(0)
    cfg = _config()
    boundary, sites = DepthMemoryBoundary(cfg), [DepthMemorySite(cfg) for _ in range(3)]
    emb = torch.randn(8, 2, H, requires_grad=True)
    state = boundary.enter(emb)
    lin = torch.nn.Linear(H, H)
    for site in sites:
        h = site.read_input(state)
        site.write_output(state, torch.tanh(lin(h)))
    out = boundary.exit(state)
    out.square().sum().backward()
    params = [p for m in (boundary, *sites) for p in m.parameters()]
    assert all(p.is_residual_mixer for p in params)
    # every read and write network gets a gradient (the last write through the final read)
    missing = [n for m in (boundary, *sites) for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing
    assert emb.grad is not None and torch.isfinite(emb.grad).all()


def test_packed_documents_are_independent_per_token():
    """Per-token state: tokens never interact, so document masking needs nothing."""
    torch.manual_seed(0)
    cfg = _config()
    boundary, site = DepthMemoryBoundary(cfg), DepthMemorySite(cfg)

    def run(x):
        state = boundary.enter(x)
        site.write_output(state, torch.sin(site.read_input(state)))
        return boundary.exit(state)

    x = torch.randn(12, 1, H)
    torch.testing.assert_close(run(x)[:5], run(x[:5]))


def test_config_validation():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _config(mhc=True)
    with pytest.raises(ValueError, match="must divide hidden_size"):
        _config(depth_memory_groups=5)
    with pytest.raises(ValueError, match="CUDA-graph"):
        _config(cuda_graph_impl="local")


# ---- stream form (depth_memory_impl torch): static workspace + unwinding backward ----------


def _stream_chain(cfg, sites: int):
    from megatron.core.transformer.depth_memory import DepthMemoryMixer, DepthMemoryStreamBoundary
    from megatron.core.transformer.depth_memory_state import make_workspace

    boundary = DepthMemoryStreamBoundary(cfg)
    mixers = [DepthMemoryMixer(cfg) for _ in range(sites)]
    ws = make_workspace(cfg, total_edges=sites + 1)
    boundary.wire(ws)
    for e, m in enumerate(mixers, start=1):
        m.wire(e, ws)
    return boundary, mixers


@pytest.mark.parametrize("sites", [1, 4])
def test_stream_form_matches_the_reference(sites):
    """The torch workspace (state in static buffers, M unwound in the backward) gives the
    reference's outputs and every parameter / input gradient."""
    torch.manual_seed(0)
    cfg = _config(depth_memory_impl="torch")
    boundary, mixers = _stream_chain(cfg, sites)
    ref_boundary, ref_sites = DepthMemoryBoundary(cfg), [DepthMemorySite(cfg) for _ in range(sites)]
    with torch.no_grad():  # random (not init) parameters, so gates / gains / forgetting matter
        for p in [*boundary.parameters(), *(p for m in mixers for p in m.parameters())]:
            p.add_(torch.randn_like(p) * 0.3)
    # site i's read is owned by the writer before it (the boundary for the first)
    ref_boundary.embed_write.load_state_dict(boundary.embed_write.state_dict())
    ref_boundary.final_read.load_state_dict(mixers[-1].next_query.state_dict())
    queries = [boundary.first_query] + [m.next_query for m in mixers[:-1]]
    for site, mixer, query in zip(ref_sites, mixers, queries):
        site.read.load_state_dict(query.state_dict())
        site.write.load_state_dict(mixer.write_site.state_dict())
    lins = [torch.nn.Linear(H, H) for _ in range(sites)]
    emb = torch.randn(7, 3, H)

    e1 = emb.clone().requires_grad_(True)
    streams = boundary.expand(e1)
    for mixer, lin in zip(mixers, lins):
        streams = mixer.write(streams, torch.tanh(lin(mixer.read(streams))))
    out = boundary.collapse(streams)

    e2 = emb.clone().requires_grad_(True)
    state = ref_boundary.enter(e2)
    for site, lin in zip(ref_sites, lins):
        site.write_output(state, torch.tanh(lin(site.read_input(state))))
    ref = ref_boundary.exit(state)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    w = torch.randn_like(ref)
    grads = torch.autograd.grad((out * w).sum(), [e1, *boundary.parameters(),
                                                  *(p for m in mixers for p in m.parameters())])
    # the stream-form parameter order: boundary (embed_write, first_query), then per mixer
    # (write_site, next_query)
    ref_params = [*ref_boundary.embed_write.parameters(), *ref_sites[0].read.parameters()]
    nexts = [s.read for s in ref_sites[1:]] + [ref_boundary.final_read]
    for site, nxt in zip(ref_sites, nexts):
        ref_params += [*site.write.parameters(), *nxt.parameters()]
    ref_grads = torch.autograd.grad((ref * w).sum(), [e2, *ref_params])
    names = ["emb"] + [n for n, _ in boundary.named_parameters()] + [
        f"{i}.{n}" for i, m in enumerate(mixers) for n, _ in m.named_parameters()]
    for name, g, rg in zip(names, grads, ref_grads):
        torch.testing.assert_close(g, rg, rtol=1e-4, atol=1e-5, msg=name)


def test_stream_form_runs_again_with_the_same_buffers():
    """Two forward / backward passes (micro-batches) on one workspace: the same buffers, and
    the second pass is as correct as the first (edge 0 resets the state)."""
    torch.manual_seed(0)
    cfg = _config(depth_memory_impl="torch")
    boundary, mixers = _stream_chain(cfg, 2)

    def loss(x):
        s = boundary.expand(x)
        for m in mixers:
            s = m.write(s, torch.sin(m.read(s)))
        return boundary.collapse(s).square().sum()

    x = torch.randn(5, 2, H)
    g1 = torch.autograd.grad(loss(x), list(boundary.parameters()))
    ws = mixers[0]._workspace[0]
    ptr = ws.m.data_ptr()
    loss(torch.randn(5, 2, H)).backward()  # another micro-batch in between
    g2 = torch.autograd.grad(loss(x), list(boundary.parameters()))
    assert ws.m.data_ptr() == ptr
    for a, b in zip(g1, g2):
        torch.testing.assert_close(a, b)
