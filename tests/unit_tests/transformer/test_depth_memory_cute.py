# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""The paper's CuTe kernels (cute_kernels cute_ldm) against the PyTorch workspace oracle
(depth_memory_state.py): reads and every input gradient over a whole forward / backward, and the
model under CUDA graphs. Hopper / Blackwell only (cluster tests)."""

import pytest
import torch

from megatron.core.transformer.depth_memory_state import TorchDepthMemoryWorkspace

cute_ldm = pytest.importorskip("cute_ldm.workspace")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10),
    reason="the CuTe LDM kernels need sm90 / sm100",
)


def _inputs(edges, n, g, k, d, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, device="cuda", generator=gen)  # noqa: E731
    out = []
    for _ in range(edges):
        khat = torch.nn.functional.normalize(rnd(n, g, k), dim=-1)
        query = torch.nn.functional.normalize(rnd(n, g, k), dim=-1) * (1 + 0.3 * rnd(n, g, 1))
        value = rnd(n, g, d).to(torch.bfloat16)
        scale = (torch.rand(n, g, device="cuda", generator=gen) * 0.8 + 0.6)
        forget = torch.rand(g, device="cuda", generator=gen) * 0.1 + 0.88
        out.append((khat, query, value, scale, forget))
    return out


@pytest.mark.parametrize("edges,d", [(1, 64), (5, 64), (11, 128), (9, 48)])
@pytest.mark.parametrize("cov", ["float32"])  # the backward unwinds P from the running FP32 state
def test_kernels_match_the_torch_oracle(edges, d, cov):
    n, g, k = 512, 16, 8
    torch_ws = TorchDepthMemoryWorkspace(edges, g, k, d, alpha=1.0)
    cute_ws = cute_ldm.CuteDepthMemoryWorkspace(total_edges=edges, groups=g, key_dim=k,
                                                value_dim=d, span=4, running_covariance_dtype=cov)
    torch_ws.ensure(n, "cuda")
    cute_ws.ensure(n, "cuda")
    steps = _inputs(edges, n, g, k, d, seed=edges * 7 + d)
    grads = [torch.randn(n, g, d, device="cuda").to(torch.bfloat16) for _ in range(edges)]
    for _ in range(2):  # twice: the static buffers carry nothing over between passes
        reads_t = [torch_ws.forward(e, *steps[e]) for e in range(edges)]
        reads_c = [cute_ws.forward(e, *steps[e]) for e in range(edges)]
        back_t = [torch_ws.backward(e, grads[e]) for e in reversed(range(edges))]
        back_c = [cute_ws.backward(e, grads[e]) for e in reversed(range(edges))]
        for e in range(edges):
            ref = reads_t[e]
            err = (reads_c[e].float() - ref).abs().max() / ref.abs().max()
            assert err < 3e-2, (e, float(err))
        for e, (bt, bc) in enumerate(zip(back_t, back_c)):
            for name, a, b in zip(("khat", "query", "value", "scale", "forget"), bt, bc):
                scale = a.float().abs().max().clamp_min(1e-6)
                err = (b.float() - a.float()).abs().max() / scale
                assert err < 5e-2, (edges - 1 - e, name, float(err))


@pytest.mark.parametrize("d", [64, 128])
def test_recompute_reads_match_the_torch_oracle(d):
    """The read kernel: a window's reads from the running state at the window's end (before its
    window-end launch) and at its start (after it), against the oracle's reads of the forward."""
    edges, n, g, k = 9, 512, 16, 8
    torch_ws = TorchDepthMemoryWorkspace(edges, g, k, d, alpha=1.0)
    cute_ws = cute_ldm.CuteDepthMemoryWorkspace(total_edges=edges, groups=g, key_dim=k,
                                                value_dim=d, span=4, running_covariance_dtype="float32")
    torch_ws.ensure(n, "cuda")
    cute_ws.ensure(n, "cuda")
    steps = _inputs(edges, n, g, k, d, seed=11 * d)
    grads = [torch.randn(n, g, d, device="cuda").to(torch.bfloat16) for _ in range(edges)]
    reads_t = [torch_ws.forward(e, *steps[e]) for e in range(edges)]
    for e in range(edges):
        cute_ws.forward(e, *steps[e])

    def check(got, first, tag):
        for j, read in enumerate(got):
            ref = reads_t[first + j]
            err = (read.float() - ref).abs().max() / ref.abs().max()
            assert err < 3e-2, (tag, first + j, float(err))

    # the state is after edge 8 = after the last edge of the window [8, 9): its own read, from the end
    check(cute_ws.recompute_reads(8, 9, 0, 1, from_end=True), 8, "end, last window")
    torch_ws.backward(8, grads[8])
    cute_ws.backward(8, grads[8])
    # now before edge 8 = after edge 7, the end of the window [4, 8): its reads from the end
    check(cute_ws.recompute_reads(4, 8, 0, 4, from_end=True), 4, "end")
    check(cute_ws.recompute_reads(4, 8, 1, 3, from_end=True), 5, "end, slice")
    for e in (7, 6, 5, 4):
        torch_ws.backward(e, grads[e])
        cute_ws.backward(e, grads[e])
    # now before edge 4 = the start of the window [4, 8): the same reads from the start
    check(cute_ws.recompute_reads(4, 8, 0, 4, from_end=False), 4, "start")
    check(cute_ws.recompute_reads(4, 8, 2, 4, from_end=False), 6, "start, slice")
