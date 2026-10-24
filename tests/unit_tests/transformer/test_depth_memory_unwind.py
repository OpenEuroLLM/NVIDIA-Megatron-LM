# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""depth_memory_unwind: the tape-free workspace (closed-form recovery of each edge's update at
its recompute) gives the taped oracle's reads and gradients."""

import pytest
import torch

from megatron.core.transformer.depth_memory_state import (
    TorchDepthMemoryWorkspace,
    UnwindingDepthMemoryWorkspace,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _inputs(edges, n, g, k, d, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(DEV)

    out = []
    for _ in range(edges):
        khat = torch.nn.functional.normalize(r(n, g, k), dim=-1)
        query = torch.nn.functional.normalize(r(n, g, k), dim=-1)
        value = r(n, g, d, scale=0.5)
        scale = (0.8 + 0.4 * torch.rand(n, g, generator=gen)).to(DEV).sqrt()
        lam = (0.95 + 0.04 * torch.rand(g, generator=gen)).to(DEV)
        out.append((khat, query, value, scale, lam))
    grads = [r(n, g, d, scale=0.1) for _ in range(edges)]
    return out, grads


@pytest.mark.parametrize("edges", [1, 2, 7, 33, 97])
@pytest.mark.parametrize("alpha", [1.0, 4.0])
def test_unwind_matches_taped_oracle(edges, alpha):
    torch.backends.cuda.matmul.allow_tf32 = False
    n, g, k, d = 24, 4, 8, 16
    inputs, read_grads = _inputs(edges, n, g, k, d, seed=edges)
    ref = TorchDepthMemoryWorkspace(edges, g, k, d, alpha)
    unw = UnwindingDepthMemoryWorkspace(edges, g, k, d, alpha)
    ref.ensure(n, torch.device(DEV))
    unw.ensure(n, torch.device(DEV))
    for _micro_batch in range(2):  # the phases reset per micro-batch
        ref_reads = [ref.forward(e, *inputs[e]) for e in range(edges)]
        unw_reads = [unw.forward(e, *inputs[e]) for e in range(edges)]
        for a, b in zip(ref_reads, unw_reads):
            torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6)
        for e in reversed(range(edges)):
            ref_g = ref.backward(e, read_grads[e])
            if e > 0:  # the recompute of edge e's layer: the same read, from the unwound state
                torch.testing.assert_close(unw.forward(e, *inputs[e]), ref_reads[e],
                                           rtol=1e-3, atol=1e-4)
            unw_g = unw.backward(e, read_grads[e])
            for a, b in zip(ref_g, unw_g):
                torch.testing.assert_close(b, a, rtol=1e-3, atol=1e-4)
        # fully unwound: M back to zero, P to alpha I
        assert unw.m.abs().max() < 1e-3  # fp32 round-off: ~2e-4 after 97 unwinds
        torch.testing.assert_close(unw.p, unw.p0, rtol=1e-3, atol=1e-3)


def test_out_of_order_recompute_raises():
    n, g, k, d = 4, 2, 8, 8
    inputs, _ = _inputs(3, n, g, k, d, seed=0)
    unw = UnwindingDepthMemoryWorkspace(3, g, k, d, 1.0)
    unw.ensure(n, torch.device(DEV))
    for e in range(3):
        unw.forward(e, *inputs[e])
    with pytest.raises(RuntimeError):
        unw.forward(1, *inputs[1])  # the state is at edge 2
