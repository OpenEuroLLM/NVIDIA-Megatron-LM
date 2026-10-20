# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""LDM state across segments (recompute chunks / pipeline stages): a chain of edges split into two
workspaces -- the second loads the first's state (load_state), the first's backward starts from
the second's exported state gradient (export_grad, with the CuTe window fold) -- gives the reads
and the input gradients of the whole chain in one workspace."""

import pytest
import torch

from megatron.core.transformer.depth_memory_state import TorchDepthMemoryWorkspace

N, G, K, D, EDGES, SPLIT = 512, 4, 8, 64, 8, 4


def _inputs(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    per_edge = []
    for _ in range(EDGES):
        key = torch.nn.functional.normalize(torch.randn(N, G, K, device="cuda", generator=g), dim=-1)
        query = torch.nn.functional.normalize(torch.randn(N, G, K, device="cuda", generator=g), dim=-1)
        value = (0.5 * torch.randn(N, G, D, device="cuda", generator=g)).bfloat16()
        scale = 0.8 + 0.4 * torch.rand(N, G, device="cuda", generator=g)
        forgetting = 0.98 + 0.015 * torch.rand(G, device="cuda", generator=g)
        per_edge.append((key, query, value, scale, forgetting))
    cot = [(0.1 * torch.randn(N, G, D, device="cuda", generator=g)).bfloat16() for _ in range(EDGES)]
    return per_edge, cot


def _make(impl, edges):
    if impl == "torch":
        ws = TorchDepthMemoryWorkspace(edges, G, K, D, 1.0)
    else:
        cute = pytest.importorskip("cute_ldm.workspace")
        ws = cute.CuteDepthMemoryWorkspace(total_edges=edges, groups=G, key_dim=K, value_dim=D,
                                           span=4, running_covariance_dtype="float32")
    ws.ensure(N, "cuda")
    return ws


def _run(ws, per_edge, cot):
    reads = [ws.forward(e, *per_edge[e]).float().clone() for e in range(len(per_edge))]
    grads = [None] * len(per_edge)
    for e in reversed(range(len(per_edge))):
        grads[e] = [t.float().clone() for t in ws.backward(e, cot[e])]
    return reads, grads


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("impl", ["torch", "cute"])
def test_two_segments_equal_one_chain(impl):
    if impl == "cute" and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("cute_ldm needs Hopper / Blackwell")
    per_edge, cot = _inputs(0)
    whole_reads, whole_grads = _run(_make(impl, EDGES), per_edge, cot)

    first, second = _make(impl, SPLIT), _make(impl, EDGES - SPLIT)
    reads = [first.forward(e, *per_edge[e]).float().clone() for e in range(SPLIT)]
    m, p = (t.clone() for t in first.export_state())
    second.load_state(m, p)
    reads += [second.forward(e, *per_edge[SPLIT + e]).float().clone()
              for e in range(EDGES - SPLIT)]
    grads = [None] * EDGES
    for e in reversed(range(EDGES - SPLIT)):
        grads[SPLIT + e] = [t.float().clone() for t in second.backward(e, cot[SPLIT + e])]
    dm, dp = (t.clone() for t in second.export_grad())
    first.load_grad(dm, dp)
    for e in reversed(range(SPLIT)):
        grads[e] = [t.float().clone() for t in first.backward(e, cot[e])]

    tol = dict(rtol=1e-4, atol=1e-5) if impl == "torch" else dict(rtol=2e-2, atol=2e-3)
    for e in range(EDGES):
        torch.testing.assert_close(reads[e], whole_reads[e], **tol, msg=f"read {e}")
        for name, a, b in zip(("khat", "query", "value", "scale", "forgetting"), grads[e],
                              whole_grads[e]):
            torch.testing.assert_close(a, b, **tol, msg=f"edge {e} d{name}")
