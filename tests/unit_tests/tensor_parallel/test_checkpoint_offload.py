# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""RECOMPUTE_OFFLOAD: checkpoint inputs in pinned host memory give the same gradients and free the
GPU copies (megatron/core/tensor_parallel/checkpoint_offload.py)."""

import pytest
import torch

from megatron.core.tensor_parallel import checkpoint_offload
from megatron.core.tensor_parallel.random import checkpoint, model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils

T, H, CHUNKS, SEQS = 4096, 2048, 6, 3  # 16 MB chunk inputs (bf16)


def _run(offload, monkeypatch, last=False, leaf=False, offload_leaf=False):
    monkeypatch.setattr(checkpoint_offload, "ENABLED", offload)
    monkeypatch.setattr(checkpoint_offload, "OFFLOAD_LAST", last)
    monkeypatch.setattr(checkpoint_offload, "OFFLOAD_LEAF", offload_leaf)
    checkpoint_offload._queues.clear()
    torch.manual_seed(0)
    weights = [
        (torch.randn(H, H, device="cuda") / H**0.5).to(torch.bfloat16).requires_grad_()
        for _ in range(CHUNKS)
    ]
    x0 = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def chunk(w):
        return lambda h: torch.tanh(h @ w) + h

    outs, recv, peak_fwd = [], [], 0
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    for s in range(SEQS):  # several sequences in flight, like a pipelined schedule
        checkpoint_offload.new_sequence("chunk")
        if leaf:  # a pipeline receive buffer: a leaf the schedule keeps for its .grad
            h = (x0.detach() * (s + 1)).requires_grad_()
            recv.append(h)
        else:
            h = x0 * (s + 1)  # non-leaf: the first chunk's input is offloadable too
        for w in weights:
            checkpoint_offload.arm()
            h = checkpoint(chunk(w), False, h)
        outs.append(h)
        torch.cuda.synchronize()
        peak_fwd = max(peak_fwd, torch.cuda.memory_allocated() - base)
    checkpoint_offload.new_sequence(None)  # (what the next forward would do: close the last one)
    for s in range(SEQS):  # backward in forward order, like a model chunk's micro-batches
        outs[s].float().pow(2).mean().backward()
    torch.cuda.synchronize()
    gx = torch.stack([r.grad for r in recv]) if leaf else x0.grad.clone()
    return peak_fwd, gx, [w.grad.clone() for w in weights]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_offload_same_grads_less_memory(monkeypatch):
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    try:
        mem_ref, gx_ref, gw_ref = _run(False, monkeypatch)
        mem_off, gx_off, gw_off = _run(True, monkeypatch)
    finally:
        Utils.destroy_model_parallel()
    assert torch.equal(gx_ref, gx_off)
    for a, b in zip(gw_ref, gw_off):
        assert torch.equal(a, b)
    chunk_bytes = T * H * 2
    # without: every chunk input of every sequence on the GPU; with: about two per sequence
    assert mem_ref >= SEQS * CHUNKS * chunk_bytes
    assert mem_off <= mem_ref - SEQS * (CHUNKS - 3) * chunk_bytes, (mem_ref, mem_off)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_offload_last_same_grads(monkeypatch):
    """RECOMPUTE_OFFLOAD_LAST: the last chunk of every sequence too (fetched back ahead in the
    backward's forward order) -- same gradients."""
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    try:
        _, gx_ref, gw_ref = _run(False, monkeypatch)
        _, gx_off, gw_off = _run(True, monkeypatch, last=True)
    finally:
        Utils.destroy_model_parallel()
    assert torch.equal(gx_ref, gx_off)
    for a, b in zip(gw_ref, gw_off):
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_offload_leaf_same_grads_less_memory(monkeypatch):
    """RECOMPUTE_OFFLOAD_LEAF: a leaf first input (pipeline receive buffer) goes to the host too;
    its object keeps the gradient the schedule reads."""
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    try:
        mem_keep, gx_ref, gw_ref = _run(True, monkeypatch, leaf=True)
        mem_leaf, gx_off, gw_off = _run(True, monkeypatch, leaf=True, offload_leaf=True)
        _, gx_last, gw_last = _run(True, monkeypatch, last=True, leaf=True, offload_leaf=True)
        _, gx_none, gw_none = _run(False, monkeypatch, leaf=True)  # no offload at all
    finally:
        Utils.destroy_model_parallel()
    for gx, gw in ((gx_ref, gw_ref), (gx_off, gw_off), (gx_last, gw_last)):
        assert torch.equal(gx_none, gx)
        for a, b in zip(gw_none, gw):
            assert torch.equal(a, b)
    # the leaf of every sequence leaves the GPU
    assert mem_leaf <= mem_keep - SEQS * T * H * 2, (mem_keep, mem_leaf)
