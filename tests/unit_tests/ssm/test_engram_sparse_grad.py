# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""ENGRAM_SPARSE_GRAD: the touched-rows gradient buffer accumulates what the dense one does, and
TableAdam over it is the dense update (untouched rows still move by their momentum)."""

import pytest
import torch

from megatron.core.optimizer.engram_optimizer import TableAdam
from megatron.core.ssm.engram_sharded import SparseRowGrad

ROWS, DIM = 3000, 64


def _contributions(seed, n_calls=6, per_call=500):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for _ in range(n_calls):  # duplicates within and across calls, like the lookups' backwards
        idx = torch.randint(0, ROWS // 3, (per_call,), device="cuda", generator=gen)
        out.append((idx, torch.randn(per_call, DIM, device="cuda", generator=gen)))
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("coalesce_bytes", [256 << 20, 4096])  # never / every call
def test_sparse_matches_dense_buffer(coalesce_bytes):
    dense = torch.zeros(ROWS, DIM, device="cuda")
    sparse = SparseRowGrad(ROWS, DIM, "cuda", coalesce_bytes=coalesce_bytes)
    for idx, val in _contributions(0):
        dense.index_add_(0, idx, val)
        sparse.index_add_(0, idx, val)
    torch.testing.assert_close(sparse.to_dense(), dense)
    vals = sparse.coalesced_values()
    assert vals.shape[0] < ROWS // 3 + 1  # touched rows only
    torch.testing.assert_close(vals.square().sum(), dense.square().sum())
    sparse.mul_(0.5)
    torch.testing.assert_close(sparse.to_dense(), dense * 0.5)
    sparse.zero_()
    assert sparse.to_dense().abs().sum() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("rowwise_v,momentum", [(True, torch.bfloat16), (False, torch.float32)])
def test_table_adam_sparse_is_dense(monkeypatch, rowwise_v, momentum):
    import megatron.core.optimizer.engram_optimizer as eo

    monkeypatch.setattr(eo, "_CHUNK_ELEMENTS", 700 * DIM)  # several chunks, ragged last one
    torch.manual_seed(0)
    init = torch.randn(ROWS, DIM, device="cuda")
    p_dense = torch.nn.Parameter(init.clone(), requires_grad=False)
    p_sparse = torch.nn.Parameter(init.clone(), requires_grad=False)
    p_dense.engram_grad = torch.zeros(ROWS, DIM, device="cuda")
    p_sparse.engram_grad = SparseRowGrad(ROWS, DIM, "cuda")
    kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, rowwise_v=rowwise_v, momentum_dtype=momentum)
    opt_d, opt_s = TableAdam([p_dense], **kw), TableAdam([p_sparse], **kw)
    for step in range(4):
        # step 3 touches nothing: the momentum alone moves the rows, in both
        contrib = _contributions(10 + step) if step != 3 else []
        for idx, val in contrib:
            p_sparse.engram_grad.index_add_(0, idx, val)
        # the dense path gets the same summed gradient: the two buffers' atomic summation
        # orders differ in the last bits (test_sparse_matches_dense_buffer), which Adam amplifies
        # for elements whose gradient nearly cancels -- here only the update is compared
        p_dense.engram_grad.copy_(p_sparse.engram_grad.to_dense())
        p_dense.grad = p_dense.engram_grad
        p_sparse.grad = None  # as EngramTableOptimizer.prepare_grads leaves a sparse one
        opt_d.step()
        opt_s.step()
        p_dense.engram_grad.zero_()
        p_sparse.engram_grad.zero_()
    torch.testing.assert_close(p_sparse.data, p_dense.data, rtol=1e-6, atol=1e-6)
    sd, ss = opt_d.state[p_dense], opt_s.state[p_sparse]
    torch.testing.assert_close(ss["exp_avg"].float(), sd["exp_avg"].float(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ss["exp_avg_sq"], sd["exp_avg_sq"], rtol=1e-6, atol=1e-6)
    # rows no step touched (>= ROWS // 3) moved neither
    torch.testing.assert_close(p_sparse.data[ROWS // 3 + 1 :], init[ROWS // 3 + 1 :])
