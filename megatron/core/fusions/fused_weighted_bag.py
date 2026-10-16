# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Weighted bag of table rows, forward and backward, in any float dtype.

``out[t] = sum_j w[t, j] * V[idx[t, j]]`` for ``V`` ``[U, D]``, ``idx`` / ``w`` ``[T, J]``:
the read of product-key memory (``J = heads * topk`` rows per token). It is
``F.embedding_bag(idx, V, per_sample_weights=w, mode="sum")``, whose backward for the
per-sample weights has no bf16 kernel -- so it either upcasts the rows or materializes
the ``[T, J, D]`` expansion (GBs per layer). Here:

* forward: one program per (token, column block) gathers ``[BJ, BD]`` row tiles and
  accumulates in fp32;
* ``dw[t, j] = <dout[t], V[idx[t, j]]>``: one program per token;
* ``dV[u] = sum_{(t, j): idx[t, j] = u} w[t, j] * dout[t]``: the lookups are sorted by row
  once and every row sums its own occurrences (a segment reduction -- deterministic, no
  atomics).

Inputs keep their dtype (bf16 rows, fp32 weights); all sums are fp32.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


def weighted_bag_reference(table: torch.Tensor, index: torch.Tensor, weights: torch.Tensor):
    """fp32 torch semantics: ``[T, D]``."""
    return torch.nn.functional.embedding_bag(
        index, table.float(), per_sample_weights=weights.float(), mode="sum"
    )


if HAVE_TRITON:

    @triton.jit
    def _bag_fwd(idx_ptr, w_ptr, v_ptr, out_ptr, J, D, BJ: tl.constexpr, BD: tl.constexpr):
        t = tl.program_id(0).to(tl.int64)
        d = tl.program_id(1) * BD + tl.arange(0, BD)
        dm = d < D
        acc = tl.zeros((BD,), tl.float32)
        for j0 in range(0, J, BJ):
            jj = j0 + tl.arange(0, BJ)
            jm = jj < J
            ix = tl.load(idx_ptr + t * J + jj, mask=jm, other=0).to(tl.int64)
            wv = tl.load(w_ptr + t * J + jj, mask=jm, other=0.0).to(tl.float32)
            rows = tl.load(
                v_ptr + ix[:, None] * D + d[None, :], mask=jm[:, None] & dm[None, :], other=0.0
            ).to(tl.float32)
            acc += tl.sum(wv[:, None] * rows, axis=0)
        tl.store(out_ptr + t * D + d, acc.to(out_ptr.dtype.element_ty), mask=dm)

    @triton.jit
    def _bag_bwd_w(idx_ptr, v_ptr, g_ptr, dw_ptr, J, D, BJ: tl.constexpr, BD: tl.constexpr):
        t = tl.program_id(0).to(tl.int64)
        d = tl.arange(0, BD)
        dm = d < D
        g = tl.load(g_ptr + t * D + d, mask=dm, other=0.0).to(tl.float32)
        for j0 in range(0, J, BJ):
            jj = j0 + tl.arange(0, BJ)
            jm = jj < J
            ix = tl.load(idx_ptr + t * J + jj, mask=jm, other=0).to(tl.int64)
            rows = tl.load(
                v_ptr + ix[:, None] * D + d[None, :], mask=jm[:, None] & dm[None, :], other=0.0
            ).to(tl.float32)
            tl.store(dw_ptr + t * J + jj, tl.sum(rows * g[None, :], axis=1), mask=jm)

    @triton.jit
    def _bag_bwd_v(
        order_ptr, offsets_ptr, w_ptr, g_ptr, dv_ptr, J, D, BP: tl.constexpr, BD: tl.constexpr
    ):
        u = tl.program_id(0).to(tl.int64)
        d = tl.program_id(1) * BD + tl.arange(0, BD)
        dm = d < D
        start = tl.load(offsets_ptr + u)
        end = tl.load(offsets_ptr + u + 1)
        acc = tl.zeros((BD,), tl.float32)
        for p0 in range(start, end, BP):
            pp = p0 + tl.arange(0, BP)
            pm = pp < end
            pos = tl.load(order_ptr + pp, mask=pm, other=0).to(tl.int64)  # flat t * J + j
            wv = tl.load(w_ptr + pos, mask=pm, other=0.0).to(tl.float32)
            g = tl.load(
                g_ptr + (pos // J)[:, None] * D + d[None, :], mask=pm[:, None] & dm[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(wv[:, None] * g, axis=0)
        tl.store(dv_ptr + u * D + d, acc.to(dv_ptr.dtype.element_ty), mask=dm)


_BJ, _BD, _BP = 16, 128, 16


class _WeightedBag(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table, index, weights, out_dtype):
        table, index, weights = table.contiguous(), index.contiguous(), weights.contiguous()
        t, j = index.shape
        d = table.shape[1]
        out = torch.empty((t, d), dtype=out_dtype, device=table.device)
        _bag_fwd[(t, triton.cdiv(d, _BD))](index, weights, table, out, j, d, BJ=_BJ, BD=_BD)
        ctx.save_for_backward(table, index, weights)
        return out

    @staticmethod
    def backward(ctx, grad):
        table, index, weights = ctx.saved_tensors
        grad = grad.contiguous()
        t, j = index.shape
        u, d = table.shape
        dw = None
        if ctx.needs_input_grad[2]:
            dw = torch.empty((t, j), dtype=torch.float32, device=table.device)
            _bag_bwd_w[(t,)](
                index, table, grad, dw, j, d, BJ=_BJ, BD=triton.next_power_of_2(d), num_warps=8
            )
            dw = dw.to(weights.dtype)
        dv = None
        if ctx.needs_input_grad[0]:
            flat = index.reshape(-1)
            order = torch.argsort(flat, stable=True)
            offsets = torch.zeros(u + 1, dtype=torch.int64, device=table.device)
            offsets[1:] = torch.cumsum(torch.bincount(flat, minlength=u), 0)
            dv = torch.empty_like(table)
            _bag_bwd_v[(u, triton.cdiv(d, _BD))](
                order, offsets, weights, grad, dv, j, d, BP=_BP, BD=_BD
            )
        return dv, None, dw, None


def weighted_bag(
    table: torch.Tensor, index: torch.Tensor, weights: torch.Tensor, out_dtype=torch.float32
) -> torch.Tensor:
    """``out[t] = sum_j weights[t, j] * table[index[t, j]]``, ``[T, D]`` in ``out_dtype``.

    ``table`` ``[U, D]`` (any float dtype), ``index`` ``[T, J]`` (int, rows of ``table``),
    ``weights`` ``[T, J]``. Torch fallback (fp32 embedding_bag) off CUDA / without Triton.
    """
    if not (HAVE_TRITON and table.is_cuda):
        return weighted_bag_reference(table, index, weights).to(out_dtype)
    return _WeightedBag.apply(table, index, weights, out_dtype)
