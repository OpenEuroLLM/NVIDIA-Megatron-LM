# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

r"""Triton kernels for the mHC write backward.

Profiling a 130M mHC run on GH200 (`dump/residual_arch_final.md`) put roughly
half of everything mHC costs in the *backward* of the stream read and write --
a part no earlier optimization round had looked at. The write backward has four
terms:

.. code-block:: text

    grad_streams[t,j,c] = sum_i h_res[t,i,j] * g[t,i,c]     contract over n
    grad_out[t,c]       = sum_i h_post[t,i] * g[t,i,c]      contract over n
    grad_h_res[t,i,j]   = sum_c g[t,i,c] * streams[t,j,c]   contract over C
    grad_h_post[t,i]    = sum_c g[t,i,c] * out[t,c]         contract over C

Autograd derives each from the forward expression and Inductor emits four
kernels, every one of which re-reads ``g`` -- the largest tensor mHC owns, 604 MB
at the training shapes. Measured against a 3617 GB/s ceiling, three of the four
run at 63-84% of peak; ``grad_h_res`` runs at 28%, and is the single largest
kernel mHC adds to the model.

Measured on GH200 at the training shapes (T = 4096 x 32, C = 576, n = 4):

.. code-block:: text

    grad_h_res alone        inductor 1.164 ms | matmul 0.848 | this file 1.162
    full write backward     inductor 2.150 ms | this file 1.842            1.17x

Two kernels are provided, and only one of them is worth using:

* :func:`write_backward` -- all four terms in one pass over ``g``. Traffic drops
  from ~3.9 GB to ~2.1 GB per sublayer because ``g`` is read once instead of four
  times, which is the part Inductor cannot do: it will not fuse reductions whose
  iteration spaces differ. Worth 1.17x on the backward.
* :func:`grad_h_res` -- that one term on its own. It ties Inductor and loses to a
  plain ``torch.matmul``, which reaches 39% of peak against its 29%: cuBLAS is
  simply better at a C-long contraction into an n x n output than anything here.
  Kept because it isolates the term for testing and gives :func:`write_backward`
  something to be checked against, not because it should be deployed.

Note that 1.842 ms is still only 31% of peak for the traffic it moves, so the
fused kernel has headroom left; the tile search below is coarse.

Both fall back to the PyTorch formulation when Triton is unavailable or the
inputs do not meet the kernels' constraints, so neither can break a run.
"""

import os

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - depends on the runtime
    HAVE_TRITON = False


_ENABLED = os.environ.get("MHC_TRITON_BACKWARD", "1") != "0"


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


if HAVE_TRITON:
    # A first cut with one token per program reached only 20% of peak: with n=4
    # each program loaded a 4 x BLOCK_C sliver and then ran a short serial loop,
    # so the kernel was latency-bound rather than bandwidth-bound. Tiling over
    # tokens as well gives each program enough independent work to hide that.
    #
    # The right tile is a hardware question, and the answer measured on a local
    # 3090 Ti does not transfer to GH200 -- so let Triton pick it on whatever
    # device actually runs. Autotuning keys on (C, N), which are fixed for a
    # given model, so the search happens once.
    # The bound on BLOCK_T * BLOCK_C is what keeps this list sane. The inner
    # products materialize a [BLOCK_T, N, N, BLOCK_C] tile of fp32 accumulators,
    # so the register footprint is N^2 * BLOCK_T * BLOCK_C * 4 bytes -- 32 KB at
    # the cap below with n=4. Past that, occupancy collapses and the kernel gets
    # slower rather than faster, so there is no point searching there.
    _CONFIGS = [
        triton.Config({"BLOCK_T": bt, "BLOCK_C": bc}, num_warps=w, num_stages=st)
        for bt in (1, 2, 4, 8, 16, 32)
        for bc in (32, 64, 128, 256)
        for w in (4, 8, 16)
        for st in (2, 3)
        if bt * bc <= 512
    ]

    @triton.autotune(configs=_CONFIGS, key=["C", "N"])
    @triton.jit
    def _grad_h_res_kernel(
        g_ptr,
        s_ptr,
        out_ptr,
        T,
        C,
        N: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Accumulate the n x n outer product over C, for a tile of tokens."""
        t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = t < T
        i = tl.arange(0, N)
        acc = tl.zeros((BLOCK_T, N, N), dtype=tl.float32)

        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask = mask_t[:, None, None] & (offs_c < C)[None, None, :]
            block = t[:, None, None] * N * C + i[None, :, None] * C + offs_c[None, None, :]
            g = tl.load(g_ptr + block, mask=mask, other=0.0).to(tl.float32)
            s = tl.load(s_ptr + block, mask=mask, other=0.0).to(tl.float32)
            # acc[bt, i, j] += sum_c g[bt, i, c] * s[bt, j, c]
            acc += tl.sum(g[:, :, None, :] * s[:, None, :, :], axis=3)

        offs = t[:, None, None] * N * N + i[None, :, None] * N + i[None, None, :]
        tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask_t[:, None, None])

    @triton.autotune(configs=_CONFIGS, key=["C", "N"])
    @triton.jit
    def _write_backward_kernel(
        g_ptr,
        s_ptr,
        out_ptr,
        h_post_ptr,
        h_res_ptr,
        d_streams_ptr,
        d_out_ptr,
        d_h_post_ptr,
        d_h_res_ptr,
        T,
        C,
        N: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """All four write-backward terms in a single pass over ``g``.

        Each program owns a tile of tokens and walks C once. The two terms that
        contract over the stream axis are written out chunk by chunk; the two
        that contract over C accumulate in registers and are stored at the end.
        Reading ``g`` once instead of four times is the whole point -- it is what
        Inductor cannot do, because it will not fuse reductions whose iteration
        spaces differ.
        """
        t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = t < T
        i = tl.arange(0, N)

        h_res = tl.load(
            h_res_ptr + t[:, None, None] * N * N + i[None, :, None] * N + i[None, None, :],
            mask=mask_t[:, None, None],
            other=0.0,
        ).to(tl.float32)
        h_post = tl.load(
            h_post_ptr + t[:, None] * N + i[None, :], mask=mask_t[:, None], other=0.0
        ).to(tl.float32)

        acc_h_res = tl.zeros((BLOCK_T, N, N), dtype=tl.float32)
        acc_h_post = tl.zeros((BLOCK_T, N), dtype=tl.float32)

        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask_c = offs_c < C
            mask = mask_t[:, None, None] & mask_c[None, None, :]
            block = t[:, None, None] * N * C + i[None, :, None] * C + offs_c[None, None, :]
            flat = t[:, None] * C + offs_c[None, :]
            mask_flat = mask_t[:, None] & mask_c[None, :]

            g = tl.load(g_ptr + block, mask=mask, other=0.0).to(tl.float32)
            s = tl.load(s_ptr + block, mask=mask, other=0.0).to(tl.float32)
            o = tl.load(out_ptr + flat, mask=mask_flat, other=0.0).to(tl.float32)

            # grad_streams[bt, j, c] = sum_i h_res[bt, i, j] * g[bt, i, c]
            d_streams = tl.sum(h_res[:, :, :, None] * g[:, :, None, :], axis=1)
            tl.store(d_streams_ptr + block, d_streams.to(d_streams_ptr.dtype.element_ty), mask=mask)
            # grad_out[bt, c] = sum_i h_post[bt, i] * g[bt, i, c]
            d_out = tl.sum(h_post[:, :, None] * g, axis=1)
            tl.store(d_out_ptr + flat, d_out.to(d_out_ptr.dtype.element_ty), mask=mask_flat)

            acc_h_res += tl.sum(g[:, :, None, :] * s[:, None, :, :], axis=3)
            acc_h_post += tl.sum(g * o[:, None, :], axis=2)

        tl.store(
            d_h_res_ptr + t[:, None, None] * N * N + i[None, :, None] * N + i[None, None, :],
            acc_h_res.to(d_h_res_ptr.dtype.element_ty),
            mask=mask_t[:, None, None],
        )
        tl.store(
            d_h_post_ptr + t[:, None] * N + i[None, :],
            acc_h_post.to(d_h_post_ptr.dtype.element_ty),
            mask=mask_t[:, None],
        )


def can_use(streams: torch.Tensor, n: int) -> bool:
    """Whether the Triton path can run on these inputs.

    ``tl.arange`` requires a power-of-two extent, so the stream count has to be
    one; mHC uses 2 or 4 in practice. Everything else falls back.

    Callers should check this *before* routing through a custom autograd
    Function: the PyTorch fallbacks below are correct but materialize a
    ``[t, n, n, C]`` intermediate, which is four times the stream state. Leaving
    the backward to Inductor is the better answer when the kernel cannot run.

    Set ``MHC_TRITON_BACKWARD=0`` to force that path -- a kill switch for a
    hand-written kernel, and the toggle used to A/B it.
    """
    return _ENABLED and HAVE_TRITON and streams.is_cuda and _is_pow2(n)


_usable = can_use  # internal alias used by the launchers below


def grad_h_res(
    g: torch.Tensor, streams: torch.Tensor, fallback: bool = True
) -> torch.Tensor | None:
    """``grad_h_res[t,i,j] = sum_c g[t,i,c] * streams[t,j,c]``.

    Args:
        g: ``[s, b, n, C]`` gradient of the written streams.
        streams: ``[s, b, n, C]`` stream state.
        fallback: Return the PyTorch result when the Triton path is unusable.
            When False, return None instead so a caller can branch.

    Returns:
        ``[s, b, n, n]`` in the dtype of ``g``, or None -- see ``fallback``.
    """
    s_, b_, n, c = streams.shape
    if not _usable(streams, n):
        if not fallback:
            return None
        return (g.unsqueeze(-2) * streams.unsqueeze(-3)).sum(dim=-1)

    g = g.contiguous()
    streams = streams.contiguous()
    out = torch.empty((s_, b_, n, n), dtype=g.dtype, device=g.device)
    grid = lambda meta: (triton.cdiv(s_ * b_, meta["BLOCK_T"]),)
    _grad_h_res_kernel[grid](g, streams, out, s_ * b_, c, N=n)
    return out


def write_backward(
    g: torch.Tensor,
    streams: torch.Tensor,
    sublayer_out: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    fallback: bool = True,
):
    """All four mHC write-backward terms, reading ``g`` once.

    Args:
        g: ``[s, b, n, C]`` gradient of the written streams.
        streams: ``[s, b, n, C]`` stream state the forward was given.
        sublayer_out: ``[s, b, C]`` sublayer output the forward was given.
        h_post: ``[s, b, n]`` write coefficients.
        h_res: ``[s, b, n, n]`` stream mixer.
        fallback: Use the PyTorch formulation when Triton is unusable. When
            False, return None instead.

    Returns:
        ``(grad_streams, grad_sublayer_out, grad_h_post, grad_h_res)``, or None.
    """
    s_, b_, n, c = streams.shape
    if not _usable(streams, n):
        if not fallback:
            return None
        return (
            (h_res.unsqueeze(-1) * g.unsqueeze(-2)).sum(dim=-3),
            (h_post.unsqueeze(-1) * g).sum(dim=-2),
            (g * sublayer_out.unsqueeze(-2)).sum(dim=-1),
            (g.unsqueeze(-2) * streams.unsqueeze(-3)).sum(dim=-1),
        )

    g = g.contiguous()
    streams = streams.contiguous()
    sublayer_out = sublayer_out.contiguous()
    h_post = h_post.contiguous()
    h_res = h_res.contiguous()

    d_streams = torch.empty_like(streams)
    d_out = torch.empty_like(sublayer_out)
    d_h_post = torch.empty_like(h_post)
    d_h_res = torch.empty_like(h_res)

    grid = lambda meta: (triton.cdiv(s_ * b_, meta["BLOCK_T"]),)
    _write_backward_kernel[grid](
        g,
        streams,
        sublayer_out,
        h_post,
        h_res,
        d_streams,
        d_out,
        d_h_post,
        d_h_res,
        s_ * b_,
        c,
        N=n,
    )
    return d_streams, d_out, d_h_post, d_h_res
