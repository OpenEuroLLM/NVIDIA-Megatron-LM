# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

r"""mHC: Manifold-Constrained Hyper-Connections.

Implements mHC (`arXiv:2512.24880 <https://arxiv.org/abs/2512.24880>`_), which
widens the residual stream into ``n`` parallel streams (Hyper-Connections, Zhu et
al. 2024) and restores the identity-mapping property that plain HC loses.

Given an ``n``-stream residual :math:`\mathbf{x}_l \in \mathbb{R}^{n \times C}`, a
layer propagates as (paper Eq. 3)

.. math::
    \mathbf{x}_{l+1} = \mathcal{H}^{\mathrm{res}}_l \mathbf{x}_l
        + {\mathcal{H}^{\mathrm{post}}_l}^\top
          \mathcal{F}(\mathcal{H}^{\mathrm{pre}}_l \mathbf{x}_l, \mathcal{W}_l)

where :math:`\mathcal{H}^{\mathrm{pre}} \in \mathbb{R}^{1 \times n}` reads a
``C``-dim layer input out of the streams, :math:`\mathcal{H}^{\mathrm{post}} \in
\mathbb{R}^{1 \times n}` writes the layer output back, and
:math:`\mathcal{H}^{\mathrm{res}} \in \mathbb{R}^{n \times n}` mixes the streams.

Plain HC leaves :math:`\mathcal{H}^{\mathrm{res}}` unconstrained, so the composite
map :math:`\prod_l \mathcal{H}^{\mathrm{res}}_l` drifts away from the identity --
the paper measures signal gains above 3000x at 27B, and training diverges. mHC
projects :math:`\mathcal{H}^{\mathrm{res}}` onto the **Birkhoff polytope** (doubly
stochastic matrices) with Sinkhorn-Knopp. Doubly stochastic matrices are
non-expansive (:math:`\lVert \cdot \rVert_2 \le 1`) and closed under
multiplication, so the composite map stays norm-preserving at any depth, and the
mix acts as a convex combination of permutations.

All coefficients are input-dependent (paper Eq. 7): they are linear projections of
the RMSNormed, flattened stream state, gated by learnable scalars
:math:`\alpha^{\mathrm{pre}}, \alpha^{\mathrm{post}}, \alpha^{\mathrm{res}}`.

Composition with nGPT
---------------------
mHC and nGPT both constrain residual magnitude but at different places: mHC makes
the *mixing operator* norm-preserving while leaving stream magnitude free; nGPT
pins the *stream* to the unit sphere. When ``lerp`` is supplied, the stream mix
runs **first** and each stream then takes an nGPT LERP step toward the normalized
sublayer output -- mixing before renormalization, since a convex combination of
unit vectors is not itself unit norm.
"""

import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def _sinkhorn_knopp_impl(
    logits: torch.Tensor, num_iters: int = 20, eps: float = 1e-8
) -> torch.Tensor:
    """Project a matrix onto the Birkhoff polytope via Sinkhorn-Knopp.

    Follows paper Eq. 9: exponentiate to make every entry positive, then
    alternately normalize columns and rows until both sum to one. The result is
    doubly stochastic, hence non-expansive and closed under composition.

    Runs in fp32 regardless of the incoming dtype -- the alternating
    normalization is numerically delicate and the matrices are tiny (n x n).

    Convergence is asymptotic: because the sweep ends with a row normalization,
    row sums are exact while column sums approach 1 only as ``num_iters`` grows.
    The paper picks 20 as a practical value. Residual column error is negligible
    for diffuse matrices but grows for near-permutation ones (where Sinkhorn
    converges slowly); raise ``num_iters`` if a run needs tighter double
    stochasticity.

    Args:
        logits: Tensor of shape ``[..., n, n]`` holding the unconstrained scores.
        num_iters: Number of alternating normalization sweeps (paper uses 20).
        eps: Numerical floor for the divisions.

    Returns:
        A row-stochastic, approximately doubly stochastic tensor, in fp32.
    """
    # exp() in fp32; subtract the max for numerical safety (this cancels in the
    # subsequent normalization, so it does not change the fixed point).
    x = logits.float()
    x = x - x.amax(dim=(-2, -1), keepdim=True)
    m = torch.exp(x)
    for _ in range(num_iters):
        m = m / m.sum(dim=-2, keepdim=True).clamp(min=eps)  # columns sum to 1
        m = m / m.sum(dim=-1, keepdim=True).clamp(min=eps)  # rows sum to 1
    return m


# The loop is the hot spot: each sweep is ~4 memory-bound elementwise passes over
# an [s, b, n, n] tensor, so 8 sweeps is ~32 separate kernels per sublayer and
# ~1150 per forward at 18 layers. Compiling unrolls the loop and fuses it into a
# handful of kernels, which is the difference between mHC being launch-bound and
# not. Falls back to eager if Inductor is unavailable (e.g. no libcuda on the
# compile path), so this can never break a run.
_sinkhorn_compiled = None


def _mhc_coeffs(flat, phi, b_pre, b_post, b_res, a_pre, a_post, a_res, n, eps, iters):
    """Fused mHC coefficient computation (paper Eq. 7-9).

    Kept as a free function so Inductor can fuse the projection, the RMS scale,
    the three gated adds and the Sinkhorn sweeps into a handful of kernels
    instead of ~40 launches per sublayer.
    """
    s_, b_, _ = flat.shape
    r = torch.linalg.vector_norm(flat, dim=-1, keepdim=True).float() / (flat.shape[-1] ** 0.5)
    coeffs = torch.matmul(flat, phi.to(flat.dtype)).float() / r.clamp(min=eps)
    pre_raw, post_raw, res_raw = torch.split(coeffs, [n, n, n * n], dim=-1)
    h_pre = torch.sigmoid(a_pre.float() * pre_raw + b_pre.float())
    h_post = 2.0 * torch.sigmoid(a_post.float() * post_raw + b_post.float())
    h_res = a_res.float() * res_raw.view(s_, b_, n, n) + b_res.float()
    return h_pre, h_post, _sinkhorn_knopp_impl(h_res, iters, 1e-8)


_mhc_coeffs_compiled = None


def _mhc_write(streams, sublayer_out, h_post, h_res):
    """Fused mHC stream update: ``H_res @ x + H_post^T f(x)`` in one pass.

    Args:
        streams: ``[s, b, n, C]`` current stream state.
        sublayer_out: ``[s, b, C]`` attention or MLP output.
        h_post: ``[s, b, n]`` write coefficients.
        h_res: ``[s, b, n, n]`` doubly stochastic stream mixer.

    Returns:
        ``[s, b, n, C]`` updated streams.

    The previous formulation materialized the write-back term as a full
    ``[s, b, n, C]`` tensor and then called ``baddbmm``. That is the wrong shape
    of computation here: with ``n`` around 2-4 the "matmul" is a handful of
    multiply-adds per output element, so a batched GEMM with a 4-long inner
    dimension gets no efficiency while the intermediate doubles traffic over the
    widest tensor mHC owns.

    Written as one expression, Inductor fuses the broadcasted product straight
    into the reduction and the write-back into the same epilogue, so nothing
    between the inputs and the output is ever materialized. Traffic drops from
    roughly 4x to 2x the stream state per sublayer.
    """
    mixed = (h_res.unsqueeze(-1) * streams.unsqueeze(-3)).sum(dim=-2)
    return mixed + h_post.unsqueeze(-1) * sublayer_out.unsqueeze(-2)


_mhc_write_compiled = None


class _MHCWrite(torch.autograd.Function):
    """The mHC write, with the backward routed through a fused Triton kernel.

    The forward is unchanged -- still the compiled expression above, still at 84%
    of GH200's bandwidth. Only the backward differs: autograd would derive four
    separate kernels from that expression, each re-reading the upstream gradient,
    which is 604 MB at the training shapes and the largest tensor mHC owns.
    :func:`~megatron.core.transformer.mhc_kernels.write_backward` computes all
    four in one pass, cutting the backward's traffic from ~3.9 GB to ~2.1 GB.

    Only used when the kernel can actually run on the inputs; see
    :func:`~megatron.core.transformer.mhc_kernels.can_use`. When it cannot, the
    caller stays on the plain compiled path and lets autograd do the backward,
    because the kernel module's PyTorch fallback materializes a ``[t, n, n, C]``
    intermediate that would be worse than what Inductor emits.
    """

    @staticmethod
    def forward(ctx, streams, sublayer_out, h_post, h_res):
        ctx.save_for_backward(streams, sublayer_out, h_post, h_res)
        out = _mhc_write_dispatch(streams, sublayer_out, h_post, h_res)
        _prewarm_backward(out, streams, sublayer_out, h_post, h_res)
        return out

    @staticmethod
    def backward(ctx, g):
        from megatron.core.transformer.mhc_kernels import write_backward

        streams, sublayer_out, h_post, h_res = ctx.saved_tensors
        # The kernel indexes as if contiguous; g arrives from whatever produced it.
        return write_backward(g.contiguous(), streams, sublayer_out, h_post, h_res)


_BACKWARD_PREWARMED = False


def _prewarm_backward(grad_like, streams, sublayer_out, h_post, h_res):
    """Run the backward kernel once during a forward, and discard the result.

    Triton autotuning benchmarks every candidate tile on the first call for a
    given key. Left alone, that first call is the one Megatron makes *inside*
    ``torch.cuda.graph`` capture for the backward graph, and benchmarking there
    issues host-side work that invalidates the capture -- an abort, not a
    slowdown. Forwards always run eagerly before any capture (the graph runner
    performs warmup steps first), so this puts the search somewhere safe.

    Once per process, never while a capture is already running, and at the real
    shapes so the tuned configuration is the one the actual backward will use.
    """
    global _BACKWARD_PREWARMED
    if _BACKWARD_PREWARMED:
        return
    if grad_like.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    _BACKWARD_PREWARMED = True

    from megatron.core.transformer.mhc_kernels import write_backward

    with torch.no_grad():
        write_backward(grad_like, streams, sublayer_out, h_post, h_res)


def _mhc_write_dispatch(streams, sublayer_out, h_post, h_res):
    """Compiled forward write, falling back to eager if Inductor is unusable."""
    global _mhc_write_compiled
    if _mhc_write_compiled is None:
        try:
            _mhc_write_compiled = torch.compile(_mhc_write, dynamic=False)
        except Exception:  # pragma: no cover
            _mhc_write_compiled = _mhc_write
    try:
        return _mhc_write_compiled(streams, sublayer_out, h_post, h_res)
    except Exception:  # pragma: no cover - Inductor unavailable at runtime
        _mhc_write_compiled = _mhc_write
        return _mhc_write(streams, sublayer_out, h_post, h_res)


def _mhc_mix_normalized(streams, h_res, eps):
    """The stream mix, renormalized onto the unit sphere for the nGPT LERP.

    Args:
        streams: ``[s, b, n, C]`` current stream state.
        h_res: ``[s, b, n, n]`` doubly stochastic stream mixer.
        eps: Numerical floor for the division.

    Returns:
        ``[s, b, n, C]`` mixed streams, each row unit norm.

    Two things this deliberately avoids, both costing a full ``[s, b, n, C]``
    tensor, and together enough to OOM a 130M run at n=4 before its first
    iteration:

    * ``torch.einsum`` for an ``n``-long contraction, which dispatches a batched
      GEMM over the widest tensor in the model to do a handful of multiply-adds
      per output element. The additive write already had this removed; the LERP
      branch kept it.
    * ``F.normalize(mixed.float(), ...)``, which retains an fp32 copy of the
      whole mixed state for backward. ``vector_norm`` reduces to ``[s, b, n, 1]``
      without promoting its operand, so the division stays in the activation
      dtype.
    """
    mixed = (h_res.unsqueeze(-1) * streams.unsqueeze(-3)).sum(dim=-2)
    return mixed / torch.linalg.vector_norm(mixed, dim=-1, keepdim=True).clamp(min=eps)


# Recompute the LERP write in backward rather than keeping its intermediates.
#
# The additive write runs at micro-batch 28-32; the LERP write cannot get past 8
# on the same model, and a probe showed why: steady-state occupancy at mbs=8 is
# only 0.62, yet mbs=16 OOMs, so activations dominate and roughly double with
# batch. That batch gap is the whole 65-vs-200 TFLOP/s difference -- the arms are
# memory-bound, not compute-bound, and no amount of kernel fusion touches it
# (fusing the region was measured at 65.9 -> 65.6, i.e. nothing).
#
# The additive path avoids this already: _MHCWrite saves its four inputs and
# recomputes everything in a Triton backward. The LERP path had no equivalent, so
# AOTAutograd saved every [s, b, n, C] intermediate the chain produces -- the
# write-back term, the mixed state before and after renormalization, plus the
# residual module's own internals -- across all 36 sublayers.
#
# Checkpointing gets the same effect without hand-deriving gradients for seven
# lerp_normalization modes and for the residual's learnable alpha, which is a
# parameter and needs its own gradient. use_reentrant=False is the variant that
# handles module parameters correctly; there is no dropout in this region, so the
# recomputation is deterministic.
#
# MEASURED, AND IT FREES NOTHING -- hence off by default. With recompute on,
# occupancy at mbs=8 was 0.647/0.621 against 0.647/0.624 without, and mbs=16
# still OOMs. So the LERP write's intermediates are *not* what fills memory, and
# the reasoning above is wrong somewhere: the additive path's mbs=28-32 does not
# come from _MHCWrite's recompute alone.
#
# Where the memory actually goes is still unmeasured. Back-of-envelope from the
# probe: at mbs=8 activations are ~56 GB, about 2.6x per token what the additive
# arm uses at mbs=32 -- so something in this configuration does retain much more,
# just not here. The other things these arms turn on are ngpt, qk_l2_norm and the
# s_qk/s_uv scalers, none of which have been looked at.
#
# The next step is torch.cuda.memory._record_memory_history on a LERP sublayer,
# not another hypothesis. Three have been tried and measured flat: fusing this
# region (65.9 -> 65.6), trading CUDA graphs for batch (frees 0.02), and this.
#
# Set MHC_LERP_RECOMPUTE=1 to re-enable; the mechanism is verified correct
# (bit-identical gradients including lerp.alpha and mixer.phi), just not useful.
_LERP_RECOMPUTE = os.environ.get("MHC_LERP_RECOMPUTE", "0") != "0"


_COMPILED: dict = {}


def _compiled(fn):
    """Return a compiled ``fn``, falling back to eager if Inductor is unusable.

    Compiled lazily and cached per function, so a runtime without a working
    Triton toolchain still trains, just slower.
    """
    cached = _COMPILED.get(fn)
    if cached is None:
        try:
            cached = torch.compile(fn, dynamic=False)
        except Exception:  # pragma: no cover - depends on the runtime
            cached = fn
        _COMPILED[fn] = cached
    return cached


def _mhc_write_lerp(streams, sublayer_out, h_post, h_res, eps, lerp):
    """The mHC write composed with an nGPT LERP, as one fusible region.

    Args:
        streams: ``[s, b, n, C]`` current stream state.
        sublayer_out: ``[s, b, C]`` attention or MLP output.
        h_post: ``[s, b, n]`` write coefficients, in the stream dtype.
        h_res: ``[s, b, n, n]`` doubly stochastic stream mixer, in the stream dtype.
        eps: Numerical floor for the renormalization.
        lerp: The :class:`~megatron.core.transformer.ngpt.NGPTResidual` to apply.

    Returns:
        ``[s, b, n, C]`` updated streams, each row unit norm.

    Kept as one compiled function rather than three steps so Inductor sees the
    whole chain. Written as separate statements it materializes the write-back
    term and the mixed state as full ``[s, b, n, C]`` tensors before the LERP
    even starts, then walks the result several more times for the LERP's own
    normalize and interpolation -- on the widest tensor mHC owns, at ``n`` times
    the residual width. Dynamo inlines the residual module's forward, so the
    whole thing collapses into a handful of kernels.

    nGPT composition: the doubly stochastic mix is a *convex combination* of the
    streams, so even when every input stream is unit norm the mixed result is
    shorter (equality only when the streams are identical). Renormalize before
    the LERP, otherwise its closed-form correction -- which assumes a unit-norm
    base point -- would not land on the sphere.
    """
    contribution = h_post.unsqueeze(-1) * sublayer_out.unsqueeze(-2)
    mixed = (h_res.unsqueeze(-1) * streams.unsqueeze(-3)).sum(dim=-2)
    mixed = mixed / torch.linalg.vector_norm(mixed, dim=-1, keepdim=True).clamp(min=eps)
    return lerp(contribution, mixed)


def _mhc_read(streams, h_pre):
    """Fused mHC read-out: ``H_pre @ x``, i.e. the C-dim layer input.

    Args:
        streams: ``[s, b, n, C]`` current stream state.
        h_pre: ``[s, b, n]`` read coefficients.

    Returns:
        ``[s, b, C]`` layer input.

    Same reasoning as :func:`_mhc_write`: an ``n``-long contraction is a couple
    of multiply-adds per output element, so expressing it as a fused
    elementwise-plus-reduction lets Inductor read the stream state once and emit
    the result directly, rather than dispatching a separate contraction kernel
    over the widest tensor in the model.
    """
    return (h_pre.unsqueeze(-1) * streams).sum(dim=-2)


_mhc_read_compiled = None


def sinkhorn_knopp(logits: torch.Tensor, num_iters: int = 20, eps: float = 1e-8) -> torch.Tensor:
    """Project a matrix onto the Birkhoff polytope, fusing the Sinkhorn sweeps.

    Thin wrapper over :func:`_sinkhorn_knopp_impl` that compiles it on first use.

    Args:
        logits: Tensor of shape ``[..., n, n]`` holding the unconstrained scores.
        num_iters: Number of alternating normalization sweeps.
        eps: Numerical floor for the divisions.

    Returns:
        A row-stochastic, approximately doubly stochastic tensor, in fp32.
    """
    global _sinkhorn_compiled
    if _sinkhorn_compiled is None:
        try:
            _sinkhorn_compiled = torch.compile(_sinkhorn_knopp_impl, dynamic=False)
        except Exception:  # pragma: no cover - depends on the runtime
            _sinkhorn_compiled = _sinkhorn_knopp_impl
    try:
        return _sinkhorn_compiled(logits, num_iters, eps)
    except Exception:  # pragma: no cover - Inductor/Triton unavailable at runtime
        _sinkhorn_compiled = _sinkhorn_knopp_impl
        return _sinkhorn_knopp_impl(logits, num_iters, eps)


class MHCMixer(nn.Module):
    r"""Computes the mHC connection matrices and applies them to the stream state.

    One instance is owned per sublayer (attention and MLP each get their own),
    matching the paper's treatment of every attention/FFN as an individual layer.

    Args:
        hidden_size: Model hidden dimension ``C``.
        expansion_rate: Number of residual streams ``n``. ``n = 1`` degenerates to
            a standard residual connection.
        num_sinkhorn_iters: Sweeps in the Sinkhorn-Knopp projection.
        alpha_init: Initial value of the learnable gating scalars. Small values
            keep the dynamic (input-dependent) part near zero at init, so the
            static biases determine the initial connectivity.
        res_init: Initial connectivity of the stream mixer. ``"identity"`` biases
            the Sinkhorn input toward the identity so streams start independent
            and the identity mapping is preserved at init; ``"uniform"`` starts
            from the uniform doubly stochastic matrix (all entries ``1/n``).
        identity_bias: Magnitude of the diagonal bias used by ``res_init="identity"``.
        eps: Numerical floor for the RMS normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        expansion_rate: int = 4,
        num_sinkhorn_iters: int = 20,
        alpha_init: float = 0.01,
        res_init: str = "identity",
        identity_bias: float = 5.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if expansion_rate < 1:
            raise ValueError(f"expansion_rate must be >= 1, got {expansion_rate}")
        if res_init not in ("identity", "uniform"):
            raise ValueError(f"res_init must be 'identity' or 'uniform', got {res_init!r}")

        self.hidden_size = hidden_size
        self.n = expansion_rate
        self.num_sinkhorn_iters = num_sinkhorn_iters
        self.eps = eps

        n, flat = self.n, self.n * hidden_size

        # Dynamic (input-dependent) projections of the flattened stream state.
        # Kept as one fused projection so the whole coefficient block is a single
        # matmul, mirroring the paper's fused-kernel design (Eq. 14).
        self.phi = nn.Parameter(torch.zeros(flat, n + n + n * n))
        # Static biases carry the initial connectivity.
        self.b_pre = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.zeros(n, n))
        # Learnable gates on the dynamic part (paper Eq. 7).
        # Shape (1,) not 0-dim: FSDP cannot shard 0-dim parameters, and 0-dim
        # tensors are the ones CUDA graph capture trips over (see
        # NGPTResidual.alpha_scaler).
        self.alpha_pre = nn.Parameter(torch.full((1,), alpha_init))
        self.alpha_post = nn.Parameter(torch.full((1,), alpha_init))
        self.alpha_res = nn.Parameter(torch.full((1,), alpha_init))

        # The stream-coefficient projection is not a hidden-to-hidden linear map.
        self.phi.use_adam_with_emerging_optimizers = True
        for p in (
            self.b_pre,
            self.b_post,
            self.b_res,
            self.alpha_pre,
            self.alpha_post,
            self.alpha_res,
        ):
            # Magnitude / connectivity parameters: never row-normalized by the
            # nGPT weight normalizer, and eligible for the scaler LR group.
            p.is_ngpt_scaler = True

        self.reset_parameters(res_init, identity_bias)

    def reset_parameters(self, res_init: str = "identity", identity_bias: float = 5.0) -> None:
        """Initialize projections and static biases.

        ``phi`` starts at zero so the dynamic contribution vanishes at step 0 and
        the biases fully determine the initial connection matrices.
        """
        with torch.no_grad():
            self.phi.zero_()
            # H_pre = sigmoid(b_pre); the ablation baseline is a uniform 1/n read.
            self.b_pre.fill_(_inv_sigmoid(1.0 / self.n))
            # H_post = 2*sigmoid(b_post); 0 gives 2*0.5 = 1, a plain write-back.
            self.b_post.zero_()
            if res_init == "identity":
                self.b_res.zero_()
                self.b_res.add_(torch.eye(self.n) * identity_bias)
            else:
                self.b_res.zero_()

    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the three connection maps for the current stream state.

        Args:
            streams: Stream tensor of shape ``[s, b, n, C]``.

        Returns:
            ``(h_pre, h_post, h_res)`` with shapes ``[s, b, n]``, ``[s, b, n]`` and
            ``[s, b, n, n]``, all in fp32.
        """
        global _mhc_coeffs_compiled
        if _mhc_coeffs_compiled is None:
            try:
                _mhc_coeffs_compiled = torch.compile(_mhc_coeffs, dynamic=False)
            except Exception:  # pragma: no cover
                _mhc_coeffs_compiled = _mhc_coeffs
        s, b, n, c = streams.shape
        flat = streams.reshape(s, b, n * c)
        try:
            return _mhc_coeffs_compiled(
                flat,
                self.phi,
                self.b_pre,
                self.b_post,
                self.b_res,
                self.alpha_pre,
                self.alpha_post,
                self.alpha_res,
                n,
                self.eps,
                self.num_sinkhorn_iters,
            )
        except Exception:  # pragma: no cover - Inductor unavailable
            pass

        # RMSNorm of the flattened state, with the divide-by-norm applied after
        # the projection (paper Eq. 15-16) so the big tensor is touched once.
        # vector_norm rather than flat.float().pow(2).mean(...): the latter
        # retains an fp32 copy of the whole [s, b, n*C] stream state for backward,
        # which is the largest tensor mHC touches.
        r = torch.linalg.vector_norm(flat, dim=-1, keepdim=True).float() / (flat.shape[-1] ** 0.5)
        r = r.clamp(min=self.eps)
        coeffs = torch.matmul(flat, self.phi.to(flat.dtype)).float() / r

        pre_raw, post_raw, res_raw = torch.split(coeffs, [n, n, n * n], dim=-1)
        h_pre = self.alpha_pre.float() * pre_raw + self.b_pre.float()
        h_post = self.alpha_post.float() * post_raw + self.b_post.float()
        h_res = self.alpha_res.float() * res_raw.view(s, b, n, n) + self.b_res.float()

        # Manifold projections (paper Eq. 8). Sigmoid keeps the read/write maps
        # non-negative, preventing signal cancellation between streams.
        return (
            torch.sigmoid(h_pre),
            2.0 * torch.sigmoid(h_post),
            sinkhorn_knopp(h_res, self.num_sinkhorn_iters),
        )

    @torch.no_grad()
    def record_metrics(self, streams: torch.Tensor, coeffs) -> None:
        """Stream usage of this sublayer (arch_metrics): how the read spreads over
        the streams, the write strength, and how much the mixer keeps the streams
        apart (diagonal mass 1) or mixes them (1/n)."""
        from megatron.core.transformer import arch_metrics

        tag = arch_metrics.tag_of(self)
        if tag is None:
            return
        kind, layer, num_layers = tag
        h_pre, h_post, h_res = coeffs
        p = f"mhc_{kind}"
        read = h_pre.float()
        arch_metrics.record_share(
            f"{p}/read_share", layer, num_layers, read / read.sum(-1, keepdim=True).clamp(min=1e-12)
        )
        arch_metrics.record_mean(f"{p}/write", layer, num_layers, h_post)
        arch_metrics.record_mean(
            f"{p}/res_diag", layer, num_layers, torch.diagonal(h_res, dim1=-2, dim2=-1)
        )

    def read(self, streams: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        """Aggregate the ``n`` streams into a single ``C``-dim layer input."""
        global _mhc_read_compiled
        if _mhc_read_compiled is None:
            try:
                _mhc_read_compiled = torch.compile(_mhc_read, dynamic=False)
            except Exception:  # pragma: no cover
                _mhc_read_compiled = _mhc_read
        try:
            return _mhc_read_compiled(streams, h_pre.to(streams.dtype))
        except Exception:  # pragma: no cover - Inductor unavailable
            _mhc_read_compiled = _mhc_read
            return _mhc_read(streams, h_pre.to(streams.dtype))

    def write(
        self,
        streams: torch.Tensor,
        sublayer_out: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
        lerp: nn.Module | None = None,
    ) -> torch.Tensor:
        """Mix the streams and scatter the sublayer output back onto them.

        Implements paper Eq. 3. When ``lerp`` is given, the norm-preserving stream
        mix is applied **first** and each stream then takes an nGPT LERP step
        toward the normalized sublayer output, with its step size scaled by that
        stream's write coefficient.

        Args:
            streams: ``[s, b, n, C]`` incoming stream state.
            sublayer_out: ``[s, b, C]`` output of the attention or MLP sublayer.
            h_post: ``[s, b, n]`` write coefficients.
            h_res: ``[s, b, n, n]`` doubly stochastic stream mixer.
            lerp: Optional nGPT residual module used in place of the additive
                write-back.

        Returns:
            The updated stream state, ``[s, b, n, C]``.
        """
        s, b, n, c = streams.shape
        if lerp is None:
            from megatron.core.transformer.mhc_kernels import can_use

            post, res = h_post.to(streams.dtype), h_res.to(streams.dtype)
            if can_use(streams, n):
                # Fused Triton backward: one pass over the upstream gradient
                # instead of the four Inductor derives from the forward.
                return _MHCWrite.apply(streams, sublayer_out, post, res)
            return _mhc_write_dispatch(streams, sublayer_out, post, res)
        post, res = h_post.to(streams.dtype), h_res.to(streams.dtype)
        try:
            fn = _compiled(_mhc_write_lerp)
            if _LERP_RECOMPUTE:
                # Trade one recomputation of this region for the intermediates it
                # would otherwise keep alive; see the note on _LERP_RECOMPUTE.
                return checkpoint(fn, streams, sublayer_out, post, res, self.eps, lerp,
                                  use_reentrant=False)
            return fn(streams, sublayer_out, post, res, self.eps, lerp)
        except Exception:  # pragma: no cover - Inductor unusable at runtime
            # torch.compile defers compilation to the first call, so a broken
            # toolchain surfaces here rather than where the wrapper was built.
            return _mhc_write_lerp(streams, sublayer_out, post, res, self.eps, lerp)


def _inv_sigmoid(p: float) -> float:
    """Return the logit of ``p`` (inverse of the sigmoid)."""
    import math

    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def expand_to_streams(hidden_states: torch.Tensor, expansion_rate: int) -> torch.Tensor:
    """Broadcast a ``[s, b, C]`` hidden state into ``[s, b, n, C]`` streams.

    Used once at the start of the transformer block. Every stream starts as a copy
    of the embedding, so at initialization -- with an identity stream mixer -- the
    network behaves like ``n`` independent copies of the standard residual path.
    """
    return hidden_states.unsqueeze(-2).expand(-1, -1, expansion_rate, -1).contiguous()


def collapse_streams(streams: torch.Tensor) -> torch.Tensor:
    """Reduce ``[s, b, n, C]`` streams back to a single ``[s, b, C]`` hidden state.

    The paper's read map is a convex combination, so the mean is the natural
    magnitude-preserving collapse at the end of the trunk.
    """
    return streams.mean(dim=-2)
