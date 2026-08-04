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

import torch
import torch.nn.functional as F
from torch import nn


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

    def read(self, streams: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        """Aggregate the ``n`` streams into a single ``C``-dim layer input."""
        return torch.einsum("sbn,sbnc->sbc", h_pre.to(streams.dtype), streams)

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
        contribution = h_post.to(streams.dtype).unsqueeze(-1) * sublayer_out.unsqueeze(-2)
        if lerp is None:
            # Fuse the stream mix and the write-back into one batched GEMM:
            # baddbmm computes contribution + h_res @ streams in a single pass,
            # avoiding a separate full [s, b, n, C] `mixed` tensor and the extra
            # read/write it costs. mHC is bandwidth-bound in exactly this term
            # (paper Table 2: I/O grows with n), so removing one traversal of the
            # widened residual is the cheapest available win.
            return torch.baddbmm(
                contribution.reshape(s * b, n, c),
                h_res.to(streams.dtype).reshape(s * b, n, n),
                streams.reshape(s * b, n, c),
            ).view(s, b, n, c)
        mixed = torch.einsum("sbij,sbjc->sbic", h_res.to(streams.dtype), streams)
        # nGPT composition: the doubly stochastic mix is a *convex combination* of
        # the streams, so even when every input stream is unit norm the mixed
        # result is shorter (equality only when the streams are identical).
        # Renormalize before the LERP, otherwise the LERP's closed-form correction
        # -- which assumes a unit-norm base point -- would not land on the sphere.
        mixed = F.normalize(mixed.float(), p=2, dim=-1, eps=1e-6).to(streams.dtype)
        return lerp(contribution, mixed)


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
