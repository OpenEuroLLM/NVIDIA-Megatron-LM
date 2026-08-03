# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""nGPT geometry primitives: hypersphere normalization and LERP residual updates.

This module implements the building blocks of the *normalized transformer*
(nGPT, `arXiv:2410.01131 <https://arxiv.org/abs/2410.01131>`_), where token
representations live on the unit hypersphere :math:`S^{d-1}` and each sublayer
applies a normalized displacement instead of an additive residual.

Two ingredients are provided:

* :class:`EigenLRScaler` -- a learnable scaling vector stored with the muP-style
  "eigen learning rate" reparameterization. The raw parameter is initialized to
  ``base_scale`` and multiplied by a fixed (non-trainable) ``scaler`` at forward
  time, so the *effective* initial value and the *effective* per-step update size
  can be chosen independently.
* :class:`NGPTResidual` -- the LERP-on-the-sphere residual update replacing the
  usual ``x + sublayer(x)``.

All features are opt-in; when nGPT is disabled these modules are never
constructed and the model is bit-identical to upstream Megatron.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

# Modes accepted by ``NGPTResidual``. See the class docstring for the math.
LERP_NORMALIZATION_MODES = (
    "approx",
    "analytic",
    "direct",
    "both",
    "direct_ste",
    "both_ste",
    "approx_fwd",
)

ALPHA_ACTIVATIONS = ("abs", "half_sigmoid", "half_tanh_sq", "tanh_smooth_bound")


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize along the last (hidden) dimension in fp32.

    Args:
        x: Tensor whose last dimension is the model hidden dimension.
        eps: Numerical floor passed to :func:`torch.nn.functional.normalize`.

    Returns:
        A tensor of the same shape and dtype as ``x`` with unit-norm rows.
    """
    return F.normalize(x.float(), p=2, dim=-1, eps=eps).type_as(x)


def resolve_scale_factor(mode: str, dim: int, *, name: str) -> float:
    """Resolve a muP-style chain-rule reparameterization factor.

    Args:
        mode: One of ``"one"``, ``"sqrt_dim"`` or ``"inv_sqrt_dim"``.
        dim: Model hidden dimension used by the dimension-dependent modes.
        name: Field name, used only to build a helpful error message.

    Returns:
        The multiplicative factor implied by ``mode``.

    Raises:
        ValueError: If ``mode`` is not a recognized value.
    """
    if mode == "one":
        return 1.0
    if mode == "sqrt_dim":
        return math.sqrt(dim)
    if mode == "inv_sqrt_dim":
        return 1.0 / math.sqrt(dim)
    raise ValueError(f"unknown {name}={mode!r} (expected 'one', 'sqrt_dim' or 'inv_sqrt_dim')")


class EigenLRScaler(nn.Module):
    """Learnable per-channel scaler with the nGPT eigen-learning-rate reparameterization.

    The effective scale applied at forward time is ``weight * scaler``, where
    ``weight`` is trainable and ``scaler`` is a fixed buffer. Initializing
    ``weight = base_scale`` and ``scaler = init_value / base_scale`` makes the
    effective value equal ``init_value`` at step 0 while the optimizer sees a
    parameter whose natural scale is ``base_scale`` -- decoupling the initial
    value from the effective learning rate.

    Args:
        size: Number of channels (last-dimension width) of the scaler.
        init_value: Effective value of the scaler at initialization.
        base_scale: Value the raw trainable parameter is initialized to.
        dtype: Parameter dtype; kept in fp32 for numerical stability.
    """

    def __init__(
        self,
        size: int,
        init_value: float = 1.0,
        base_scale: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if base_scale <= 0.0:
            raise ValueError(f"base_scale must be positive, got {base_scale}")
        self.weight = nn.Parameter(torch.full((size,), base_scale, dtype=dtype))
        # Python float rather than a 0-dim buffer -- see NGPTResidual.alpha_scaler.
        self.scaler = float(init_value / base_scale)
        # Consumed by the weight normalizer / optimizer grouping: these are 1-D
        # magnitude parameters and must never be row-normalized.
        self.weight.is_ngpt_scaler = True

    def effective(self) -> torch.Tensor:
        """Return the effective scale vector ``weight * scaler`` in fp32."""
        return self.weight.float() * self.scaler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the scaler to the last dimension of ``x``."""
        return x * self.effective().to(x.dtype)


class NGPTResidual(nn.Module):
    r"""LERP residual update on the unit hypersphere.

    Replaces the additive residual ``x + f(x)`` with a spherical linear
    interpolation toward the (optionally L2-normalized) sublayer output:

    .. math::
        x' = \mathrm{Norm}\big(x + \alpha \cdot (\hat{a} - x)\big)
           = \mathrm{Norm}\big((1-\alpha)\,x + \alpha\,\hat{a}\big)

    where :math:`\hat{a}` is the normalized sublayer output and :math:`\alpha` is
    a learnable per-dimension step size (the nGPT *eigen learning rate*).

    How :math:`\mathrm{Norm}` is realized is selected by ``lerp_normalization``:

    ``approx``
        nGPT's closed form under the assumption that the update is orthogonal to
        ``x`` (:math:`\cos\theta \approx 0`), giving the scalar correction
        :math:`f = 1/\sqrt{1 - 2\alpha(1-\alpha)}`. Cheap; the residual norm
        drifts upward slowly.
    ``analytic``
        The exact factor for unit-norm ``x`` and :math:`\hat{a}`:
        :math:`f = 1/\sqrt{(1-\alpha)^2 + 2\alpha(1-\alpha)\cos\theta + \alpha^2}`
        with :math:`\cos\theta = x^\top \hat{a}`. Keeps a full-rank Jacobian
        (unlike an explicit renormalization, whose Jacobian is a rank-deficient
        tangent projector) while landing exactly on the sphere.
    ``direct``
        Explicit :func:`torch.nn.functional.normalize` after the interpolation.
    ``both``
        ``approx`` correction followed by an explicit normalization.
    ``direct_ste`` / ``both_ste`` / ``approx_fwd``
        Straight-through variants that decouple the forward normalization from
        the backward Jacobian; useful to ablate whether the benefit comes from
        the forward residual-norm growth or from the scalar-vs-projection
        backward.

    Args:
        hidden_size: Model hidden dimension (per-dimension ``alpha`` width).
        alpha_init_value: Effective value of ``alpha`` at initialization.
        base_scale: Raw-parameter init for the eigen-LR reparameterization.
            ``-1`` selects ``1/sqrt(hidden_size)``.
        alpha_activation: Mapping from the raw parameter to the effective
            ``alpha``. ``abs`` is unbounded; the others cap ``alpha``.
        lerp_normalization: One of :data:`LERP_NORMALIZATION_MODES`.
        post_layer_norm: If True, L2-normalize the sublayer output before the
            interpolation (this is what puts ``a`` on the sphere).
        eps: Numerical floor for normalization and the ``rsqrt`` clamp.
    """

    def __init__(
        self,
        hidden_size: int,
        alpha_init_value: float = 0.05,
        base_scale: float = -1.0,
        alpha_activation: str = "abs",
        lerp_normalization: str = "approx",
        post_layer_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if lerp_normalization not in LERP_NORMALIZATION_MODES:
            raise ValueError(
                f"lerp_normalization must be one of {LERP_NORMALIZATION_MODES}, "
                f"got {lerp_normalization!r}"
            )
        if alpha_activation not in ALPHA_ACTIVATIONS:
            raise ValueError(
                f"alpha_activation must be one of {ALPHA_ACTIVATIONS}, got {alpha_activation!r}"
            )

        self.hidden_size = hidden_size
        self.alpha_init_value = alpha_init_value
        self.alpha_activation = alpha_activation
        self.lerp_normalization = lerp_normalization
        self.post_layer_norm = post_layer_norm
        self.eps = eps
        self.base_scale = 1.0 / math.sqrt(hidden_size) if base_scale == -1.0 else base_scale

        # Pre-activation target t such that activation(t) == alpha_init_value, so
        # the effective alpha at init is independent of the activation choice.
        target = self._alpha_preact_init_target()
        self.alpha = nn.Parameter(torch.full((hidden_size,), self.base_scale, dtype=torch.float32))
        # Plain Python float, NOT a 0-dim buffer. A scalar tensor here is a real
        # liability: CUDA graph capture version-bumps it and autograd then raises
        # "a variable needed for gradient computation has been modified by an
        # inplace operation", which disabled graphs for every LERP-based arm. As a
        # float it disappears from the graph and the multiply folds into the
        # following elementwise op instead of costing its own kernel.
        self.alpha_scaler = float(target / self.base_scale)
        self.alpha.is_ngpt_scaler = True

    def _alpha_preact_init_target(self) -> float:
        """Return the pre-activation value mapping to ``alpha_init_value``."""
        a = self.alpha_init_value
        act = self.alpha_activation
        if act == "abs":
            return a
        if act == "half_sigmoid":
            if not 0.0 < a < 0.5:
                raise ValueError(f"half_sigmoid needs 0 < alpha_init < 0.5, got {a}")
            return math.log((2.0 * a) / (1.0 - 2.0 * a))
        if act == "half_tanh_sq":
            if not 0.0 < a < 0.5:
                raise ValueError(f"half_tanh_sq needs 0 < alpha_init < 0.5, got {a}")
            return math.atanh(math.sqrt(2.0 * a))
        # tanh_smooth_bound
        if not 0.0 < a < 0.95:
            raise ValueError(f"tanh_smooth_bound needs 0 < alpha_init < 0.95, got {a}")
        return math.sqrt(math.atanh(a) / math.atanh(0.95))

    def _alpha_activation_fn(self, raw: torch.Tensor) -> torch.Tensor:
        """Map the raw pre-activation parameter to the effective LERP coefficient."""
        act = self.alpha_activation
        if act == "abs":
            return torch.abs(raw)
        if act == "half_sigmoid":
            return 0.5 * torch.sigmoid(raw)
        if act == "half_tanh_sq":
            return 0.5 * torch.tanh(raw).pow(2)
        return torch.tanh(math.atanh(0.95) * raw.pow(2))

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply the LERP residual update.

        Args:
            hidden_states: Sublayer output ``a`` (attention or MLP), shape
                ``[seq, batch, hidden]``.
            residual: Incoming residual-stream value ``x``, same shape.

        Returns:
            The updated residual stream, in the dtype of ``residual``.
        """
        input_dtype = residual.dtype
        alpha = self._alpha_activation_fn(self.alpha.float() * self.alpha_scaler)

        if self.post_layer_norm:
            a = F.normalize(hidden_states.float(), p=2, dim=-1, eps=self.eps)
        else:
            a = hidden_states.float()

        x = residual.float()
        mode = self.lerp_normalization

        if mode == "analytic":
            # Exact factor for unit-norm x and a; full-rank Jacobian.
            cos_theta = (x * a).sum(dim=-1, keepdim=True)
            one_minus = 1.0 - alpha
            denom = one_minus.pow(2) + 2.0 * alpha * one_minus * cos_theta + alpha.pow(2)
            out = (x + alpha * (a - x)) * torch.rsqrt(torch.clamp(denom, min=self.eps))
        elif mode == "approx":
            correct = torch.rsqrt(torch.clamp(1.0 - 2.0 * alpha * (1.0 - alpha), min=self.eps))
            out = (x + alpha * (a - x)) * correct
        elif mode == "direct":
            out = F.normalize(x + alpha * (a - x), p=2, dim=-1, eps=self.eps)
        elif mode == "both":
            correct = torch.rsqrt(torch.clamp(1.0 - 2.0 * alpha * (1.0 - alpha), min=self.eps))
            out = F.normalize((x + alpha * (a - x)) * correct, p=2, dim=-1, eps=self.eps)
        elif mode == "direct_ste":
            # Forward: exact sphere. Backward: identity through h.
            h = x + alpha * (a - x)
            out = h + (F.normalize(h, p=2, dim=-1, eps=self.eps) - h).detach()
        elif mode == "both_ste":
            # Forward: exact sphere. Backward: approx scalar Jacobian.
            h = x + alpha * (a - x)
            correct = torch.rsqrt(torch.clamp(1.0 - 2.0 * alpha * (1.0 - alpha), min=self.eps))
            h_approx = h * correct
            out = h_approx + (F.normalize(h_approx, p=2, dim=-1, eps=self.eps) - h_approx).detach()
        else:  # approx_fwd
            # Forward: growing norm (approx). Backward: projection Jacobian.
            h = x + alpha * (a - x)
            correct = torch.rsqrt(torch.clamp(1.0 - 2.0 * alpha * (1.0 - alpha), min=self.eps))
            h_approx = h * correct
            h_direct = F.normalize(h_approx, p=2, dim=-1, eps=self.eps)
            out = h_direct + (h_approx - h_direct).detach()

        return out.to(input_dtype)
