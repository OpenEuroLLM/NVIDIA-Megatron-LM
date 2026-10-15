# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Muown: row-norm control for Muon.

Implements the optimizer of *Muown: Row-Norm Control for Muon Optimization*
(Lion, Hübler, Li, Orvieto & He, `arXiv:2605.10797
<https://arxiv.org/abs/2605.10797>`_).

Motivation
----------
Muon orthogonalizes the momentum before applying it, which controls the
*coherence* of a weight matrix but not its scale: the spectral norm of the
weights drifts upward over training. The paper observes that this drift lives
entirely in the **row-magnitude** factor of the decomposition

.. math::
    W_i = m_i \\cdot D_i, \\qquad m_i = \\lVert W_i \\rVert_2, \\quad
    \\lVert D_i \\rVert_2 = 1

while the **row-direction** factor :math:`D` stays well-behaved. Muown therefore
promotes :math:`m` to an explicit optimizer variable updated under the
:math:`\\ell_\\infty` geometry induced by the decomposition (whose steepest-descent
step is a sign step), and applies Muon unchanged to :math:`D`.

Contrast with nGPT
------------------
This is the optimizer-side answer to the same problem nGPT solves
architecturally. nGPT *pins* every row to unit norm (``m ≡ 1``) and reinjects
scale through learnable activation scalers; Muown keeps ``m`` as a real degree of
freedom and simply optimizes it in the right geometry. The two can be compared
directly by toggling the weight normalizer against this optimizer.
"""

import logging
from typing import Any, Callable, Iterable

import torch

from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

# Quintic Newton-Schulz coefficients (Jordan's Muon defaults): the iteration
# X <- a*X + b*(X X^T)X + c*(X X^T)^2 X pushes singular values toward 1.
_NS_QUINTIC = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz(
    G: torch.Tensor, steps: int = 5, eps: float = 1e-7, coefficients: tuple = _NS_QUINTIC
) -> torch.Tensor:
    """Approximately orthogonalize a matrix via Newton-Schulz iteration.

    Runs in bfloat16 (as in the reference Muon implementation) and always
    iterates on the shorter side so the Gram matrix stays small.

    Args:
        G: Matrix of shape ``[m, n]`` to orthogonalize.
        steps: Number of Newton-Schulz iterations.
        eps: Floor for the initial spectral normalization.
        coefficients: The ``(a, b, c)`` quintic coefficients.

    Returns:
        A matrix with approximately unit singular values, same shape as ``G``.
    """
    if G.ndim != 2:
        raise ValueError(f"newton_schulz expects a 2-D matrix, got shape {tuple(G.shape)}")
    a, b, c = coefficients
    X = G.bfloat16()
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    X = X / X.norm().clamp(min=eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muown(torch.optim.Optimizer):
    """Muon with explicit row-magnitude control.

    Each 2-D parameter is decomposed into a row-magnitude vector ``m`` and a
    unit-row direction matrix ``D``. Per step:

    1. Split the gradient into a radial part (``<G_i, D_i>``, which moves ``m``)
       and a tangential part (which rotates ``D``).
    2. Accumulate momentum separately for both parts.
    3. Orthogonalize the tangential momentum with Newton-Schulz (Muon) and take a
       spectral-scaled step on ``D``, then renormalize its rows.
    4. Take an :math:`\\ell_\\infty` steepest-descent (sign) step on ``m``, with
       optional decoupled weight decay, and clamp it to ``max_row_norm``.
    5. Recompose ``W = m * D``.

    Parameters with fewer than 2 dimensions are not handled here -- route them to
    Adam via parameter groups (see the registry entry in
    ``emerging_optimizers.py``).

    Args:
        params: Iterable of parameters or parameter-group dicts.
        lr: Learning rate for the direction (Muon) update.
        magnitude_lr_scale: Multiplier on ``lr`` for the row-magnitude update.
        momentum: Momentum coefficient for both components.
        nesterov: Use Nesterov-style lookahead momentum.
        weight_decay: Decoupled weight decay applied to the row magnitudes.
        num_ns_steps: Newton-Schulz iterations.
        scale_mode: ``"spectral"`` scales the update by ``sqrt(max(out, in))``
            (Muon/Moonlight convention); ``"unit"`` applies no extra scaling.
        extra_scale_factor: Additional multiplier on the direction update.
        max_row_norm: Upper clamp for row magnitudes; None disables clamping.
        eps: Numerical floor.
        orthogonalize_fn: Optional override for the orthogonalization routine,
            used to plug in a tensor-parallel-aware Newton-Schulz.
        row_sum_fn: ``(partial_row_sums, param) -> row_sums``. Under tensor
            parallelism a column-sharded weight (``partition_dim == 1``) holds a
            column slice of every row, so each per-row reduction (row norm, radial
            component, renormalization) is a partial sum to be all-reduced across
            the TP group. None (single process) uses the local sums.
        tp_shape_fn: ``param -> (rows, cols)`` of the unsharded matrix, for the
            spectral scale. None uses the local shape.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-4,
        magnitude_lr_scale: float = 1.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        max_row_norm: float | None = None,
        eps: float = 1e-8,
        orthogonalize_fn: Callable[..., torch.Tensor] | None = None,
        row_sum_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        tp_shape_fn: Callable[[torch.Tensor], tuple] | None = None,
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        if scale_mode not in ("spectral", "unit"):
            raise ValueError(f"scale_mode must be 'spectral' or 'unit', got {scale_mode!r}")
        defaults = dict(
            lr=lr,
            magnitude_lr_scale=magnitude_lr_scale,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            max_row_norm=max_row_norm,
            eps=eps,
        )
        super().__init__(params, defaults)
        self._orthogonalize_fn = orthogonalize_fn or (
            lambda g, steps=5, **_: newton_schulz(g, steps=steps)
        )
        self._row_sum = row_sum_fn or (lambda t, _p: t)
        self._tp_shape = tp_shape_fn
        log_single_rank(logger, logging.INFO, "[Muown] initialized row-norm-controlled Muon")

    def _orthogonalize(self, grad: torch.Tensor, param: torch.Tensor, steps: int) -> torch.Tensor:
        """Orthogonalize ``grad``, flattening any leading (expert) dimensions."""
        if grad.ndim == 2:
            return self._orthogonalize_fn(grad, steps=steps, param=param)
        # Grouped MoE experts: orthogonalize each expert matrix independently.
        out = torch.empty_like(grad)
        for idx in range(grad.shape[0]):
            out[idx] = self._orthogonalize_fn(grad[idx], steps=steps, param=param)
        return out

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Callable | None = None) -> Any:
        """Perform a single Muown optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mag_lr = lr * group["magnitude_lr_scale"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            steps = group["num_ns_steps"]
            eps = group["eps"]
            max_row_norm = group["max_row_norm"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim < 2:
                    raise ValueError(
                        f"Muown received a {p.ndim}-D parameter of shape {tuple(p.shape)}; "
                        "route non-matrix parameters to Adam via parameter groups."
                    )

                grad = p.grad.float()
                w = p.data.float()

                # --- decompose W into row magnitude m and unit-row direction D ---
                # Row reductions go through _row_sum: partial under column sharding.
                m = self._row_sum(w.square().sum(dim=-1, keepdim=True), p).sqrt().clamp(min=eps)
                d = w / m

                # Radial component moves the magnitude; the remainder rotates D.
                radial = self._row_sum((grad * d).sum(dim=-1, keepdim=True), p)
                tangential = grad - radial * d

                state = self.state[p]
                if len(state) == 0:
                    state["direction_momentum"] = torch.zeros_like(tangential)
                    # Allocated at the *parameter* shape, not the natural [out, 1]
                    # of a per-row magnitude. Megatron's optimizer-state handling
                    # (checkpoint save, distributed optimizer sharding) requires
                    # every state tensor to match the parameter shape; a [out, 1]
                    # buffer makes every save fail with "Optimizer shape ... does
                    # not match model shape". Only column 0 is used.
                    state["magnitude_momentum"] = torch.zeros_like(p, dtype=torch.float32)

                dir_buf = state["direction_momentum"]
                mag_buf = state["magnitude_momentum"][..., :1]
                dir_buf.mul_(momentum).add_(tangential, alpha=1.0 - momentum)
                mag_buf.mul_(momentum).add_(radial, alpha=1.0 - momentum)

                if nesterov:
                    dir_update = tangential.mul(1.0 - momentum).add_(dir_buf, alpha=momentum)
                    mag_update = radial.mul(1.0 - momentum).add_(mag_buf, alpha=momentum)
                else:
                    dir_update = dir_buf
                    mag_update = mag_buf

                # --- direction: Muon (orthogonalized) step, then renormalize ---
                ortho = self._orthogonalize(dir_update, p, steps)
                if group["scale_mode"] == "spectral":
                    rows, cols = self._tp_shape(p) if self._tp_shape else ortho.shape[-2:]
                    scale = max(rows, cols) ** 0.5
                else:
                    scale = 1.0
                scale *= group["extra_scale_factor"]
                d = d - lr * scale * ortho
                d = d / self._row_sum(d.square().sum(dim=-1, keepdim=True), p).sqrt().clamp(min=eps)

                # --- magnitude: l-infinity steepest descent == sign step ---
                m = m - mag_lr * torch.sign(mag_update)
                if wd != 0.0:
                    m = m - mag_lr * wd * m
                m = m.clamp(min=eps)
                if max_row_norm is not None:
                    m = m.clamp(max=max_row_norm)

                p.data.copy_((m * d).to(p.dtype))

        return loss
