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
    """Approximately orthogonalize a matrix (or a stack of matrices) via Newton-Schulz.

    Runs in bfloat16 (as in the reference Muon implementation) and always
    iterates on the shorter side so the Gram matrix stays small.

    Args:
        G: Matrix of shape ``[m, n]``, or a stack ``[..., m, n]`` orthogonalized
            matrix by matrix.
        steps: Number of Newton-Schulz iterations.
        eps: Floor for the initial spectral normalization.
        coefficients: The ``(a, b, c)`` quintic coefficients.

    Returns:
        Matrices with approximately unit singular values, same shape as ``G``.
    """
    if G.ndim < 2:
        raise ValueError(f"newton_schulz expects a matrix, got shape {tuple(G.shape)}")
    a, b, c = coefficients
    X = G.bfloat16()
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
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
        batched_orthogonalize_fn: ``(stack [B, m, n], steps, param) -> stack``, the
            orthogonalization of a stack of same-shape matrices that share ``param``'s
            sharding. None uses the local batched Newton-Schulz when no
            ``orthogonalize_fn`` is given and disables batching otherwise.
        bucket_key_fn: ``param -> hashable`` grouping parameters that can be updated
            as one stack (same shape, dtype and sharding); None in a key opts that
            parameter out. Batching turns the per-matrix kernels of e.g. a
            grouped-GEMM MoE layer (one parameter per expert) into a few large
            batched ones; the math per matrix is unchanged.
        cpu_offload: Keep the momentum buffers in pinned host memory instead of on the
            GPU (8 bytes per matrix parameter). The step streams them: while one
            parameter (or stack of same-shape parameters) is orthogonalized, the next
            one's state is copied in on a side stream and the previous one's copied back, so the transfers hide behind
            the Newton-Schulz matmuls. The state keeps its shapes (checkpoints are
            unchanged); state a checkpoint load put on the GPU moves back on the next
            step.
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
        batched_orthogonalize_fn: Callable[..., torch.Tensor] | None = None,
        bucket_key_fn: Callable[[torch.Tensor], Any] | None = None,
        cpu_offload: bool = False,
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
        local_ns = lambda g, steps=5, **_: newton_schulz(g, steps=steps)
        if batched_orthogonalize_fn is None and orthogonalize_fn is None:
            batched_orthogonalize_fn = local_ns
        self._orthogonalize_fn = orthogonalize_fn or local_ns
        self._batched_orthogonalize_fn = batched_orthogonalize_fn
        self._bucket_key = bucket_key_fn or (lambda p: (tuple(p.shape), p.dtype, p.device))
        self._row_sum = row_sum_fn or (lambda t, _p: t)
        self._tp_shape = tp_shape_fn
        self.cpu_offload = cpu_offload
        self._copy_stream = None
        # Pinned host buffers per update unit (see _host_unit); the staging stack for the
        # used magnitude column is kept out of self.state so checkpoints only ever see
        # parameter-shaped state.
        self._host_units: dict = {}
        log_single_rank(logger, logging.INFO, "[Muown] initialized row-norm-controlled Muon")

    @torch.no_grad()
    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        """Allocate the momentum buffers (checkpoint loading needs them before a step)."""
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["direction_momentum"] = self._zeros(p)
                state["magnitude_momentum"] = self._zeros(p)

    def _zeros(self, p: torch.Tensor) -> torch.Tensor:
        """A zero fp32 state tensor of ``p``'s shape: on ``p``'s device, or pinned host
        memory under cpu_offload."""
        if self.cpu_offload:
            return torch.zeros(p.shape, dtype=torch.float32, pin_memory=True)
        return torch.zeros_like(p, dtype=torch.float32)

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

        work = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim < 2:
                    raise ValueError(
                        f"Muown received a {p.ndim}-D parameter of shape {tuple(p.shape)}; "
                        "route non-matrix parameters to Adam via parameter groups."
                    )
                work.append((group, p))
        units = self._units(work)
        if self.cpu_offload and units:
            self._step_offloaded(units)
        else:
            for group, ps in units:
                for p in ps:
                    state = self.state[p]
                    if len(state) == 0:
                        # Allocated at the *parameter* shape, not the natural [out, 1]
                        # of a per-row magnitude. Megatron's optimizer-state handling
                        # (checkpoint save, distributed optimizer sharding) requires
                        # every state tensor to match the parameter shape; a [out, 1]
                        # buffer makes every save fail with "Optimizer shape ... does
                        # not match model shape". Only column 0 is used.
                        state["direction_momentum"] = torch.zeros_like(p, dtype=torch.float32)
                        state["magnitude_momentum"] = torch.zeros_like(p, dtype=torch.float32)
                if len(ps) == 1:
                    state = self.state[ps[0]]
                    self._update(
                        group,
                        ps[0],
                        state["direction_momentum"],
                        state["magnitude_momentum"][..., :1],
                    )
                    continue
                dir_bufs = [self.state[p]["direction_momentum"] for p in ps]
                mag_bufs = [self.state[p]["magnitude_momentum"][..., :1] for p in ps]
                dir_stack, mag_stack = torch.stack(dir_bufs), torch.stack(mag_bufs)
                self._update_stack(group, ps, dir_stack, mag_stack)
                torch._foreach_copy_(dir_bufs, list(dir_stack.unbind(0)))
                torch._foreach_copy_(mag_bufs, list(mag_stack.unbind(0)))
        return loss

    def _units(self, work: list) -> list:
        """``work`` as ``(group, params)`` update units: same-key parameters (see
        ``bucket_key_fn``) of a group stacked, in runs below ``_STACK_NUMEL``; every
        other parameter alone."""
        buckets: dict = {}
        for group, p in work:
            key = self._bucket_key(p) if self._batched_orthogonalize_fn is not None else None
            if key is None:
                buckets[("single", id(p))] = (group, [p])
            else:
                buckets.setdefault((id(group), key), (group, []))[1].append(p)
        return [(group, chunk) for group, ps in buckets.values() for chunk in self._chunks(ps)]

    def _host_unit(self, ps: list) -> tuple:
        """Pinned host momentum of the unit ``ps``: one contiguous ``[n, *shape]`` buffer
        each for the direction and the magnitude momentum (``self.state`` holds views of
        its rows, so checkpoints keep parameter-shaped state) and a contiguous staging
        stack for the used magnitude column. State that is not such a view -- fresh, or
        put in place (on the GPU) by a checkpoint load -- is copied in."""
        key = tuple(id(p) for p in ps)
        bufs = self._host_units.get(key)
        if bufs is None:
            shape = (len(ps), *ps[0].shape)
            bufs = (
                torch.zeros(shape, dtype=torch.float32, pin_memory=True),
                torch.zeros(shape, dtype=torch.float32, pin_memory=True),
                torch.empty((*shape[:-1], 1), dtype=torch.float32, pin_memory=True),
            )
            self._host_units[key] = bufs
        dir_host, mag_host, _ = bufs
        for i, p in enumerate(ps):
            state = self.state[p]
            for name, host in (("direction_momentum", dir_host), ("magnitude_momentum", mag_host)):
                buf = state.get(name)
                if buf is None or buf.data_ptr() != host[i].data_ptr():
                    if buf is not None:
                        host[i].copy_(buf)
                    state[name] = host[i]
        return bufs

    def _step_offloaded(self, units: list) -> None:
        """The step over ``units`` with the state streamed from / to host memory: the
        prefetch of unit i + 1 and the write-back of unit i - 1 run on a side stream
        while unit i is updated on the current stream; a unit moves in one copy each
        way."""
        compute = torch.cuda.current_stream()
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()
        copy = self._copy_stream

        def fetch(ps):
            dir_host, mag_host, staging = self._host_unit(ps)
            staging.copy_(mag_host[..., :1])
            with torch.cuda.stream(copy):
                d = dir_host.to(ps[0].device, non_blocking=True)
                m = staging.to(ps[0].device, non_blocking=True)
            ready = copy.record_event()
            return d, m, ready

        pending = []  # (magnitude host buffer, its staging stack) to fold back after the sync
        nxt = fetch(units[0][1])
        for i, (group, ps) in enumerate(units):
            d, m, ready = nxt
            if i + 1 < len(units):
                nxt = fetch(units[i + 1][1])
            compute.wait_event(ready)
            if len(ps) == 1:
                self._update(group, ps[0], d[0], m[0])
            else:
                self._update_stack(group, ps, d, m)
            updated = compute.record_event()
            dir_host, mag_host, staging = self._host_unit(ps)
            with torch.cuda.stream(copy):
                copy.wait_event(updated)
                dir_host.copy_(d, non_blocking=True)
                staging.copy_(m, non_blocking=True)
            # allocated on the copy stream, used on the compute stream
            d.record_stream(compute)
            m.record_stream(compute)
            pending.append((mag_host, staging))
        copy.synchronize()
        for mag_host, staging in pending:
            mag_host[..., :1].copy_(staging)

    # Elements per stacked bucket: bounds the transient fp32 copies of a stacked step.
    _STACK_NUMEL = 1 << 26

    def _chunks(self, ps: list) -> list:
        """``ps`` split into runs whose stacked size stays below ``_STACK_NUMEL``."""
        per = max(1, self._STACK_NUMEL // max(1, ps[0].numel()))
        return [ps[i : i + per] for i in range(0, len(ps), per)]

    def _update_stack(
        self, group: dict, ps: list, dir_stack: torch.Tensor, mag_stack: torch.Tensor
    ) -> None:
        """:meth:`_update` for same-shape, same-sharding parameters at once, with their
        stacked momenta ``dir_stack`` ``[n, *shape]`` and ``mag_stack`` ``[n, ..., 1]``
        (updated in place): gradients and weights are stacked too and every operation,
        Newton-Schulz included, runs batched over the stack."""
        weights = self._update(
            group,
            ps[0],
            dir_stack,
            mag_stack,
            grad=torch.stack([p.grad for p in ps]).float(),
            weight=torch.stack([p.data for p in ps]).float(),
        )
        torch._foreach_copy_([p.data for p in ps], list(weights.to(ps[0].dtype).unbind(0)))

    def _update(
        self,
        group: dict,
        p: torch.Tensor,
        dir_buf: torch.Tensor,
        mag_buf: torch.Tensor,
        grad: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
    ):
        """One parameter's Muown update with its direction momentum ``dir_buf`` and its
        magnitude momentum column ``mag_buf`` (both updated in place).

        With ``grad`` / ``weight`` given (fp32 stacks ``[n, *p.shape]`` of parameters
        sharing ``p``'s shape and sharding), updates the stack and returns the new
        weights instead of writing ``p``."""
        lr = group["lr"]
        mag_lr = lr * group["magnitude_lr_scale"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        wd = group["weight_decay"]
        steps = group["num_ns_steps"]
        eps = group["eps"]
        max_row_norm = group["max_row_norm"]

        stacked = grad is not None
        grad = grad if stacked else p.grad.float()
        w = weight if stacked else p.data.float()

        # --- decompose W into row magnitude m and unit-row direction D ---
        # Row reductions go through _row_sum: partial under column sharding.
        m = self._row_sum(w.square().sum(dim=-1, keepdim=True), p).sqrt().clamp(min=eps)
        d = w / m

        # Radial component moves the magnitude; the remainder rotates D.
        radial = self._row_sum((grad * d).sum(dim=-1, keepdim=True), p)
        tangential = grad - radial * d

        dir_buf.mul_(momentum).add_(tangential, alpha=1.0 - momentum)
        mag_buf.mul_(momentum).add_(radial, alpha=1.0 - momentum)

        if nesterov:
            dir_update = tangential.mul(1.0 - momentum).add_(dir_buf, alpha=momentum)
            mag_update = radial.mul(1.0 - momentum).add_(mag_buf, alpha=momentum)
        else:
            dir_update = dir_buf
            mag_update = mag_buf

        # --- direction: Muon (orthogonalized) step, then renormalize ---
        if stacked:
            ortho = self._batched_orthogonalize_fn(
                dir_update.reshape(-1, *dir_update.shape[-2:]), steps=steps, param=p
            ).view_as(dir_update)
        else:
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

        if stacked:
            return m * d
        p.data.copy_((m * d).to(p.dtype))
