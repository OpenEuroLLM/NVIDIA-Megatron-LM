# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Emerging optimizer registry.

To add a new emerging optimizer:
  1. Define its optimizer class (or import it).
  2. Write its ``_<name>_init_state_fn`` and ``_<name>_config_to_kwargs``.
  3. Add an ``EmergingOptimizerEntry`` to ``_EMERGING_OPTIMIZERS`` at the bottom.
"""

import inspect
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, get_args

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size, log_single_rank

from .optimizer_config import ParamKey, ParamPredicate

try:
    from emerging_optimizers import registry
    from emerging_optimizers import utils as eopt_utils
    from emerging_optimizers.orthogonalized_optimizers import (
        AdaptiveMuon,
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT, newton_schulz_tp

    # It is necessary to import optimizers for the registry to work.
    from emerging_optimizers.scalar_optimizers import Lion  # pylint: disable=unused-import
    from emerging_optimizers.soap import SOAP  # pylint: disable=unused-import

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object
    AdaptiveMuon = object


logger = logging.getLogger(__name__)


def get_supported_coefficient_types() -> tuple[str, ...]:
    """Return the coefficient types supported by the installed emerging_optimizers.

    Reads the members of the ``NSCoeffT`` Literal type so that new types
    added upstream are automatically available without code changes here.
    """
    assert (
        HAVE_EMERGING_OPTIMIZERS
    ), "emerging_optimizers >= 0.2 is required for NSCoeffT. Please install or upgrade it."
    return get_args(NSCoeffT)


def validate_coefficient_type(coefficient_type: str) -> None:
    """Raise ``ValueError`` if *coefficient_type* is not supported."""
    supported = get_supported_coefficient_types()
    if coefficient_type not in supported:
        raise ValueError(
            f"Unsupported muon coefficient type '{coefficient_type}'. "
            f"Supported types: {supported}"
        )


# ===========================================================================
# Registry dataclass and public API
# ===========================================================================


def _eopt_init_state_fn(opt, config=None):
    """Initialize emerging optimizer state for torch_dist checkpoint format."""
    for group in opt.param_groups:
        # Checkpoint init needs state for all parameters, including those without grads yet.
        opt._init_group(group, skip_non_grad_params=False)


def _default_param_overrides_factory() -> Dict[ParamKey, Dict[str, Any]]:
    """Default param overrides: route non-linear/embedding params to Adam."""
    return {
        ParamKey(
            predicate=ParamPredicate(name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding)
        ): {'optimizer': 'adam'}
    }


@dataclass
class EmergingOptimizerEntry:
    """Everything needed to create and configure an emerging optimizer.

    Attributes:
        optimizer_cls: The torch optimizer class.
        init_state_fn: Lazily initialises optimizer state (needed for checkpoint formats).
        config_to_kwargs: ``(config, model_chunks, pg_collection) -> dict`` of constructor kwargs.
        default_param_overrides: Per-parameter config overrides applied automatically
            (e.g. route non-linear params to Adam).
    """

    optimizer_cls: type
    init_state_fn: Callable = _eopt_init_state_fn
    config_to_kwargs: Callable | None = None
    default_param_overrides: Dict[ParamKey, Dict[str, Any]] = field(
        default_factory=_default_param_overrides_factory
    )


def _create_emerging_optimizer(config, param_groups, eopt_name, model_chunks, pg_collection):
    """Instantiate an emerging optimizer and return it with its init_state_fn."""
    entry = _EMERGING_OPTIMIZERS[eopt_name]
    if entry.config_to_kwargs is not None:
        eopt_kwargs = entry.config_to_kwargs(config, model_chunks, pg_collection)
    else:
        eopt_kwargs = _default_adam_based_eopt_config_to_kwargs(
            eopt_name, config, model_chunks, pg_collection
        )
    optimizer = entry.optimizer_cls(param_groups, **eopt_kwargs)
    return optimizer, entry.init_state_fn


# ===========================================================================
# Shared helpers
# ===========================================================================


def _is_nonlinear_or_embedding(param):
    """True for parameters that should NOT use the emerging optimizer."""
    return getattr(param, 'is_embedding_or_output_parameter', False) or len(param.shape) != 2


def _get_qkv_split_shapes(model_cfg) -> List[int]:
    """Compute QKV split shapes from model config."""
    return [
        model_cfg.num_attention_heads // model_cfg.num_query_groups * model_cfg.kv_channels,
        model_cfg.kv_channels,
        model_cfg.kv_channels,
    ]


# ===========================================================================
# Registry – populated below only when emerging_optimizers is installed.
# ===========================================================================

_EMERGING_OPTIMIZERS: Dict[str, EmergingOptimizerEntry] = {}


# ===========================================================================
# Muon
# ===========================================================================


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, '
                f'{coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )
            size = [grad.size(-2), grad.size(-1)]
            if partition_dim is not None:
                size[partition_dim] *= get_pg_size(tp_group)
            orth_grad = newton_schulz_tp(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                tp_mode="duplicated" if tp_mode == "blockwise" else tp_mode,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.pg_collection = pg_collection
        self.tp_mode = tp_mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        # Use explicit class call instead of super() so that subclasses with
        # multiple inheritance (e.g. TensorParallelAdaptiveMuon) don't route
        # through an intermediate class that doesn't accept scaled_orthogonalize_fn.
        OrthogonalizedOptimizer.__init__(
            self,
            params,
            lr,
            momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to
                momentum because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        # TODO(deyuf): switch to group
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.tp_mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        if self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            grad_shape = grad.shape
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad_shape}, ' f'split shapes {self.qkv_split_shapes}',
            )
            num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)
            qkv_grads = torch.split(
                grad.view(num_query_groups, sum(self.qkv_split_shapes), -1),
                self.qkv_split_shapes,
                dim=1,
            )
            qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]

            qkv_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, partition_dim).view(
                    num_query_groups, -1, grad_shape[-1]
                )
                for g in qkv_grads
            ]
            grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        else:
            grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        return grad


class TensorParallelAdaptiveMuon(TensorParallelMuon, AdaptiveMuon):
    """Tensor Parallel Adaptive Muon optimizer.

    This class extends Muon by adding AdamW-style or NorMuon-style second moment
    accumulation after orthogonalization. This idea was first explored in D.E. Carlson,
    E. Collins, Ya-Ping Hsieh, L. Carin, and V. Cevher. *Preconditioned spectral
    descent for deep learning.* In Advances in neural information processing systems 28 (2015).
    The step() method is overridden to include second moment normalization logic.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        momentum: The exponential decay rate for momentum.
        nesterov: Whether to use Nesterov momentum.
        weight_decay: Weight decay coefficient.
        use_decoupled_weight_decay: Whether to use decoupled weight decay.
        split_qkv: Whether to split QKV weights for orthogonalization.
        is_qkv_fn: Function to determine if a tensor is a QKV weight.
        qkv_split_shapes: Shapes for splitting QKV weights.
        fp32_matmul_prec: Precision for FP32 matrix multiplication.
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration.
        num_ns_steps: The number of iteration steps to use in the Newton-Schulz iteration.
        scale_mode: The type of scale factor to use for the update.
        extra_scale_factor: The additional scale factor to use for the update.
        pg_collection: Process group collection for distributed training.
        tp_mode: Tensor parallel mode ("blockwise", "duplicated", or "distributed").
        moment2_method: Method for second moment accumulation ("adamuon" or "normuon").
        beta2: The exponential decay rate for second moment.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
        moment2_method: Literal["adamuon", "normuon"] = "adamuon",
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        TensorParallelMuon.__init__(
            self,
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            split_qkv=split_qkv,
            is_qkv_fn=is_qkv_fn,
            qkv_split_shapes=qkv_split_shapes,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
        )
        self.moment2_method = moment2_method

        for group in self.param_groups:
            group.setdefault("beta2", beta2)
            group.setdefault("eps", eps)

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Step function"""
        return AdaptiveMuon.step(self, closure)


# ===========================================================================
# AngularMuown
# ===========================================================================

ANGULAR_MUOWN_EPS = 1e-7
# Expected row norm of a default (Kaiming-uniform) linear init, used to seed
# the `g` state for rows that start at exactly zero.
ANGULAR_MUOWN_ZERO_ROW_SCALE = 0.33**0.5
# Per-row optimizer states stored as (rows, 1). These are expanded to the
# weight's full (rows, cols) shape for torch_dist checkpointing (see
# ``TensorParallelAngularMuown.state_dict``). ``m_u`` already matches the weight
# shape and ``step`` is a scalar, so neither needs expansion.
ANGULAR_MUOWN_ROW_STATE_KEYS = ("g", "m_g", "v_g")


@torch.compile
def _angular_muown_u_gradients(
    w: torch.Tensor, g: torch.Tensor, grad_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return U, dL/dg, and the Riemannian U gradient for W = diag(g) @ U."""
    g_safe = torch.copysign(g.abs().clamp_min(ANGULAR_MUOWN_EPS), g)
    u = w / g_safe
    grad_g = (grad_w * u).sum(dim=1, keepdim=True)
    grad_u = g * (grad_w - u * grad_g)
    return u, grad_g, grad_u


@torch.compile
def _angular_muown_u_recompose(
    w: torch.Tensor, g: torch.Tensor, u_step: torch.Tensor, eps: float = ANGULAR_MUOWN_EPS
) -> None:
    """Normalize U rows and write W = diag(g) @ U in place."""
    u_step_norm = u_step.norm(dim=1, keepdim=True).clamp_min(eps)
    w.copy_(g * (u_step / u_step_norm))


class TensorParallelAngularMuown(TensorParallelMuon):
    """AngularMuown: optimizer for 2D hidden weight matrices in ``W = diag(g) @ U`` coordinates.

    AngularMuown stores one row magnitude ``g`` per row and treats the normalized rows
    ``U`` as points on a product of unit spheres. The row magnitudes are optimized
    with Adam. The row directions are optimized with a Muon-style orthogonalized
    direction followed by row-normalization retraction:

        grad_g = <grad_W, U>_row
        grad_U = g * (grad_W - U * grad_g)
        M = momentum * M + grad_U
        Q = scaled_orthogonalize(grad_U + momentum * M)   # with nesterov=True
        U = row_norm(U - lr * u_lr_multiplier * Q)
        g = Adam(g, grad_g, lr)

    The orthogonalization, tensor-parallel handling, and split-QKV handling are
    inherited from :class:`TensorParallelMuon` (Newton-Schulz via
    ``emerging_optimizers``). The shape-dependent scale is applied inside the
    inherited ``scaled_orthogonalize_fn`` via ``get_muon_scale_factor``:

    - ``scale_mode="spectral"`` (default) gives ``sqrt(max(m, n))``, matching
      the Muon recipe convention.
    - ``scale_mode="shape_scaling"`` gives ``sqrt(max(1, m / n))``,
      the original AngularMuown "ratio" scaling.
    - ``scale_mode="spectral"`` with ``extra_scale_factor=0.2`` gives the
      standard Muon ``0.2 * sqrt(max(m, n))`` scaling (original "muon" mode).

    Pass only 2D hidden matrices to this optimizer; embeddings, output heads,
    biases, norms etc. must go to a separate AdamW (the Megatron registry routes
    them automatically). Weight decay is not part of the AngularMuown update.

    AngularMuown owns an internal decay schedule for the directional step
    (``u_lr_multiplier``); the ordinary learning rate (which still controls Adam
    on ``g``) can use the regular external scheduler.

    Args:
        params: Iterable of 2D parameters or parameter-group dictionaries.
        lr: Base learning rate used by Adam on ``g`` and by the ``U`` update.
        momentum: Momentum coefficient for the ``U``-gradient buffer.
        nesterov: Whether to use Nesterov-style lookahead for the ``U`` update.
        betas: Adam betas for the row magnitude ``g``.
        adam_eps: Adam epsilon for the row magnitude ``g``.
        split_qkv: Whether to split fused QKV weights for orthogonalization.
        is_qkv_fn: Function to determine if a tensor is a fused QKV weight.
        qkv_split_shapes: Per-query-group (q, k, v) split shapes.
        fp32_matmul_prec: Precision of matmuls in the orthogonalization.
        coefficient_type: Newton-Schulz coefficient set. ``"simple"`` matches the
            original AngularMuown ``newtonschulz5`` backend coefficients.
        num_ns_steps: Newton-Schulz iteration count.
        scale_mode: Shape-dependent scale mode (see above).
        extra_scale_factor: Additional scalar applied to the ``U`` direction.
        pg_collection: Process group collection for distributed training.
        tp_mode: Tensor parallel mode ("blockwise", "duplicated", "distributed").
        u_decay_schedule: Internal decay schedule for the directional step
            multiplier; one of ``"poly"`` or ``"cosine"``.
        u_decay_scale: Scale for the ``"poly"`` schedule
            ``(1 + u_decay_scale * steps_after_warmup) ** (-u_decay_p)``. ``None``
            disables decay (multiplier stays ``1.0``).
        u_decay_p: Exponent for the ``"poly"`` schedule.
        u_decay_warmup_steps: Steps before the decay begins.
        u_decay_steps: Post-warmup steps over which the ``"cosine"`` schedule
            decays from ``1.0`` to ``u_decay_min_multiplier``.
        u_decay_min_multiplier: Floor of the ``"cosine"`` schedule.

    Row magnitudes whose absolute value would fall below ``ANGULAR_MUOWN_EPS`` are
    projected back to magnitude ``ANGULAR_MUOWN_EPS`` with their proposed sign so the
    explicit ``W = diag(g) @ U`` coordinates stay well-defined.

    Reference:
        Florian Hübler, Kai Lion, Antonio Orvieto, Niao He.
        "Muown Implicitly Performs Angular Step-size Decay." arXiv:2606.23637, 2026.
        https://arxiv.org/abs/2606.23637
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "simple",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "blockwise",
        u_decay_schedule: str = "poly",
        u_decay_scale: float | None = None,
        u_decay_p: float = 1.0,
        u_decay_warmup_steps: int = 0,
        u_decay_steps: int | None = None,
        u_decay_min_multiplier: float = 0.0,
    ) -> None:
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if adam_eps < 0.0:
            raise ValueError(f"Invalid adam_eps: {adam_eps}")
        self._validate_u_decay(
            u_decay_schedule,
            u_decay_scale,
            u_decay_p,
            u_decay_warmup_steps,
            u_decay_steps,
            u_decay_min_multiplier,
        )

        TensorParallelMuon.__init__(
            self,
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            # AngularMuown has no weight decay; magnitudes are fully owned by Adam on g.
            weight_decay=0.0,
            use_decoupled_weight_decay=True,
            split_qkv=split_qkv,
            is_qkv_fn=is_qkv_fn,
            qkv_split_shapes=qkv_split_shapes,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
        )

        self._u_decay_schedule = u_decay_schedule
        self._u_decay_scale = None if u_decay_scale is None else float(u_decay_scale)
        self._u_decay_p = float(u_decay_p)
        self._u_decay_warmup_steps = int(u_decay_warmup_steps)
        self._u_decay_steps = None if u_decay_steps is None else int(u_decay_steps)
        self._u_decay_min_multiplier = float(u_decay_min_multiplier)

        for group in self.param_groups:
            group.setdefault("betas", betas)
            group.setdefault("adam_eps", adam_eps)
            group.setdefault("u_lr_multiplier", 1.0)
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        "AngularMuown only supports 2D parameters, but found "
                        f"a parameter with size: {p.size()}"
                    )

    @staticmethod
    def _validate_u_decay(schedule, scale, p, warmup_steps, decay_steps, min_multiplier) -> None:
        """Validate the internal U-direction decay schedule arguments."""
        if schedule not in ("poly", "cosine"):
            raise ValueError(
                f"Invalid u_decay_schedule: {schedule!r}. Choose from 'poly' or 'cosine'."
            )
        if not math.isfinite(p) or p < 0.0:
            raise ValueError(f"Invalid u_decay_p: {p}")
        if int(warmup_steps) != warmup_steps or warmup_steps < 0:
            raise ValueError(f"Invalid u_decay_warmup_steps: {warmup_steps}")
        if schedule == "poly":
            if scale is not None and (not math.isfinite(scale) or scale <= 0.0):
                raise ValueError(f"Invalid u_decay_scale: {scale}")
        else:  # "cosine"
            if decay_steps is None:
                raise ValueError("u_decay_schedule='cosine' requires u_decay_steps")
            if int(decay_steps) != decay_steps or decay_steps <= 0:
                raise ValueError(f"Invalid u_decay_steps: {decay_steps}")
            if not math.isfinite(min_multiplier) or not 0.0 <= min_multiplier <= 1.0:
                raise ValueError(f"Invalid u_decay_min_multiplier: {min_multiplier}")

    def _u_lr_multiplier(self, step: int) -> float:
        """Return the directional-step multiplier for the given schedule step."""
        steps_after_warmup = max(0, step - self._u_decay_warmup_steps)
        if self._u_decay_schedule == "cosine":
            progress = min(1.0, steps_after_warmup / self._u_decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self._u_decay_min_multiplier + (1.0 - self._u_decay_min_multiplier) * cosine
        if self._u_decay_scale is None or self._u_decay_p == 0.0:
            return 1.0
        return (1.0 + self._u_decay_scale * steps_after_warmup) ** (-self._u_decay_p)

    @torch.no_grad()  # type: ignore[misc]
    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        """Lazily initialize AngularMuown state (g, m_u, m_g, v_g, step) for 2D params."""
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue
            state = self.state[p]
            if len(state) != 0:
                continue
            # Zero rows have no direction for U = W / g. Give just those rows the
            # expected row scale of the default linear init so the first U update
            # can create a unit direction without starting Adam's g near zero.
            w_norm = p.detach().norm(dim=1, keepdim=True)
            zero_rows = w_norm <= ANGULAR_MUOWN_EPS
            if zero_rows.any():
                w_norm = torch.where(
                    zero_rows, w_norm.new_full(w_norm.shape, ANGULAR_MUOWN_ZERO_ROW_SCALE), w_norm
                )
            state["g"] = w_norm.clone()
            state["m_u"] = torch.zeros_like(p)
            state["m_g"] = torch.zeros_like(w_norm)
            state["v_g"] = torch.zeros_like(w_norm)
            state["step"] = 0

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform one AngularMuown update on all 2D parameters with gradients."""
        if closure is not None:
            raise ValueError("closure is not supported")

        for group in self.param_groups:
            self._init_group(group)

            lr = group["lr"]
            momentum = group["momentum"]
            beta1, beta2 = group["betas"]
            adam_eps = group["adam_eps"]
            group_kwargs = {k: v for k, v in group.items() if k != "params"}

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                state["step"] += 1
                step = state["step"]
                g = state["g"]
                m_u = state["m_u"]
                m_g = state["m_g"]
                v_g = state["v_g"]

                u, grad_g, grad_u = _angular_muown_u_gradients(p, g, p.grad)

                # Momentum on the Riemannian U gradient (sum convention, as in
                # the original AngularMuown, not the EMA convention of the base class).
                m_u.mul_(momentum).add_(grad_u)
                if self.nesterov:
                    update = grad_u.add(m_u, alpha=momentum)
                else:
                    update = m_u.clone()

                # Orthogonalize (and scale) via the inherited TP/QKV-aware path.
                with eopt_utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    direction = self.orthogonalize(p, update, **group_kwargs)

                # Original AngularMuown evaluated the schedule before incrementing its
                # global step counter, hence `step - 1`.
                u_multiplier = self._u_lr_multiplier(step - 1)
                group["u_lr_multiplier"] = u_multiplier
                u_step = u.add(direction, alpha=-lr * u_multiplier)

                # Adam update on the row magnitudes g, with sign-preserving
                # projection away from zero.
                m_g.mul_(beta1).add_(grad_g, alpha=1 - beta1)
                v_g.mul_(beta2).addcmul_(grad_g, grad_g, value=1 - beta2)
                bc1 = 1 - beta1**step
                bc2 = 1 - beta2**step
                g_update = (m_g / bc1) / (v_g / bc2).sqrt().add_(adam_eps)
                g_candidate = g.add(g_update, alpha=-lr)
                g.copy_(torch.copysign(g_candidate.abs().clamp_min(ANGULAR_MUOWN_EPS), g_candidate))

                _angular_muown_u_recompose(p, g, u_step)

        return None

    def state_dict(self) -> dict:
        """Return a checkpoint-friendly state dict with full-shape per-row states.

        AngularMuown keeps the per-row states ``g``/``m_g``/``v_g`` at shape
        ``(rows, 1)``, but Megatron's generic optimizer-state sharding
        (``optim_state_to_sharding_state`` -> ``make_sharded_optimizer_tensor``)
        assumes every per-parameter state tensor has the same shape as the model
        weight ``(rows, cols)`` and reuses the weight's sharded metadata. The
        reduced ``(rows, 1)`` states would otherwise trip the shape assertion and
        crash the first ``torch_dist`` save.

        To keep the fix fully inside this optimizer (no Megatron-core changes),
        we expand the per-row states to the weight's full shape here so they
        inherit the weight's exact sharding (correct for any TP/EP, row- or
        column-parallel layout). ``load_state_dict`` collapses them back. The
        expanded columns are identical copies, so the round-trip is exact; the
        only cost is extra checkpoint footprint for these states. The live
        optimizer state is left untouched (still ``(rows, 1)``).
        """
        state_dict = super().state_dict()
        packed_state = state_dict.get("state")
        if not packed_state:
            return state_dict

        expanded_state = {}
        for param_id, param_state in packed_state.items():
            new_param_state = dict(param_state)
            m_u = param_state.get("m_u")
            if m_u is not None and m_u.dim() == 2 and m_u.shape[1] > 1:
                cols = m_u.shape[1]
                for key in ANGULAR_MUOWN_ROW_STATE_KEYS:
                    tensor = param_state.get(key)
                    if tensor is not None and tensor.dim() == 2 and tensor.shape[1] == 1:
                        new_param_state[key] = tensor.expand(-1, cols).contiguous()
            expanded_state[param_id] = new_param_state

        state_dict["state"] = expanded_state
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Collapse full-shape per-row states back to ``(rows, 1)`` before loading.

        Inverse of :meth:`state_dict`. Checkpoints written by this optimizer
        store ``g``/``m_g``/``v_g`` expanded to the weight shape ``(rows, cols)``
        with identical columns; we restore the canonical ``(rows, 1)`` layout by
        taking the first column. Guarded so already-reduced states (e.g. a fresh
        optimizer or a torch-format checkpoint) pass through unchanged.
        """
        packed_state = state_dict.get("state")
        if packed_state:
            collapsed_state = {}
            for param_id, param_state in packed_state.items():
                new_param_state = dict(param_state)
                for key in ANGULAR_MUOWN_ROW_STATE_KEYS:
                    tensor = param_state.get(key)
                    if tensor is not None and tensor.dim() == 2 and tensor.shape[1] > 1:
                        new_param_state[key] = tensor[:, :1].contiguous()
                collapsed_state[param_id] = new_param_state
            state_dict = dict(state_dict)
            state_dict["state"] = collapsed_state

        super().load_state_dict(state_dict)


def _kwargs_from_config(optimizer_cls: type, prefix: str, config) -> Dict[str, Any]:
    """Match ``optimizer_cls.__init__`` parameters to config attributes.

    For each init parameter, looks for ``{prefix}_{name}`` on *config* first,
    then falls back to ``{name}`` (unprefixed).  ``self`` and ``params`` are
    always skipped.
    """
    skip_params = {"self", "params"}
    sig = inspect.signature(optimizer_cls.__init__)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters:
        if name in skip_params:
            continue
        prefixed = f"{prefix}_{name}"
        if hasattr(config, prefixed):
            kwargs[name] = getattr(config, prefixed)
        elif hasattr(config, name):
            kwargs[name] = getattr(config, name)
    return kwargs


def _muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelMuon constructor kwargs."""
    kwargs = _kwargs_from_config(TensorParallelMuon, "muon", config)
    kwargs["is_qkv_fn"] = lambda p: getattr(p, "is_qkv", False)
    kwargs["qkv_split_shapes"] = _get_qkv_split_shapes(model_chunks[0].config)
    kwargs["pg_collection"] = pg_collection
    return kwargs


def _adaptive_muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelAdaptiveMuon constructor kwargs."""
    kwargs = _muon_config_to_kwargs(config, model_chunks, pg_collection)
    kwargs.update(_kwargs_from_config(TensorParallelAdaptiveMuon, "adaptive_muon", config))
    return kwargs


def _angular_muown_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelAngularMuown constructor kwargs."""
    kwargs = _kwargs_from_config(TensorParallelAngularMuown, "angular_muown", config)
    kwargs["betas"] = (config.angular_muown_beta1, config.angular_muown_beta2)
    kwargs["is_qkv_fn"] = lambda p: getattr(p, "is_qkv", False)
    kwargs["qkv_split_shapes"] = _get_qkv_split_shapes(model_chunks[0].config)
    kwargs["pg_collection"] = pg_collection
    return kwargs


def _default_adam_based_eopt_config_to_kwargs(
    eopt_name, config, model_chunks, pg_collection
) -> Dict[str, Any]:
    """Convert OptimizerConfig to default emerging optimizer constructor kwargs."""
    kwargs = _kwargs_from_config(registry.get_optimizer_cls(eopt_name), eopt_name, config)
    kwargs["betas"] = (config.adam_beta1, config.adam_beta2)
    return kwargs


# -----------------------------------------------------------------------
# Register emerging optimizers
# -----------------------------------------------------------------------
_EMERGING_OPTIMIZERS.update(
    {
        'muon': EmergingOptimizerEntry(
            optimizer_cls=TensorParallelMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'}
            },
        ),
        "adaptive_muon": EmergingOptimizerEntry(
            optimizer_cls=TensorParallelAdaptiveMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_adaptive_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'}
            },
        ),
        "angular_muown": EmergingOptimizerEntry(
            optimizer_cls=TensorParallelAngularMuown,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_angular_muown_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'},
                # MoE router/gate weights are 2D but small and sensitive; keep
                # them on Adam rather than AngularMuown.
                ParamKey(name=('*router.weight', '*gate_weight')): {'optimizer': 'adam'},
            },
        ),
    }
)

# Register soap with default config
# TODO(skyw): register all emerging optimizers.
if HAVE_EMERGING_OPTIMIZERS:
    for eopt_name in registry.get_optimizer_name_list():
        if eopt_name in _EMERGING_OPTIMIZERS:
            # skip already registered local versions, e.g. TensorParallel versions.
            continue
        _EMERGING_OPTIMIZERS[eopt_name] = EmergingOptimizerEntry(
            optimizer_cls=registry.get_optimizer_cls(eopt_name)
        )
