# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Emerging optimizer registry.

To add a new emerging optimizer:
  1. Define its optimizer class (or import it).
  2. Write its ``_<name>_init_state_fn`` and ``_<name>_config_to_kwargs``.
  3. Add an ``EmergingOptimizerEntry`` to ``_EMERGING_OPTIMIZERS`` at the bottom.
"""

import inspect
import os
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, get_args

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_rank, get_pg_size, log_single_rank

from .fsdp_row_exchange import is_row_sharded, local, orthogonalize_rows, row_state_like
from .muown import Muown
from .optimizer_config import ParamKey, ParamPredicate

try:
    from emerging_optimizers import registry
    from emerging_optimizers import utils as eopt_utils
    from emerging_optimizers.orthogonalized_optimizers import (
        AdaptiveMuon,
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
        _COEFFICIENT_SETS,
        NSCoeffT,
        get_coefficient_iterator,
        newton_schulz_tp,
    )

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
    """True for parameters that should NOT use the emerging optimizer.

    Besides embeddings and non-matrices, that covers 2-D parameters that are not
    linear maps: the Engram and product-key-memory lookup tables (embeddings, read
    sparsely) and parameters tagged ``use_adam_with_emerging_optimizers`` (per-channel
    conv kernels such as Canon's, and the mHC / routed residual stream mixers).
    """
    return (
        getattr(param, 'is_embedding_or_output_parameter', False)
        or len(param.shape) != 2
        or getattr(param, 'is_engram_table', False)
        or getattr(param, 'is_memory_table', False)
        or getattr(param, 'use_adam_with_emerging_optimizers', False)
    )


def _get_qkv_split_shapes(model_cfg, headwise: bool = False) -> list[int]:
    """Compute QKV split shapes from model config (per query group: q [, gate], k, v; headwise:
    one kv_channels block per head, in the same layout)."""
    if headwise:
        heads_per_group = model_cfg.num_attention_heads // model_cfg.num_query_groups
        gate = 2 if getattr(model_cfg, 'attention_output_gate', False) else 1
        return [model_cfg.kv_channels] * (gate * heads_per_group + 2)
    query_projection_size = (
        model_cfg.num_attention_heads // model_cfg.num_query_groups * model_cfg.kv_channels
    )
    if getattr(model_cfg, 'attention_output_gate', False):
        return [
            query_projection_size,
            query_projection_size,
            model_cfg.kv_channels,
            model_cfg.kv_channels,
        ]
    return [query_projection_size, model_cfg.kv_channels, model_cfg.kv_channels]


# ===========================================================================
# Registry – populated below only when emerging_optimizers is installed.
# ===========================================================================

_EMERGING_OPTIMIZERS: Dict[str, EmergingOptimizerEntry] = {}


# ===========================================================================
# Muon
# ===========================================================================


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer.

    ``batched_step`` stacks parameters that share shape and distributed semantics,
    reducing launch and collective overhead while retaining per-parameter state.
    It is disabled by default to preserve the upstream execution path.
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
        qkv_split_shapes: list[int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
        batched_step: bool = False,
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
        self.batched_step = batched_step
        self._num_ns_steps = num_ns_steps
        self._coefficient_type = coefficient_type
        self._scale_mode = scale_mode
        self._extra_scale_factor = extra_scale_factor

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

    def scaled_orthogonalize_fn_with_gtp_remat(self, p, grad, tp_group, partition_dim):
        """All-gather grad along GTP_remat/EGTP_remat dim 0, orthogonalize, then slice back.

        GTP_remat shards weights along dim 0 independently of TP's partition_dim. Newton-Schulz
        needs the full weight matrix, so we reconstruct the GTP_remat dimension before running
        the TP-aware orthogonalization, then extract the local GTP_remat shard from the result.
        When GTP_remat is inactive this is a plain passthrough to scaled_orthogonalize_fn.
        """
        # TODO: Clean up code that determines if parameter is a MoE layer and which TP group to use
        is_expert = getattr(p, 'expert_tp', False)
        gtp_remat_group = (
            (self.pg_collection.expt_gtp_remat if is_expert else self.pg_collection.gtp_remat)
            if self.pg_collection
            else None
        )

        if gtp_remat_group is None or get_pg_size(gtp_remat_group) <= 1:
            return self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)

        # Parameters with is_gtp_weight_remat=False are not sharded along the
        # GTP process group, and do not require all-gathering prior to
        # orthogonalization.
        if not getattr(p, 'is_gtp_weight_remat', False):
            return self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)

        gtp_remat_size = get_pg_size(gtp_remat_group)
        gtp_rank = get_pg_rank(gtp_remat_group)
        shards = [torch.empty_like(grad) for _ in range(gtp_remat_size)]
        torch.distributed.all_gather(shards, grad, gtp_remat_group)
        gathered_grad = torch.cat(shards, dim=0)

        gathered_grad = self.scaled_orthogonalize_fn(gathered_grad, tp_group, partition_dim)

        shard_size = gathered_grad.shape[0] // gtp_remat_size
        return gathered_grad[gtp_rank * shard_size : (gtp_rank + 1) * shard_size].contiguous()

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
        if partition_dim is not None and (tp_group is None or get_pg_size(tp_group) <= 1):
            partition_dim = None

        if self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            grad_shape = grad.shape
            qkv_split_shapes = getattr(p, "qkv_split_shapes", None)
            if qkv_split_shapes is None:
                qkv_split_shapes = self.qkv_split_shapes
            if qkv_split_shapes is None:
                raise RuntimeError("Muon QKV split requested but qkv_split_shapes is not set")
            qkv_split_dim = sum(qkv_split_shapes)
            if grad_shape[0] % qkv_split_dim != 0:
                raise RuntimeError(
                    f"Muon QKV split shape mismatch: grad_shape={tuple(grad_shape)}, "
                    f"split_shapes={qkv_split_shapes}"
                )
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad_shape}, split shapes {qkv_split_shapes}',
            )
            num_query_groups = grad_shape[0] // qkv_split_dim
            qkv_grads = torch.split(
                grad.view(num_query_groups, qkv_split_dim, -1), qkv_split_shapes, dim=1
            )
            qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]

            qkv_grads = [
                self.scaled_orthogonalize_fn_with_gtp_remat(p, g, tp_group, partition_dim).view(
                    num_query_groups, -1, grad_shape[-1]
                )
                for g in qkv_grads
            ]
            grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        else:
            grad = self.scaled_orthogonalize_fn_with_gtp_remat(p, grad, tp_group, partition_dim)
        return grad

    def _tp_row_reduce_group(self, p: torch.Tensor) -> Optional["torch.distributed.ProcessGroup"]:
        """Return the TP group used for per-row reductions, if any."""
        return None

    def _uses_gtp_remat(self, p: torch.Tensor) -> bool:
        """Whether *p* requires the v0.19 GTP-remat orthogonalization path."""
        if self.pg_collection is None or not getattr(p, 'is_gtp_weight_remat', False):
            return False
        is_expert = getattr(p, 'expert_tp', False)
        group = self.pg_collection.expt_gtp_remat if is_expert else self.pg_collection.gtp_remat
        return group is not None and get_pg_size(group) > 1

    def _effective_qkv_split_shapes(self, p: torch.Tensor) -> tuple[int, ...] | None:
        """Return the parameter-specific QKV split layout used by v0.19."""
        if not (self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p)):
            return None
        shapes = getattr(p, "qkv_split_shapes", None)
        if shapes is None:
            shapes = self.qkv_split_shapes
        if shapes is None:
            raise RuntimeError("Muon QKV split requested but qkv_split_shapes is not set")
        return tuple(shapes)

    def _batch_key_for(self, p: torch.Tensor) -> tuple[tuple, dict]:
        """Build a key for parameters that share all batched-step semantics."""
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
        if partition_dim is not None and (tp_group is None or get_pg_size(tp_group) <= 1):
            partition_dim = None
        if partition_dim is None:
            tp_group = None

        reduce_group = self._tp_row_reduce_group(p)
        qkv_split_shapes = self._effective_qkv_split_shapes(p)
        key = (tuple(p.shape), partition_dim, id(tp_group), id(reduce_group), qkv_split_shapes)
        return key, {
            "tp_group": tp_group,
            "reduce_group": reduce_group,
            "partition_dim": partition_dim,
            "qkv_split_shapes": qkv_split_shapes,
        }

    def _scaled_ns_batched(self, x: torch.Tensor, batch: dict) -> torch.Tensor:
        """Batched equivalent of the configured scaled orthogonalization."""
        partition_dim = batch["partition_dim"]
        tp_group = batch["tp_group"]
        size = [x.size(-2), x.size(-1)]
        if partition_dim is not None:
            size[partition_dim] *= get_pg_size(tp_group)
        orth = _batched_newton_schulz(
            x,
            steps=self._num_ns_steps,
            coefficient_type=self._coefficient_type,
            tp_group=tp_group,
            partition_dim=partition_dim,
            tp_mode="duplicated" if self.tp_mode == "blockwise" else self.tp_mode,
        )
        scale_factor = get_muon_scale_factor(size[0], size[1], mode=self._scale_mode)
        return orth * (scale_factor * self._extra_scale_factor)

    def _orthogonalize_batched(self, update: torch.Tensor, batch: dict) -> torch.Tensor:
        """Batched orthogonalization with v0.19 parameter-specific QKV layouts."""
        qkv_split_shapes = batch["qkv_split_shapes"]
        if qkv_split_shapes is None:
            return self._scaled_ns_batched(update, batch)

        batch_size, rows, cols = update.shape
        split_rows = sum(qkv_split_shapes)
        if rows % split_rows != 0:
            raise RuntimeError(
                f"Muon QKV split shape mismatch: grad_shape={tuple(update.shape)}, "
                f"split_shapes={qkv_split_shapes}"
            )
        num_query_groups = rows // split_rows
        qkv = torch.split(
            update.view(batch_size, num_query_groups, split_rows, cols), qkv_split_shapes, dim=2
        )
        outputs = [
            self._scaled_ns_batched(part.reshape(batch_size, -1, cols), batch).view(
                batch_size, num_query_groups, -1, cols
            )
            for part in qkv
        ]
        return torch.cat(outputs, dim=2).view(batch_size, rows, cols)

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a Muon update, batching same-shape parameters when enabled."""
        if any(is_row_sharded(p) for group in self.param_groups for p in group["params"]):
            loss = None if closure is None else closure()
            for group in self.param_groups:
                self._init_group(group, skip_non_grad_params=False)
                self._step_group_row_sharded(group)
            return loss
        active_params = (
            p for group in self.param_groups for p in group["params"] if p.grad is not None
        )
        if not self.batched_step or any(self._uses_gtp_remat(p) for p in active_params):
            # The v0.19 GTP-remat path gathers an additional sharded dimension.
            # Keep its established per-parameter implementation until batching
            # that second process-group dimension is explicitly supported.
            return OrthogonalizedOptimizer.step(self, closure)

        loss = None if closure is None else closure()
        for group in self.param_groups:
            self._init_group(group)
            self._step_group_batched(group)
        return loss

    def _step_group_row_sharded(self, group: dict) -> None:
        """Muon on Megatron-FSDP (row-sharded) parameters: weight decay and momentum on the
        local rows, the Newton-Schulz on each matrix's owner rank (fsdp_row_exchange)."""
        if any(self._uses_gtp_remat(p) for p in group["params"]):
            raise NotImplementedError("GTP remat with Megatron-FSDP")
        lr = group["lr"]
        momentum = group["momentum"]
        weight_decay = group["weight_decay"]
        # requires_grad, not p.grad: the parameter list must be the same on every rank
        params = [p for p in group["params"] if p.requires_grad]
        if not params:
            return
        weights = [local(p).detach() for p in params]
        grads = [_local_grad(p) for p in params]
        if weight_decay != 0.0:
            if self.weight_decay_method == "decoupled":
                torch._foreach_add_(weights, weights, alpha=-weight_decay * lr)
            elif self.weight_decay_method == "independent":
                torch._foreach_add_(weights, weights, alpha=-weight_decay)
            elif self.weight_decay_method == "l2":
                grads = torch._foreach_add(grads, weights, alpha=weight_decay)
            else:
                raise ValueError(f"Invalid weight decay method: {self.weight_decay_method}")
        momentum_buffers = [local(self.state[p]["momentum_buffer"]) for p in params]
        torch._foreach_lerp_(momentum_buffers, grads, 1 - momentum)
        updates = (
            torch._foreach_lerp(grads, momentum_buffers, momentum)
            if self.nesterov
            else momentum_buffers
        )
        updates = [u.float() for u in updates]

        def ns(stack, metadata):
            with eopt_utils.fp32_matmul_precision(self.fp32_matmul_prec):
                return self._orthogonalize_batched(stack, metadata)

        metas = [self._batch_key_for(p) for p in params]
        directions = orthogonalize_rows(params, updates, metas, ns)
        directions = [d.to(w.dtype) for d, w in zip(directions, weights)]
        torch._foreach_add_(weights, directions, alpha=-lr)

    def _step_group_batched(self, group: dict) -> None:
        """Run the base Muon update on stacks of same-shape parameters."""
        lr = group["lr"]
        momentum = group["momentum"]
        weight_decay = group["weight_decay"]
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        grads = [p.grad for p in params]

        if weight_decay != 0.0:
            if self.weight_decay_method == "decoupled":
                torch._foreach_add_(params, params, alpha=-weight_decay * lr)
            elif self.weight_decay_method == "independent":
                torch._foreach_add_(params, params, alpha=-weight_decay)
            elif self.weight_decay_method == "l2":
                torch._foreach_add_(grads, params, alpha=weight_decay)
            else:
                raise ValueError(f"Invalid weight decay method: {self.weight_decay_method}")

        momentum_buffers = [self.state[p]["momentum_buffer"] for p in params]
        torch._foreach_lerp_(momentum_buffers, grads, 1 - momentum)
        updates = (
            torch._foreach_lerp(grads, momentum_buffers, momentum)
            if self.nesterov
            else momentum_buffers
        )

        batches: Dict[tuple, dict] = {}
        for index, param in enumerate(params):
            key, metadata = self._batch_key_for(param)
            batch = batches.setdefault(key, {**metadata, "indices": []})
            batch["indices"].append(index)

        for batch in batches.values():
            indices = batch["indices"]
            x = torch.stack([updates[index] for index in indices])
            with eopt_utils.fp32_matmul_precision(self.fp32_matmul_prec):
                direction = self._orthogonalize_batched(x, batch)
            batch_params = [params[index] for index in indices]
            directions = list(direction.unbind(0))
            for param, param_direction in zip(batch_params, directions):
                self.pre_weight_update_fn_inplace(param, param_direction)
            torch._foreach_add_(batch_params, directions, alpha=-lr)
            for param in batch_params:
                self.post_weight_update_fn_inplace(param)


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
        qkv_split_shapes: list[int] | None = None,
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
# Checkpoint the per-row states at (rows, 1) instead of expanded to the weight shape: ~12 bytes per
# matrix parameter less in every save's host staging (the expanded form OOM-killed s2 MoE nodes
# at saves) and in the checkpoint. A different checkpoint format: a run must use one setting
# throughout (expanded checkpoints do not load into compact templates and vice versa).
_COMPACT_ROW_STATES = os.environ.get("ANGULAR_MUOWN_COMPACT_CKPT", "0") == "1"

# The adaptive schedules' current direction multiplier (and the hypergradient signal), for
# training_log; written by every TensorParallelAngularMuown step (identical across ranks).
ANGULAR_MUOWN_KAPPA: Dict[str, float] = {}


@torch.compile
def _local_grad(p: torch.Tensor) -> torch.Tensor:
    """The local gradient rows of p; zeros where Megatron-FSDP left none (an empty shard)."""
    if p.grad is not None:
        return local(p.grad)
    return torch.zeros_like(local(p))


def _angular_muown_u_and_grad_g(
    w: torch.Tensor, g: torch.Tensor, grad_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return U and the row gradient dL/dg for W = diag(g) @ U.

    ``grad_g = <grad_W, U>_row`` is a ``dim=-1`` reduction across the columns of
    each row. Under column tensor-parallel sharding (``partition_dim == 1``)
    each rank only holds a column slice of every row, so this is a *partial*
    sum that the caller must all-reduce across the TP group before it is used
    (both for the Adam-on-``g`` update and for :func:`_angular_muown_grad_u`).
    For row-sharded / replicated / single-GPU weights it is already global.
    The all-reduce is kept out of this ``torch.compile`` region on purpose.

    Accepts a single ``(rows, cols)`` weight or a ``(B, rows, cols)`` stack of
    same-shape weights (with ``g`` of shape ``(B, rows, 1)``).
    """
    g_safe = torch.copysign(g.abs().clamp_min(ANGULAR_MUOWN_EPS), g)
    u = w / g_safe
    grad_g = (grad_w * u).sum(dim=-1, keepdim=True)
    return u, grad_g


@torch.compile
def _angular_muown_grad_u(
    u: torch.Tensor, g: torch.Tensor, grad_w: torch.Tensor, grad_g: torch.Tensor
) -> torch.Tensor:
    """Return the Riemannian U gradient ``g * (grad_W - U * grad_g)``.

    ``grad_g`` must already be the *global* row gradient (i.e. all-reduced
    across the TP group when the weight is column-sharded).
    """
    return g * (grad_w - u * grad_g)


@torch.compile
def _angular_muown_u_step_norm_sq(u_step: torch.Tensor) -> torch.Tensor:
    """Return the per-row squared norm of the proposed ``U`` step.

    This is the ``dim=-1`` reduction used by the retraction. It is a partial
    sum-of-squares under column tensor-parallel sharding (``partition_dim == 1``)
    and must be all-reduced across the TP group before the row-normalization in
    :func:`_angular_muown_u_recompose`; otherwise it is already the global
    squared norm. Kept separate from the recompose write so the all-reduce can
    run outside ``torch.compile``. Accepts ``(rows, cols)`` or a batched
    ``(B, rows, cols)`` stack.
    """
    return u_step.pow(2).sum(dim=-1, keepdim=True)


@torch.compile
def _angular_muown_u_recompose(
    w: torch.Tensor,
    g: torch.Tensor,
    u_step: torch.Tensor,
    u_step_norm_sq: torch.Tensor,
    eps: float = ANGULAR_MUOWN_EPS,
) -> None:
    """Normalize U rows and write W = diag(g) @ U in place.

    ``u_step_norm_sq`` is the (global) per-row squared norm of ``u_step`` from
    :func:`_angular_muown_u_step_norm_sq`, already all-reduced across the TP
    group for column-sharded weights.
    """
    u_step_norm = u_step_norm_sq.sqrt().clamp_min(eps)
    w.copy_(g * (u_step / u_step_norm))


def _is_expert_param(p) -> bool:
    """Routed-expert weights carry allreduce=False (expert-data-parallel gradient reduction)."""
    return getattr(p, "allreduce", True) is False


@torch.no_grad()
def _stable_rank_floor_(
    w: torch.Tensor, r_min: float, g: Optional[torch.Tensor] = None, iters: int = 6, rounds: int = 3
) -> int:
    """:func:`_spectral_bound_` with the stable-rank floor alone."""
    return _spectral_bound_(w, r_min=r_min, g=g, iters=iters, rounds=rounds)


@torch.no_grad()
def _spectral_bound_(
    w: torch.Tensor, r_min: float = 0.0, cap: float = 0.0, g: Optional[torch.Tensor] = None,
    iters: int = 6, rounds: int = 3, cap_iters: int = 10,
) -> int:
    """Bound the top of every matrix's spectrum, in place: ``r_min`` > 1 a stable-rank floor (a
    deflation of the top direction), then ``cap`` > 0 a spectral-norm cap sigma_max <= cap (the
    whole matrix scaled by min(1, cap / sigma_max): spectral normalization above the cap).

    The floor: hold every matrix of ``w`` ([m, n] or [B, m, n]) at stable rank |W|_F^2 / sigma_max^2 >=
    ``r_min``, in place, by deflating the top singular direction to the exact value
    c = sqrt((|W|_F^2 - sigma_max^2) / (r_min - 1)) at which the deflated matrix has stable rank
    r_min (repeated while a second direction is still above it). The rows aligned with that
    direction shrink: ``g`` (Muown's row magnitudes, [.., m, 1]) is set to the new row norms with
    its signs kept, so W = g U / |U| stays consistent -- restoring the old row norms instead
    re-inflated the direction (a collapsed matrix's rows are nearly parallel) and held a fast-
    collapsing matrix at about half the floor. sigma_max and its singular vectors come from a
    power iteration started at the column sums of |W| (deterministic on every replica; a
    collapsed matrix has a large spectral gap).

    The cap scales instead of deflating: a matrix whose whole spectrum grows has many directions
    above the cap, which a few rank-1 deflations per step do not bound. Its sigma_max comes from
    ``cap_iters`` power iterations (an underestimate at a small spectral gap: the bound is then
    slightly loose). Cost: per power iteration two batched matrix-vector products (memory-bound,
    a fraction of the Newton-Schulz step's matrix-matrix products).
    Returns the number of matrices bounded."""
    floor_on, cap_on = r_min > 1.0, cap > 0.0
    if not (floor_on or cap_on) or w.numel() == 0:
        return 0
    W = w.float().reshape(-1, w.shape[-2], w.shape[-1])
    capped_any = None
    for _ in range(rounds if floor_on else 0):
        fro2 = W.square().sum(dim=(-2, -1))
        v = W.abs().sum(dim=-2, keepdim=True).transpose(-2, -1)  # [B, n, 1]
        for _ in range(iters):
            v = W.transpose(-2, -1) @ (W @ v)
            v = v / v.norm(dim=-2, keepdim=True).clamp_min(1e-30)
        wv = W @ v
        sigma = wv.norm(dim=(-2, -1))
        over = sigma.square() * r_min > fro2
        if not bool(over.any()):
            break
        c = ((fro2 - sigma.square()).clamp_min(0) / (r_min - 1.0)).sqrt()
        u = wv / sigma.clamp_min(1e-30)[:, None, None]
        shrink = torch.where(over, sigma - c, torch.zeros_like(sigma))[:, None, None]
        W = W - shrink * (u @ v.transpose(-2, -1))
        capped_any = over if capped_any is None else (capped_any | over)
    if cap_on:
        v = W.abs().sum(dim=-2, keepdim=True).transpose(-2, -1)
        for _ in range(cap_iters):
            v = W.transpose(-2, -1) @ (W @ v)
            v = v / v.norm(dim=-2, keepdim=True).clamp_min(1e-30)
        sigma = (W @ v).norm(dim=(-2, -1))
        over = sigma > cap
        if bool(over.any()):
            W = W * (cap / sigma.clamp_min(1e-30)).clamp_max(1.0)[:, None, None]
            capped_any = over if capped_any is None else (capped_any | over)
    if capped_any is None:
        return 0
    w.copy_(W.reshape(w.shape).to(w.dtype))
    if g is not None:
        rows = w.float().norm(dim=-1, keepdim=True).clamp_min(ANGULAR_MUOWN_EPS)
        g.copy_(torch.copysign(rows.reshape(g.shape), g.float()).to(g.dtype))
    return int(capped_any.sum())


@torch.no_grad()
def _orthogonal_directions_(w: torch.Tensor, max_iters: int = 16, tol: float = 1e-5) -> int:
    """Muon-like weight balancing, in place: every matrix of ``w`` ([m, n] or [B, m, n]) becomes
    diag(|W_i|) Q_i / |Q_i| with Q the polar factor of its row-normalized U = W / |W_i| -- the row
    magnitudes kept, the direction part with all singular values 1 (Stiefel; exactly so when
    m <= n, where Q's rows are unit; a tall matrix's rows are renormalized). No direction can
    dominate (rank collapse is impossible) and the row norms, i.e. Muown's g, do not change.

    Q by the convergent cubic Newton-Schulz iteration X <- 1.5 X - 0.5 (X X^T) X in full fp32
    (Muon's quintic leaves singular values in ~[0.7, 1.2] and TF32 adds ~5e-4 relative noise:
    either would perturb the weights at every step), after normalizing by a power-iteration
    sigma_max (convergence needs sigma < sqrt(3); the small singular values then converge
    quadratically once near 1). It stops when every matrix's X X^T is within ``tol`` (rms per
    entry) of the identity -- X X^T is the iteration's own product, so the check is free but one
    host sync per iteration: applied every step, the weights stay within O(lr) of orthogonal and
    2-3 iterations (4-6 matmuls per matrix; Muon's 5 quintic steps: 15) suffice; a large step
    (or the init) takes more, up to ``max_iters``. Returns the number of matrices."""
    if w.numel() == 0:
        return 0
    m, n = w.shape[-2], w.shape[-1]
    W = w.float().reshape(-1, m, n)
    rows = W.norm(dim=-1, keepdim=True).clamp_min(ANGULAR_MUOWN_EPS)
    prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        X = W / rows
        if m > n:
            X = X.mT
        v = X.abs().sum(dim=-2, keepdim=True).transpose(-2, -1)
        for _ in range(4):
            v = X.transpose(-2, -1) @ (X @ v)
            v = v / v.norm(dim=-2, keepdim=True).clamp_min(1e-30)
        sigma = (X @ v).norm(dim=(-2, -1), keepdim=True)
        X = X / (1.05 * sigma).clamp_min(1e-30)  # sigma_max <~ 1 (a power-iteration underestimate)
        eye = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
        for _ in range(max_iters):
            A = X @ X.mT
            if float((A - eye).square().mean(dim=(-2, -1)).max().sqrt()) < tol:
                break
            X = torch.baddbmm(X, A, X, beta=1.5, alpha=-0.5)
    finally:
        torch.set_float32_matmul_precision(prec)
    if m > n:
        X = X.mT
    W = rows * X / X.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    w.copy_(W.reshape(w.shape).to(w.dtype))
    return W.shape[0]


def _batched_newton_schulz(
    x: torch.Tensor,
    steps: int,
    coefficient_type: str,
    tp_group: Optional["torch.distributed.ProcessGroup"] = None,
    partition_dim: int | None = None,
    tp_mode: Literal["duplicated", "distributed"] = "duplicated",
    eps: float = 1e-7,
) -> torch.Tensor:
    """Newton-Schulz orthogonalization of a ``(B, m, n)`` stack of same-shape matrices.

    Batched (``bmm``/``baddbmm``) equivalent of ``emerging_optimizers``'
    ``newton_schulz`` / ``newton_schulz_tp``: each matrix in the stack is
    orthogonalized independently, with exactly the same per-matrix math
    (per-matrix Frobenius normalization, the same coefficient sets, bf16
    compute under ``fp32_matmul_prec="medium"``). ``partition_dim`` refers to
    the *matrix* dims (0 = rows, 1 = cols), i.e. batch dims ``1``/``2``.

    Collectives are batched: ``duplicated`` mode does one all_gather for the
    whole stack (instead of one per matrix), ``distributed`` mode one Gram
    all-reduce per NS step for the whole stack.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected a (B, m, n) stack, got shape {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise ValueError(f"Input stack must be float32, got {x.dtype}")
    if coefficient_type not in _COEFFICIENT_SETS:
        raise ValueError(
            f"Unsupported coefficient type for the batched Newton-Schulz: {coefficient_type!r}"
        )

    tp_size = get_pg_size(tp_group) if tp_group is not None else 1
    is_tp = partition_dim is not None and tp_size > 1

    if is_tp and tp_mode == "duplicated":
        # One all_gather for the whole stack, then reassemble the full
        # matrices by concatenating rank shards along the sharded matrix dim.
        shards = [torch.empty_like(x) for _ in range(tp_size)]
        torch.distributed.all_gather(shards, x.contiguous(), group=tp_group)
        x = torch.cat(shards, dim=partition_dim + 1)

    # Whiten along the smaller dim (mirrors newton_schulz's transpose choice);
    # in distributed mode the sharded dim stays sharded, so the choice is
    # dictated by partition_dim exactly as in newton_schulz_tp.
    if is_tp and tp_mode == "distributed":
        transpose = partition_dim == 0
    else:
        transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT

    # Per-matrix spectral-norm bound via Frobenius normalization.
    if is_tp and tp_mode == "distributed":
        x_sq_sum = (x * x).sum(dim=(-2, -1), keepdim=True)
        torch.distributed.all_reduce(x_sq_sum, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        X = x / x_sq_sum.sqrt().clamp_min(eps)
        ns_group = tp_group
    else:
        X = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=eps)  # type: ignore[arg-type]
        ns_group = None

    if torch.get_float32_matmul_precision() == "medium":
        X = X.to(torch.bfloat16)

    iter_mode = "cycle" if coefficient_type != "polar_express" else "repeat_last"
    for a, b, c in get_coefficient_iterator(
        steps, _COEFFICIENT_SETS[coefficient_type], mode=iter_mode
    ):
        A = X @ X.mT
        if ns_group is not None:
            torch.distributed.all_reduce(A, op=torch.distributed.ReduceOp.SUM, group=ns_group)
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)

    X = X.to(torch.float32)

    if transpose:
        X = X.mT
    if is_tp and tp_mode == "duplicated":
        X = X.chunk(tp_size, dim=partition_dim + 1)[get_pg_rank(tp_group)]
    return X


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

    - ``scale_mode="shape_scaling"`` (default) gives ``sqrt(max(1, m / n))``,
      the original AngularMuown "ratio" scaling.
    - ``scale_mode="spectral"`` gives ``sqrt(max(m, n))``, matching
      the Muon recipe convention.
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
        qkv_split_shapes: Per-query-group fused QKV or Q-gate-KV split shapes.
        fp32_matmul_prec: Precision of matmuls in the orthogonalization.
        coefficient_type: Newton-Schulz coefficient set. ``"simple"`` matches the
            original AngularMuown ``newtonschulz5`` backend coefficients.
        num_ns_steps: Newton-Schulz iteration count.
        scale_mode: Shape-dependent scale mode (see above).
        extra_scale_factor: Additional scalar applied to the ``U`` direction.
        pg_collection: Process group collection for distributed training.
        tp_mode: Tensor parallel mode ("duplicated" or "distributed"). Both
            reproduce the single-GPU (global) update; they only differ in how the
            Newton-Schulz work is laid out (full-matrix on every rank vs sharded
            with per-step Gram-matrix all-reduces). The ``W = diag(g) @ U``
            decomposition adds three per-row (``dim=1``) reductions on top of the
            orthogonalization: the ``g`` seed, the row gradient ``grad_g``, and
            the ``u_step`` retraction norm. For a **column-sharded** weight
            (``partition_dim == 1``: RowParallelLinear ``linear_proj`` /
            ``linear_fc2`` / MoE down-proj) each rank only owns a column slice of
            every row, so these reductions are all-reduced across the
            tensor-parallel group; ``g``/``m_g``/``v_g`` stay replicated across
            column-shard ranks. For a **row-sharded** weight
            (``partition_dim == 0``: ColumnParallelLinear ``linear_qkv`` /
            ``linear_fc1`` / MoE up-proj) and for single-GPU, every ``dim=1``
            reduction is already global, so no all-reduce is performed.

            Muon's ``"blockwise"`` (local per-shard orthogonalization) is
            rejected: it would make the row geometry local to each column shard,
            which is a different optimizer than AngularMuown, not the intended
            behavior.
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
        batched_step: Stack same-shape parameters and run the update on the
            whole stack (batched Newton-Schulz via bmm, batched collectives,
            state gathered/scattered through scratch buffers). Same per-matrix
            math as the per-parameter path up to floating-point reduction
            order; dramatically fewer kernel launches, which is what bounds
            the step with many small (e.g. per-expert) weight matrices.
            Optimizer state layout is unchanged. Default True.

    Row magnitudes whose absolute value would fall below ``ANGULAR_MUOWN_EPS`` are
    projected back to magnitude ``ANGULAR_MUOWN_EPS`` with their proposed sign so the
    explicit ``W = diag(g) @ U`` coordinates stay well-defined.

    References:
        Florian Hübler, Kai Lion, Antonio Orvieto, Niao He.
        "Muown Implicitly Performs Angular Step-size Decay." arXiv:2606.23637, 2026.
        https://arxiv.org/abs/2606.23637

        Kai Lion, Florian Hübler, Bingcong Li, Antonio Orvieto, Niao He.
        "Muown: Row-Norm Control for Muon Optimization." arXiv:2605.10797, 2026.
        https://arxiv.org/abs/2605.10797
    """

    _row_state_keys = ANGULAR_MUOWN_ROW_STATE_KEYS
    # decoupled weight decay on Muown's unnormalized v (TensorParallelMuownClassical only)
    _v_weight_decay = 0.0
    # decoupled weight decay on the row magnitudes g (TensorParallelMuownClassical only)
    _g_weight_decay = 0.0
    # stable-rank floor of the expert matrices (TensorParallelMuownClassical only; 0 = off)
    _stable_rank_floor = 0.0
    _spectral_cap = 0.0
    _expert_orthogonal = False

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
        qkv_split_shapes: list[int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "simple",
        num_ns_steps: int = 5,
        scale_mode: str = "shape_scaling",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["duplicated", "distributed"] = "duplicated",
        u_decay_schedule: str = "poly",
        u_decay_scale: float | None = None,
        u_decay_p: float = 1.0,
        u_decay_warmup_steps: int = 0,
        u_decay_steps: int | None = None,
        u_decay_min_multiplier: float = 0.0,
        u_radius_lambda: float = 0.0,
        u_hyper_lr: float = 0.02,
        u_hyper_min_multiplier: float = 0.02,
        batched_step: bool = True,
        fixed_row_norm: float | None = None,
    ) -> None:
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if adam_eps < 0.0:
            raise ValueError(f"Invalid adam_eps: {adam_eps}")
        if tp_mode not in ("duplicated", "distributed"):
            raise ValueError(
                f"Invalid tp_mode for AngularMuown: {tp_mode!r}. 'blockwise' is not "
                "supported: local per-shard orthogonalization would make the per-row "
                "W = diag(g) @ U geometry local to each column shard, which is a "
                "different optimizer. Choose 'duplicated' or 'distributed'."
            )
        self._validate_u_decay(
            u_decay_schedule,
            u_decay_scale,
            u_decay_p,
            u_decay_warmup_steps,
            u_decay_steps,
            u_decay_min_multiplier,
        )
        if not math.isfinite(u_radius_lambda) or u_radius_lambda < 0.0:
            raise ValueError(f"Invalid u_radius_lambda: {u_radius_lambda}")
        if not math.isfinite(u_hyper_lr) or u_hyper_lr < 0.0:
            raise ValueError(f"Invalid u_hyper_lr: {u_hyper_lr}")
        if not 0.0 < u_hyper_min_multiplier <= 1.0:
            raise ValueError(f"Invalid u_hyper_min_multiplier: {u_hyper_min_multiplier}")

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
            batched_step=batched_step,
        )

        self._u_decay_schedule = u_decay_schedule
        self._u_decay_scale = None if u_decay_scale is None else float(u_decay_scale)
        self._u_decay_p = float(u_decay_p)
        # Pinned row magnitudes (nGPT-style normalized weights in Muown's own coordinates): every
        # row of a matrix starts at norm fixed_row_norm (W rescaled at the first step) and g is not
        # trained, so only the direction U moves. A weight normalizer applied to W instead would be
        # undone by every step, which rebuilds W = g U / |U| from the optimizer's own g.
        if fixed_row_norm is not None and not fixed_row_norm > 0.0:
            raise ValueError(f"fixed_row_norm must be positive, got {fixed_row_norm}")
        self._fixed_row_norm = None if fixed_row_norm is None else float(fixed_row_norm)
        self._u_decay_warmup_steps = int(u_decay_warmup_steps)
        self._u_decay_steps = None if u_decay_steps is None else int(u_decay_steps)
        self._u_decay_min_multiplier = float(u_decay_min_multiplier)
        self._u_radius_lambda = float(u_radius_lambda)
        self._u_hyper_lr = float(u_hyper_lr)
        self._u_hyper_log_min = math.log(float(u_hyper_min_multiplier))
        # hypergradient sums of the current step: <grad_u, m_u_prev>, |grad_u|^2, |m_u_prev|^2
        self._hyper_acc: Optional[torch.Tensor] = None

        # Scratch buffers for the batched step, keyed by (batch key, batch
        # size). Pure scratch — fully overwritten every step, never
        # checkpointed.
        self._batch_buffers: Dict[tuple, dict] = {}

        for group in self.param_groups:
            group.setdefault("betas", betas)
            group.setdefault("adam_eps", adam_eps)
            group.setdefault("u_lr_multiplier", 1.0)
            # the adaptive schedules' state, checkpointed with the param groups:
            # radius: rho^2 (kappa = 1 / rho); hyper: log kappa
            group.setdefault("u_radius_sq", 1.0)
            group.setdefault("u_log_kappa", 0.0)
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        "AngularMuown only supports 2D parameters, but found "
                        f"a parameter with size: {p.size()}"
                    )

    @staticmethod
    def _validate_u_decay(schedule, scale, p, warmup_steps, decay_steps, min_multiplier) -> None:
        """Validate the internal U-direction decay schedule arguments."""
        if schedule not in ("poly", "cosine", "radius", "hyper"):
            raise ValueError(
                f"Invalid u_decay_schedule: {schedule!r}. Choose from 'poly', 'cosine', "
                "'radius' or 'hyper'."
            )
        if not math.isfinite(p) or p < 0.0:
            raise ValueError(f"Invalid u_decay_p: {p}")
        if int(warmup_steps) != warmup_steps or warmup_steps < 0:
            raise ValueError(f"Invalid u_decay_warmup_steps: {warmup_steps}")
        if schedule in ("poly", "radius"):
            if scale is not None and (not math.isfinite(scale) or scale <= 0.0):
                raise ValueError(f"Invalid u_decay_scale: {scale}")
        elif schedule == "hyper":
            pass
        else:  # "cosine"
            if decay_steps is None:
                raise ValueError("u_decay_schedule='cosine' requires u_decay_steps")
            if int(decay_steps) != decay_steps or decay_steps <= 0:
                raise ValueError(f"Invalid u_decay_steps: {decay_steps}")
            if not math.isfinite(min_multiplier) or not 0.0 <= min_multiplier <= 1.0:
                raise ValueError(f"Invalid u_decay_min_multiplier: {min_multiplier}")

    @property
    def _adaptive_u(self) -> bool:
        """The radius / hyper schedules keep a state per step instead of a function of the step."""
        return self._u_decay_schedule in ("radius", "hyper")

    def _u_multiplier_for(self, group: dict, step: int) -> float:
        """The directional-step multiplier of this step: the adaptive schedules' current value
        (set once per optimizer step in :meth:`step`), else the step schedule."""
        if self._adaptive_u:
            return group["u_lr_multiplier"]
        return self._u_lr_multiplier(step)

    def _adaptive_u_before_step(self) -> None:
        """Set each group's multiplier from the adaptive state, before its parameters update.

        radius: kappa = 1 / rho with rho^2 the shared "radius" of the direction rows, as a
        normal network's weight norm: it grows with every step and shrinks with a weight-decay
        term (see :meth:`_adaptive_u_after_step`) -- one value for the whole model.
        hyper: kappa = exp(log_kappa), learnt from the gradient / previous-update correlation.
        """
        for group in self.param_groups:
            if self._u_decay_schedule == "radius":
                group["u_lr_multiplier"] = float(group["u_radius_sq"]) ** -0.5
            else:
                group["u_lr_multiplier"] = math.exp(float(group["u_log_kappa"]))
        if self._u_decay_schedule == "hyper":
            self._hyper_acc = None

    def _hyper_accumulate(self, grad_u: torch.Tensor, m_u_prev: torch.Tensor) -> None:
        """Add one parameter's (or batch's) <grad_u, m_u_prev>, |grad_u|^2, |m_u_prev|^2."""
        g, m = grad_u.float(), m_u_prev.float()
        sums = torch.stack([(g * m).sum(), (g * g).sum(), (m * m).sum()])
        self._hyper_acc = sums if self._hyper_acc is None else self._hyper_acc + sums

    def _adaptive_u_after_step(self) -> None:
        """Advance the adaptive state for the next step.

        radius: rho^2 <- (1 - 2 lr lambda) rho^2 + c (lr / max_lr)^2 with c = u_decay_scale.
        Driven by the step actually taken (lr from the external schedule): at lambda = 0 and a
        constant lr it is exactly the poly schedule at p = 1/2; with lambda > 0 it settles where
        growth and shrink balance, kappa* = sqrt(2 lr lambda / c), as a normal network's weight
        norm under weight decay; a cooldown (falling lr) slows both, so it does not decay twice.
        hyper: log kappa += hyper_lr * <grad_u, m_u_prev> / (|grad_u| |m_u_prev|), the sums over
        every matrix on every rank (one all-reduce of 3 numbers): updates that keep pointing the
        same way ask for larger steps, overshooting ones for smaller; clamped to
        [log u_hyper_min_multiplier, 0].
        """
        if self._u_decay_schedule == "radius":
            c = self._u_decay_scale if self._u_decay_scale is not None else 0.0
            for group in self.param_groups:
                lr = float(group["lr"])
                max_lr = float(group.get("max_lr") or lr or 1.0)
                rho_sq = float(group["u_radius_sq"])
                rho_sq = (1.0 - 2.0 * lr * self._u_radius_lambda) * rho_sq + c * (lr / max_lr) ** 2
                group["u_radius_sq"] = max(rho_sq, 1e-12)
            kappa = float(self.param_groups[0]["u_radius_sq"]) ** -0.5 if self.param_groups else 1.0
            ANGULAR_MUOWN_KAPPA["kappa"] = kappa
            return
        acc = self._hyper_acc
        if acc is None:
            acc = torch.zeros(3, device=torch.cuda.current_device() if torch.cuda.is_available()
                              else "cpu")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(acc, op=torch.distributed.ReduceOp.SUM)
        dot, gg, mm = acc.tolist()
        h = dot / math.sqrt(gg * mm) if gg > 0 and mm > 0 else 0.0
        for group in self.param_groups:
            log_k = float(group["u_log_kappa"]) + self._u_hyper_lr * h
            group["u_log_kappa"] = min(0.0, max(self._u_hyper_log_min, log_k))
        if self.param_groups:
            ANGULAR_MUOWN_KAPPA["kappa"] = math.exp(float(self.param_groups[0]["u_log_kappa"]))
        ANGULAR_MUOWN_KAPPA["hyper_corr"] = h

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

    def _u_step_size(self, group: dict, step: int) -> float:
        """Return the ordinary LR times the internal directional multiplier."""
        multiplier = self._u_multiplier_for(group, step)
        group["u_lr_multiplier"] = multiplier
        return group["lr"] * multiplier

    def _tp_row_reduce_group(self, p: torch.Tensor) -> Optional["torch.distributed.ProcessGroup"]:
        """Return the TP group to all-reduce per-row scalars over, or ``None``.

        The ``g`` seed, the row gradient ``grad_g``, and the ``u_step``
        retraction norm are ``dim=1`` reductions across the columns of each row.
        They are only *partial* (local to a rank) for **column-sharded** weights
        (``partition_dim == 1``, RowParallelLinear: ``linear_proj`` /
        ``linear_fc2`` / MoE down-proj), where each rank holds a column slice of
        every row. In that case they must be summed across the tensor-parallel
        group (mirroring the group selection in :meth:`orthogonalize`) so that
        ``g``/``m_g``/``v_g`` stay replicated and the update matches the
        single-GPU reference.

        Returns ``None`` (no all-reduce, the reduction is already global) when:

          - there is no process-group collection (single-GPU / no TP);
          - the weight is row-sharded or replicated (``partition_dim != 1``),
            where each rank already owns complete rows; or
          - the selected TP group has size 1.
        """
        if self.pg_collection is None:
            return None
        if getattr(p, "partition_dim", None) != 1:
            return None
        tp_group = (
            self.pg_collection.expt_tp if getattr(p, "expert_tp", False) else self.pg_collection.tp
        )
        if get_pg_size(tp_group) <= 1:
            return None
        return tp_group

    @torch.no_grad()  # type: ignore[misc]
    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        """Lazily initialize AngularMuown state (g, m_u, m_g, v_g, step) for 2D params."""
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue
            state = self.state[p]
            if len(state) != 0:
                continue
            if is_row_sharded(p):
                self._init_row_sharded_state(p, state)
                continue
            # Zero rows have no direction for U = W / g. Give just those rows the
            # expected row scale of the default linear init so the first U update
            # can create a unit direction without starting Adam's g near zero.
            # Seed from the sum-of-squares (not norm) so a column-sharded weight
            # can all-reduce the partial squared row norms before the sqrt,
            # yielding the global row magnitude replicated across TP ranks.
            w_norm_sq = p.detach().pow(2).sum(dim=1, keepdim=True)
            reduce_group = self._tp_row_reduce_group(p)
            if reduce_group is not None:
                torch.distributed.all_reduce(
                    w_norm_sq, op=torch.distributed.ReduceOp.SUM, group=reduce_group
                )
            w_norm = w_norm_sq.sqrt()
            zero_rows = w_norm <= ANGULAR_MUOWN_EPS
            if zero_rows.any():
                w_norm = torch.where(
                    zero_rows, w_norm.new_full(w_norm.shape, ANGULAR_MUOWN_ZERO_ROW_SCALE), w_norm
                )
            if self._fixed_row_norm is not None:
                p.mul_((self._fixed_row_norm / w_norm).to(p.dtype))
                w_norm = w_norm.new_full(w_norm.shape, self._fixed_row_norm)
            state["g"] = w_norm.clone()
            state["m_u"] = torch.zeros_like(p)
            state["m_g"] = torch.zeros_like(w_norm)
            state["v_g"] = torch.zeros_like(w_norm)
            state["step"] = 0

    def _init_row_sharded_state(self, p: torch.Tensor, state: dict) -> None:
        """The state of a Megatron-FSDP (row-sharded) matrix: every row lives on one rank, so the
        per-row seeds are local; the [rows, 1] states are DTensors with p's row sharding."""
        w_norm = local(p).detach().float().pow(2).sum(dim=1, keepdim=True).sqrt()
        zero_rows = w_norm <= ANGULAR_MUOWN_EPS
        if zero_rows.any():
            w_norm = torch.where(
                zero_rows, w_norm.new_full(w_norm.shape, ANGULAR_MUOWN_ZERO_ROW_SCALE), w_norm
            )
        if self._fixed_row_norm is not None:
            local(p).mul_((self._fixed_row_norm / w_norm).to(local(p).dtype))
            w_norm = w_norm.new_full(w_norm.shape, self._fixed_row_norm)
        w_norm = w_norm.to(local(p).dtype)
        state["g"] = row_state_like(p, w_norm.clone())
        state["m_u"] = torch.zeros_like(p)
        state["m_g"] = row_state_like(p, torch.zeros_like(w_norm))
        state["v_g"] = row_state_like(p, torch.zeros_like(w_norm))
        state["step"] = 0

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform one AngularMuown update on all 2D parameters with gradients."""
        if closure is not None:
            raise ValueError("closure is not supported")

        if self._adaptive_u:
            self._adaptive_u_before_step()
        for group in self.param_groups:
            if any(is_row_sharded(p) for p in group["params"]):
                # Megatron-FSDP: local rows, Newton-Schulz on owner ranks. A rank whose shard of
                # a matrix is empty has no grad for it, but still takes part in its exchange.
                self._init_group(group, skip_non_grad_params=False)
                self._step_group_row_sharded(group)
                continue
            self._init_group(group)
            has_gtp_remat = any(
                p.grad is not None and self._uses_gtp_remat(p) for p in group["params"]
            )
            if self.batched_step and not has_gtp_remat:
                self._step_group_batched(group)
            else:
                # Preserve the v0.19 LUMI GTP-remat gather/slice path.
                self._step_group(group)

        if self._adaptive_u:
            self._adaptive_u_after_step()
        return None

    def _step_group(self, group: dict) -> None:
        """Reference per-parameter update (one Newton-Schulz launch per weight)."""
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

            # For column-sharded weights (partition_dim == 1) the per-row
            # (dim=-1) reductions below are only partial on this rank and must
            # be summed across the TP group; None otherwise (already global
            # or an intentional local-shard variant).
            reduce_group = self._tp_row_reduce_group(p)

            u, grad_g = _angular_muown_u_and_grad_g(p, g, p.grad)
            if reduce_group is not None:
                torch.distributed.all_reduce(
                    grad_g, op=torch.distributed.ReduceOp.SUM, group=reduce_group
                )
            grad_u = _angular_muown_grad_u(u, g, p.grad, grad_g)
            v_norm = state.get("v_norm")
            if v_norm is not None:
                grad_u.div_(v_norm)
                u.mul_(v_norm)
                if self._v_weight_decay:  # decoupled decay of v: only its norm (the rotation rate)
                    u.mul_(1.0 - group["lr"] * self._v_weight_decay)

            # Momentum on the Riemannian U gradient (sum convention, as in
            # the original AngularMuown, not the EMA convention of the base class).
            if self._u_decay_schedule == "hyper":
                self._hyper_accumulate(grad_u, m_u)  # m_u: the previous step's momentum
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
            u_step = u.add(direction, alpha=-self._u_step_size(group, step - 1))

            # Adam update on the row magnitudes g, with sign-preserving
            # projection away from zero (pinned: g stays at fixed_row_norm).
            if self._fixed_row_norm is None:
                if self._g_weight_decay:  # decoupled: a soft cap at |g| ~ 1 / lambda
                    g.mul_(1.0 - lr * self._g_weight_decay)
                m_g.mul_(beta1).add_(grad_g, alpha=1 - beta1)
                v_g.mul_(beta2).addcmul_(grad_g, grad_g, value=1 - beta2)
                bc1 = 1 - beta1**step
                bc2 = 1 - beta2**step
                g_update = (m_g / bc1) / (v_g / bc2).sqrt().add_(adam_eps)
                g_candidate = g.add(g_update, alpha=-lr)
                g.copy_(torch.copysign(g_candidate.abs().clamp_min(ANGULAR_MUOWN_EPS), g_candidate))

            # Retraction: row-normalize the proposed U step. The row norm is
            # a dim=-1 reduction, so all-reduce the partial squared norms for
            # column-sharded weights before the sqrt inside recompose.
            u_step_norm_sq = _angular_muown_u_step_norm_sq(u_step)
            if reduce_group is not None:
                torch.distributed.all_reduce(
                    u_step_norm_sq, op=torch.distributed.ReduceOp.SUM, group=reduce_group
                )
            _angular_muown_u_recompose(p, g, u_step, u_step_norm_sq)
            if v_norm is not None:
                v_norm.copy_(u_step_norm_sq.sqrt().clamp_min(ANGULAR_MUOWN_EPS))
            if self._bound_on() and reduce_group is None and _is_expert_param(p):
                self._bound_(p.data, g=g)

    # ------------------------------------------------------------------
    # Batched step
    # ------------------------------------------------------------------
    #
    # The reference path above launches ~40 kernels + several torch.compile
    # dispatches per weight matrix; with hundreds of small (expert) matrices
    # per rank the optimizer step is completely CPU/launch-bound (measured
    # ~330 ms CPU dispatch vs ~44 ms GPU work per step on moonlight-1B at
    # TP1/EP4). The batched path groups same-shape parameters, stacks them
    # into (B, m, n) scratch buffers with a handful of _foreach_copy_ calls,
    # and performs the identical per-matrix math on the whole stack (the
    # Newton-Schulz becomes bmm/baddbmm, collectives are batched). Optimizer
    # state stays per-parameter (checkpoint layout unchanged); it is gathered
    # into and scattered back out of the scratch buffers each step. Batch
    # grouping (_batch_key_for) and the batched orthogonalization
    # (_orthogonalize_batched / _scaled_ns_batched) are inherited from
    # TensorParallelMuon.

    def _get_batch_buffers(
        self, key: tuple, batch_size: int, ref: torch.Tensor, cache: bool = True
    ) -> dict:
        """Return (allocating once) the stacked scratch buffers for a batch (``cache=False``:
        fresh ones, for batches that are alive at the same time)."""
        cache_key = (key, batch_size)
        bufs = self._batch_buffers.get(cache_key) if cache else None
        if bufs is None:
            rows, cols = ref.shape

            def _mk(c):
                buf = torch.empty((batch_size, rows, c), device=ref.device, dtype=torch.float32)
                return buf, list(buf.unbind(0))

            bufs = {}
            for name, c in (
                ("w", cols),
                ("grad", cols),
                ("m_u", cols),
                ("g", 1),
                ("m_g", 1),
                ("v_g", 1),
            ):
                bufs[name], bufs[name + "_views"] = _mk(c)
            if cache:
                self._batch_buffers[cache_key] = bufs
        return bufs

    def _step_group_batched(self, group: dict) -> None:
        """Batched update: same math as :meth:`_step_group`, one launch per stack."""
        # Group params by step semantics; the Adam bias corrections and the
        # u-decay multiplier depend on the state step count, so it is part of
        # the batch key (in practice all params share the same step).
        batches: Dict[tuple, dict] = {}
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            state["step"] += 1
            static_key, metadata = self._batch_key_for(p)
            key = static_key + (state["step"],)
            batch = batches.setdefault(
                key, {**metadata, "params": [], "static_key": static_key, "step": state["step"]}
            )
            batch["params"].append(p)
        for batch in batches.values():
            # Bound the stack: the batched step keeps ~8 fp32 copies of it (params, grads,
            # states, the update and the Newton-Schulz temporaries). Unbounded, every
            # same-shape matrix on the rank is one stack -- e.g. all routed-expert weights of
            # all layers -- which OOMs a MoE that otherwise fits (round 4 s2, 2026-10-18). Same
            # bound as Muown._STACK_NUMEL; the math is per matrix, so chunking changes nothing.
            params = batch["params"]
            per = max(1, self._STACK_NUMEL // max(1, params[0].numel()))
            for i in range(0, len(params), per):
                self._step_batch(group, {**batch, "params": params[i : i + per]})

    # Elements per stacked batch (bounds the transient fp32 copies of a batched step).
    _STACK_NUMEL = 1 << 26

    def _step_batch(self, group: dict, batch: dict) -> None:
        """Run one AngularMuown update on a stack of same-shape parameters."""
        ctx = self._batch_pre(group, batch)
        with eopt_utils.fp32_matmul_precision(self.fp32_matmul_prec):
            direction = self._orthogonalize_batched(ctx["update"], batch)
        self._batch_post(group, batch, ctx, direction)

    def _batch_pre(self, group: dict, batch: dict, cache: bool = True) -> dict:
        """The batched update up to the Newton-Schulz input ``ctx["update"]``. Row-sharded
        (Megatron-FSDP) parameters run on their local rows."""
        params = batch["params"]
        states = [self.state[p] for p in params]
        reduce_group = batch["reduce_group"]
        momentum = group["momentum"]

        bufs = self._get_batch_buffers(batch["static_key"], len(params), local(params[0]), cache)
        w_b, grad_b = bufs["w"], bufs["grad"]
        g_b, m_u_b = bufs["g"], bufs["m_u"]

        # Gather params, grads and state into the stacked scratch buffers.
        torch._foreach_copy_(bufs["w_views"], [local(p).detach() for p in params])
        torch._foreach_copy_(bufs["grad_views"], [_local_grad(p) for p in params])
        torch._foreach_copy_(bufs["g_views"], [local(s["g"]) for s in states])
        torch._foreach_copy_(bufs["m_u_views"], [local(s["m_u"]) for s in states])
        torch._foreach_copy_(bufs["m_g_views"], [local(s["m_g"]) for s in states])
        torch._foreach_copy_(bufs["v_g_views"], [local(s["v_g"]) for s in states])

        u, grad_g = _angular_muown_u_and_grad_g(w_b, g_b, grad_b)
        if reduce_group is not None:
            torch.distributed.all_reduce(
                grad_g, op=torch.distributed.ReduceOp.SUM, group=reduce_group
            )
        grad_u = _angular_muown_grad_u(u, g_b, grad_b, grad_g)

        # Muown retains ||v||: reconstruct v and apply the weight-normalization Jacobian.
        v_norm = bufs.get("v_norm")
        if v_norm is not None:
            torch._foreach_copy_(bufs["v_norm_views"], [local(state["v_norm"]) for state in states])
            grad_u.div_(v_norm)
            u.mul_(v_norm)
            if self._v_weight_decay:
                u.mul_(1.0 - group["lr"] * self._v_weight_decay)

        if self._u_decay_schedule == "hyper":
            self._hyper_accumulate(grad_u, m_u_b)  # m_u_b: the previous step's momentum
        m_u_b.mul_(momentum).add_(grad_u)
        if self.nesterov:
            update = grad_u.add(m_u_b, alpha=momentum)
        else:
            # No aliasing concern: m_u_b is a scratch copy and the NS below
            # does not mutate its input.
            update = m_u_b
        return {"bufs": bufs, "u": u, "grad_g": grad_g, "update": update}

    def _batch_post(self, group: dict, batch: dict, ctx: dict, direction: torch.Tensor,
                    floor: bool = True) -> None:
        """The batched update from the orthogonalized direction on. ``floor=False``: the
        stable-rank floor is left to the caller (row-sharded: it needs the full matrices)."""
        params = batch["params"]
        states = [self.state[p] for p in params]
        reduce_group = batch["reduce_group"]
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        adam_eps = group["adam_eps"]
        step = batch["step"]
        bufs, u, grad_g = ctx["bufs"], ctx["u"], ctx["grad_g"]
        w_b, g_b, m_g_b, v_g_b = bufs["w"], bufs["g"], bufs["m_g"], bufs["v_g"]
        v_norm = bufs.get("v_norm")

        u_step = u.add(direction, alpha=-self._u_step_size(group, step - 1))

        if self._fixed_row_norm is None:  # pinned: g stays at fixed_row_norm
            if self._g_weight_decay:  # decoupled: a soft cap at |g| ~ 1 / lambda
                g_b.mul_(1.0 - lr * self._g_weight_decay)
            m_g_b.mul_(beta1).add_(grad_g, alpha=1 - beta1)
            v_g_b.mul_(beta2).addcmul_(grad_g, grad_g, value=1 - beta2)
            bc1 = 1 - beta1**step
            bc2 = 1 - beta2**step
            g_update = (m_g_b / bc1) / (v_g_b / bc2).sqrt().add_(adam_eps)
            g_candidate = g_b.add(g_update, alpha=-lr)
            g_b.copy_(torch.copysign(g_candidate.abs().clamp_min(ANGULAR_MUOWN_EPS), g_candidate))

        u_step_norm_sq = _angular_muown_u_step_norm_sq(u_step)
        if reduce_group is not None:
            torch.distributed.all_reduce(
                u_step_norm_sq, op=torch.distributed.ReduceOp.SUM, group=reduce_group
            )
        _angular_muown_u_recompose(w_b, g_b, u_step, u_step_norm_sq)
        if floor and self._bound_on() and reduce_group is None:
            experts = [i for i, p in enumerate(params) if _is_expert_param(p)]
            if len(experts) == len(params):
                self._bound_(w_b, g=g_b)
            elif experts:
                idx = torch.tensor(experts, device=w_b.device)
                sub, sub_g = w_b.index_select(0, idx), g_b.index_select(0, idx)
                self._bound_(sub, g=sub_g)
                w_b.index_copy_(0, idx, sub)
                g_b.index_copy_(0, idx, sub_g)
        if v_norm is not None:
            v_norm.copy_(u_step_norm_sq.sqrt().clamp_min(ANGULAR_MUOWN_EPS))
            torch._foreach_copy_([local(state["v_norm"]) for state in states],
                                 bufs["v_norm_views"])

        # Scatter updated weights and state back to the per-param tensors.
        torch._foreach_copy_([local(p).detach() for p in params], bufs["w_views"])
        torch._foreach_copy_([local(s["g"]) for s in states], bufs["g_views"])
        torch._foreach_copy_([local(s["m_u"]) for s in states], bufs["m_u_views"])
        torch._foreach_copy_([local(s["m_g"]) for s in states], bufs["m_g_views"])
        torch._foreach_copy_([local(s["v_g"]) for s in states], bufs["v_g_views"])

    # Elements (over all ranks) per row-sharded round: bounds the live scratch of a step.
    _ROW_SHARDED_ROUND_NUMEL = int(os.environ.get("MUON_FSDP_STEP_ROUND_NUMEL", str(1 << 30)))

    def _step_group_row_sharded(self, group: dict) -> None:
        """Megatron-FSDP: the batched math on each rank's rows; the Newton-Schulz inputs go to
        their matrices' owner ranks and come back orthogonalized (fsdp_row_exchange). Rounds of
        consecutive parameters (in the group's order, the same on every rank) bound the scratch."""
        if self._uses_gtp_remat_any(group):
            raise NotImplementedError("GTP remat with Megatron-FSDP")
        # requires_grad, not p.grad: the parameter list must be the same on every rank
        params = [p for p in group["params"] if p.requires_grad]
        start = 0
        while start < len(params):
            end, total = start, 0
            while end < len(params) and (
                end == start or total + params[end].numel() <= self._ROW_SHARDED_ROUND_NUMEL
            ):
                total += params[end].numel()
                end += 1
            self._row_sharded_round(group, params[start:end])
            start = end

    def _uses_gtp_remat_any(self, group: dict) -> bool:
        return any(p.grad is not None and self._uses_gtp_remat(p) for p in group["params"])

    def _row_sharded_round(self, group: dict, params: list) -> None:
        batches: Dict[tuple, dict] = {}
        metas = []
        for p in params:
            state = self.state[p]
            state["step"] += 1
            static_key, metadata = self._batch_key_for(p)
            metas.append((static_key, metadata))
            local_key = static_key + (tuple(local(p).shape),)
            batch = batches.setdefault(
                local_key + (state["step"],),
                {**metadata, "params": [], "static_key": local_key, "step": state["step"]},
            )
            batch["params"].append(p)
        rows_of = {}
        ctxs = []
        for batch in batches.values():
            ctx = self._batch_pre(group, batch, cache=False)
            ctxs.append((batch, ctx))
            for j, p in enumerate(batch["params"]):
                rows_of[id(p)] = ctx["update"][j]

        def ns(stack, metadata):
            with eopt_utils.fp32_matmul_precision(self.fp32_matmul_prec):
                return self._orthogonalize_batched(stack, metadata)

        directions = orthogonalize_rows(params, [rows_of[id(p)] for p in params], metas, ns)
        direction_of = {id(p): d for p, d in zip(params, directions)}
        for batch, ctx in ctxs:
            stack = torch.stack([direction_of[id(p)] for p in batch["params"]])
            self._batch_post(group, batch, ctx, stack, floor=False)
        if self._bound_on():
            self._row_sharded_floor(params)

    def _bound_on(self) -> bool:
        return self._stable_rank_floor > 1.0 or self._spectral_cap > 0.0 or self._expert_orthogonal

    def _expert_cap(self, shape) -> float:
        """sigma_max cap of an expert matrix of (full) ``shape``: spectral_cap x its init value."""
        if self._spectral_cap <= 0.0:
            return 0.0
        return self._spectral_cap * self._spectral_ref[tuple(int(d) for d in shape[-2:])]

    def _bound_(self, w: torch.Tensor, g: Optional[torch.Tensor] = None) -> int:
        if self._expert_orthogonal:  # the row norms (g) are kept
            return _orthogonal_directions_(w)
        return _spectral_bound_(w, r_min=self._stable_rank_floor,
                                cap=self._expert_cap(w.shape), g=g)

    @torch.no_grad()
    def _init_spectral_refs(self) -> None:
        """The init sigma_max of the routed-expert matrices, per full shape: rms * (sqrt(m) +
        sqrt(n)), exact for i.i.d. init (the optimizer is built before a checkpoint loads, so this
        is the init on a resume too). Sums of squares over local rows, all-reduced: no rank needs a
        full matrix."""
        stats: Dict[tuple, list] = {}
        for group in self.param_groups:
            for p in group["params"]:
                if _is_expert_param(p) and p.ndim == 2:
                    lw = local(p).detach().float()
                    acc = stats.setdefault(tuple(int(d) for d in p.shape), [0.0, 0])
                    acc[0] += float(lw.square().sum())
                    acc[1] += lw.numel()
        shapes = sorted(stats)
        dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        if dist:
            every = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(every, shapes)
            shapes = sorted({s for ss in every for s in ss})
            dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() \
                else torch.device("cpu")
            t = torch.tensor([stats.get(s, [0.0, 0])[i] for s in shapes for i in (0, 1)],
                             dtype=torch.float64, device=dev)
            torch.distributed.all_reduce(t)
            vals = t.view(-1, 2).tolist()
        else:
            vals = [stats[s] for s in shapes]
        self._spectral_ref = {
            s: math.sqrt(sq / max(n, 1)) * (math.sqrt(s[0]) + math.sqrt(s[1]))
            for s, (sq, n) in zip(shapes, vals)
        }

    def _row_sharded_floor(self, params: list) -> None:
        """The stable-rank floor on row-sharded expert matrices: a matrix's stable rank is not
        its rows', so the rows go to the matrix's owner rank (as the Newton-Schulz inputs do),
        are floored there and come back; g follows the new local row norms."""
        experts = [p for p in params if _is_expert_param(p)]  # the same list on every rank
        if not experts:
            return

        def floor_fn(stack, metadata):  # returns the change: exactly 0 where nothing is capped
            floored = stack.clone()
            self._bound_(floored)
            return floored.sub_(stack)

        rows = [local(p).detach().float() for p in experts]
        deltas = orthogonalize_rows(experts, rows, [self._batch_key_for(p) for p in experts],
                                    floor_fn)
        for p, d in zip(experts, deltas):
            lw = local(p).detach()
            if lw.numel() == 0 or not bool(d.any()):
                continue
            lw.add_(d.to(lw.dtype))
            g = local(self.state[p]["g"])
            norms = lw.float().norm(dim=-1, keepdim=True).clamp_min(ANGULAR_MUOWN_EPS)
            g.copy_(torch.copysign(norms.reshape(g.shape), g.float()).to(g.dtype))

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

        The expanded states are built on the host: three weight-sized fp32 tensors per matrix
        on the GPU ran an s2 MoE (5.7B parameters) out of memory while LOADING a checkpoint
        (the loading templates come from this method, on top of the live state); on the host
        a save copies only the ``(rows, 1)`` vectors, and the loaded values are collapsed
        there before ``load_state_dict`` moves them to the parameter's device.
        """
        state_dict = super().state_dict()
        packed_state = state_dict.get("state")
        if not packed_state or _COMPACT_ROW_STATES:
            # compact: the (rows, 1) states as they are -- the optimizer-state sharding maps them
            # onto the weight's metadata with the last axis collapsed (_row_state_sharded_tensor)
            return state_dict

        expanded_state = {}
        for param_id, param_state in packed_state.items():
            new_param_state = dict(param_state)
            m_u = param_state.get("m_u")
            if m_u is not None and m_u.dim() == 2 and m_u.shape[1] > 1:
                cols = m_u.shape[1]
                for key in self._row_state_keys:
                    tensor = param_state.get(key)
                    if tensor is not None and tensor.dim() == 2 and tensor.shape[1] == 1:
                        new_param_state[key] = tensor.detach().cpu().expand(-1, cols).contiguous()
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
            retains_norm = "v_norm" in self._row_state_keys
            if any(("v_norm" in state) != retains_norm for state in packed_state.values() if state):
                raise ValueError("Muown and AngularMuown optimizer states are incompatible")
            collapsed_state = {}
            for param_id, param_state in packed_state.items():
                new_param_state = dict(param_state)
                for key in self._row_state_keys:
                    tensor = param_state.get(key)
                    if tensor is not None and tensor.dim() == 2 and tensor.shape[1] > 1:
                        new_param_state[key] = tensor[:, :1].contiguous()
                collapsed_state[param_id] = new_param_state
            state_dict = dict(state_dict)
            state_dict["state"] = collapsed_state

        super().load_state_dict(state_dict)


class TensorParallelAngularMuownSimplified(TensorParallelAngularMuown):
    """AngularMuown with independent directional and radial learning rates.

    The training scheduler supplies ``u_lr`` using C/t with warmup and cooldown.
    All tensor updates, parallelism and row-state checkpoint conversion are
    inherited. Direction scaling is selected by ``scale_mode`` (shape_scaling
    by default), with no additional scalar factor or internal decay.

    Args:
        params: Iterable of 2D parameters or parameter-group dictionaries.
        u_lr: Absolute directional rate, before the selected shape-dependent scaling.
        lr: Adam learning rate for row magnitudes. Remaining arguments have the
            same meaning as in :class:`TensorParallelAngularMuown`.
    """

    def __init__(
        self,
        params: ParamsT,
        *,
        u_lr: float,
        lr: float = 1e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: list[int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "simple",
        num_ns_steps: int = 5,
        scale_mode: str = "shape_scaling",
        pg_collection: ProcessGroupCollection | None = None,
        tp_mode: Literal["duplicated", "distributed"] = "duplicated",
        batched_step: bool = True,
    ) -> None:
        if not math.isfinite(u_lr) or u_lr < 0.0:
            raise ValueError("u_lr must be finite and nonnegative")
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            betas=betas,
            adam_eps=adam_eps,
            split_qkv=split_qkv,
            is_qkv_fn=is_qkv_fn,
            qkv_split_shapes=qkv_split_shapes,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=1.0,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
            batched_step=batched_step,
        )
        self.defaults["u_lr"] = u_lr
        for group in self.param_groups:
            group.setdefault("u_lr", u_lr)
            group.pop("u_lr_multiplier", None)

    def _u_step_size(self, group: dict, step: int) -> float:
        """Return only the independently scheduled directional rate."""
        u_lr = group["u_lr"]
        if not math.isfinite(u_lr) or u_lr < 0.0:
            raise ValueError("u_lr must be finite and nonnegative")
        return u_lr


class TensorParallelMuownClassical(TensorParallelAngularMuown):
    """Classical Muown in W = g * v / ||v|| coordinates, initially v = W.

    Retain ||v|| instead of retracting v to unit norm. Reconstruct v from W/g
    and its saved row norm; its gradient is grad_v = grad_U / ||v||. Adam
    updates g, while a Muon-style update steps v using the ordinary LR, with
    no additional angular schedule. Matrix weight decay remains disabled.
    Shared direction, magnitude and parallelism settings are inherited from AngularMuown.
    """

    _row_state_keys = (*ANGULAR_MUOWN_ROW_STATE_KEYS, "v_norm")

    def __init__(
        self, params: ParamsT, v_weight_decay: float = 0.0, g_weight_decay: float = 0.0,
        stable_rank_floor: float = 0.0, spectral_cap: float = 0.0,
        expert_orthogonal: bool = False, **kwargs
    ) -> None:
        """``v_weight_decay`` (not in the Muown paper, default off): decoupled weight decay on v,
        v <- v - lr (update + lambda v). v / ||v|| does not depend on ||v||, so it only shrinks the
        norm and so raises the rotation per step: instead of the endless ~t^-1/2 annealing, the
        rate settles at ~sqrt(2 lr lambda) (the equilibrium of Muon / AdamW with weight decay) after
        ~1 / (2 lr lambda) steps. Directions and magnitudes are otherwise updated as without it.

        ``g_weight_decay`` (default off): decoupled weight decay on the row magnitudes g,
        g <- g (1 - lr lambda) before their Adam step. Adam moves g by ~lr per step whatever the
        gradient's size, so without it g drifts without bound along a persistent gradient sign; the
        decay balances that push at |g| ~ 1 / lambda and is negligible for rows well below it.

        ``stable_rank_floor`` (default off): after each step, the routed-expert matrices
        (allreduce=False) are held at stable rank |W|_F^2 / sigma_max^2 >= this value by a rank-1
        deflation of their top singular direction at fixed row norms. A runaway expert collapses
        its fc1 / fc2 spectra while every norm stays flat (stable rank down to single digits
        against ~200-300); a floor of ~30-50 never touches a normal expert.

        ``spectral_cap`` (default off): after each step, the routed-expert matrices are held at
        sigma_max <= this factor x their init sigma_max (per shape, from the init weights) by the
        same deflation; unlike the floor it also bounds a matrix whose norms grow.

        ``expert_orthogonal`` (default off): after each step, the routed-expert matrices'
        direction parts are replaced by their polar factor at kept row norms (Muon-like weight
        balancing: every singular value of U is 1); exclusive with the floor and the cap."""
        for name in kwargs:
            if name.startswith("u_") or name in {"weight_decay", "use_decoupled_weight_decay"}:
                raise ValueError(f"Muown does not support {name}")
        if not (math.isfinite(v_weight_decay) and v_weight_decay >= 0.0):
            raise ValueError("v_weight_decay must be finite and nonnegative")
        if not (math.isfinite(g_weight_decay) and g_weight_decay >= 0.0):
            raise ValueError("g_weight_decay must be finite and nonnegative")
        if not (math.isfinite(stable_rank_floor) and stable_rank_floor >= 0.0):
            raise ValueError("stable_rank_floor must be finite and nonnegative")
        if not (math.isfinite(spectral_cap) and spectral_cap >= 0.0):
            raise ValueError("spectral_cap must be finite and nonnegative")
        if expert_orthogonal and (stable_rank_floor > 1.0 or spectral_cap > 0.0):
            raise ValueError("expert_orthogonal excludes stable_rank_floor and spectral_cap")
        super().__init__(params, **kwargs)
        self._v_weight_decay = float(v_weight_decay)
        self._g_weight_decay = float(g_weight_decay)
        self._stable_rank_floor = float(stable_rank_floor)
        self._spectral_cap = float(spectral_cap)
        self._expert_orthogonal = bool(expert_orthogonal)
        self._spectral_ref: Dict[tuple, float] = {}
        if self._spectral_cap > 0.0:
            self._init_spectral_refs()
        for group in self.param_groups:
            if "u_lr" in group:
                raise ValueError("Muown uses the ordinary lr, not u_lr")
            group.pop("u_lr_multiplier", None)

    @torch.no_grad()
    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        fresh = [p for p in group["params"] if not self.state[p]]
        super()._init_group(group, skip_non_grad_params)
        for p in fresh:
            if self.state[p]:
                self.state[p]["v_norm"] = self.state[p]["g"].clone()

    def _get_batch_buffers(
        self, key: tuple, batch_size: int, ref: torch.Tensor, cache: bool = True
    ) -> dict:
        bufs = super()._get_batch_buffers(key, batch_size, ref, cache)
        if "v_norm" not in bufs:
            bufs["v_norm"] = torch.empty_like(bufs["g"])
            bufs["v_norm_views"] = list(bufs["v_norm"].unbind(0))
        return bufs

    def _u_step_size(self, group: dict, step: int) -> float:
        return group["lr"]


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
    # AdaptiveMuon uses its upstream step implementation and has no batched path.
    kwargs.pop("batched_step", None)
    kwargs.update(_kwargs_from_config(TensorParallelAdaptiveMuon, "adaptive_muon", config))
    return kwargs


def _angular_muown_config_to_kwargs(
    config, model_chunks, pg_collection, optimizer_cls=TensorParallelAngularMuown
) -> Dict[str, Any]:
    """Convert shared AngularMuown settings to the selected constructor kwargs."""
    kwargs = _kwargs_from_config(optimizer_cls, "angular_muown", config)
    kwargs["betas"] = (config.angular_muown_beta1, config.angular_muown_beta2)
    kwargs["is_qkv_fn"] = lambda p: getattr(p, "is_qkv", False)
    kwargs["qkv_split_shapes"] = _get_qkv_split_shapes(model_chunks[0].config)
    kwargs["pg_collection"] = pg_collection
    return kwargs


def _muown_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to Muown constructor kwargs.

    Muown orthogonalizes only the *tangential* part of the update, so it reuses the
    tensor-parallel Newton-Schulz from ``emerging_optimizers`` when that package is
    available; otherwise :mod:`megatron.core.optimizer.muown` falls back to its own
    single-process implementation.
    """
    kwargs = _kwargs_from_config(Muown, "muown", config)
    kwargs.pop("orthogonalize_fn", None)

    if HAVE_EMERGING_OPTIMIZERS and pg_collection is not None:

        def _tp_orthogonalize(grad, steps=5, param=None, **_):
            tp_group = (
                pg_collection.expt_tp if getattr(param, 'expert_tp', False) else pg_collection.tp
            )
            partition_dim = getattr(param, "partition_dim", None)
            if partition_dim == -1:
                partition_dim = None
            return newton_schulz_tp(
                grad,
                steps=steps,
                coefficient_type=config.muon_coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                tp_mode="duplicated",
            )

        kwargs["orthogonalize_fn"] = _tp_orthogonalize

    if pg_collection is not None:

        def _tp_group(param):
            return pg_collection.expt_tp if getattr(param, 'expert_tp', False) else pg_collection.tp

        if HAVE_EMERGING_OPTIMIZERS and config.muon_coefficient_type in _COEFFICIENT_SETS:

            def _batched_orthogonalize(stack, steps=5, param=None, **_):
                partition_dim = getattr(param, "partition_dim", None)
                if partition_dim == -1:
                    partition_dim = None
                return _batched_newton_schulz(
                    stack,
                    steps=steps,
                    coefficient_type=config.muon_coefficient_type,
                    tp_group=_tp_group(param),
                    partition_dim=partition_dim,
                    tp_mode="duplicated",
                )

            def _bucket_key(param):
                # Stacked parameters share one all_gather / row all-reduce, so they
                # must agree on shape, dtype, process group and sharded dim.
                return (
                    tuple(param.shape),
                    param.dtype,
                    bool(getattr(param, "expert_tp", False)),
                    getattr(param, "partition_dim", None),
                    bool(getattr(param, "tensor_model_parallel", False)),
                )

            kwargs["batched_orthogonalize_fn"] = _batched_orthogonalize
            kwargs["bucket_key_fn"] = _bucket_key

        def _row_sum(partial, param):
            # Column-sharded weights (RowParallelLinear: linear_proj / linear_fc2 / MoE
            # down-proj) hold a column slice of every row.
            group = _tp_group(param)
            if getattr(param, "partition_dim", None) != 1 or get_pg_size(group) <= 1:
                return partial
            partial = partial.contiguous()
            torch.distributed.all_reduce(partial, group=group)
            return partial

        def _tp_shape(param):
            rows, cols = param.shape[-2:]
            size = get_pg_size(_tp_group(param))
            if size > 1 and getattr(param, "tensor_model_parallel", False):
                if getattr(param, "partition_dim", None) == 0:
                    rows *= size
                elif getattr(param, "partition_dim", None) == 1:
                    cols *= size
            return rows, cols

        kwargs["row_sum_fn"] = _row_sum
        kwargs["tp_shape_fn"] = _tp_shape
    return kwargs


def _angular_muown_simplified_config_to_kwargs(
    config, model_chunks, pg_collection
) -> Dict[str, Any]:
    """Build the C/t variant using only its supported constructor settings."""
    kwargs = _angular_muown_config_to_kwargs(
        config, model_chunks, pg_collection, TensorParallelAngularMuownSimplified
    )
    # The training scheduler supplies the initial angular rate before the first update.
    kwargs["u_lr"] = 0.0
    return kwargs


def _muown_classical_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Reuse AngularMuown settings without its internal angular decay (every u_* schedule option:
    the poly / cosine decay and our radius / hyper schedules)."""
    kwargs = _angular_muown_config_to_kwargs(config, model_chunks, pg_collection)
    kwargs = {name: value for name, value in kwargs.items() if not name.startswith("u_")}
    kwargs["v_weight_decay"] = getattr(config, "muown_classical_v_weight_decay", 0.0)
    kwargs["g_weight_decay"] = getattr(config, "muown_classical_g_weight_decay", 0.0)
    kwargs["stable_rank_floor"] = getattr(config, "muown_expert_stable_rank_floor", 0.0)
    kwargs["spectral_cap"] = getattr(config, "muown_expert_spectral_cap", 0.0)
    kwargs["expert_orthogonal"] = getattr(config, "muown_expert_orthogonal", False)
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
                ): {'optimizer': 'adam'},
                # Router/gate matrices are small and sensitive, so keep them on Adam.
                ParamKey(name=('*router.weight', '*gate_weight')): {'optimizer': 'adam'},
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
                # Router/gate matrices are small and sensitive, so keep them on Adam.
                ParamKey(name=('*router.weight', '*gate_weight')): {'optimizer': 'adam'},
            },
        ),
        "muown": EmergingOptimizerEntry(
            optimizer_cls=Muown,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_muown_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'},
                ParamKey(name=('*router.weight', '*gate_weight')): {'optimizer': 'adam'},
            },
        ),
    }
)

_EMERGING_OPTIMIZERS["angular_muown_simplified"] = EmergingOptimizerEntry(
    optimizer_cls=TensorParallelAngularMuownSimplified,
    config_to_kwargs=_angular_muown_simplified_config_to_kwargs,
    # The optimizer factory copies these overrides before applying them.
    default_param_overrides=_EMERGING_OPTIMIZERS["angular_muown"].default_param_overrides,
)

_EMERGING_OPTIMIZERS["muown_classical"] = EmergingOptimizerEntry(
    optimizer_cls=TensorParallelMuownClassical,
    config_to_kwargs=_muown_classical_config_to_kwargs,
    default_param_overrides=_EMERGING_OPTIMIZERS["angular_muown"].default_param_overrides,
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
