# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in MoE router-health and expert-viability diagnostics.

Ported from OELLM ``feat/v0.16-moe-expert-viability-diagnostics`` onto Megatron
Core 0.19. These trackers are separate from :mod:`moe_logging` because they store
``[num_layers, num_experts]`` tensors, not per-layer scalars.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import is_graph_capturing

# A deliberately internal threshold: it is a diagnostic definition, not a training knob.
EXPERT_COLLAPSE_RMS_FRACTION = 0.10

_MOE_EXPERT_UTILIZATION_TRACKER: dict = {}
_MOE_ROUTER_STATS_TRACKER: dict = {}
_MOE_EXPERT_VIABILITY_TRACKER: dict = {}
_MASKED_ROUTED_MOE_LAYER: Optional[int] = None


def _dp_group(pg_collection: Optional[ProcessGroupCollection]):
    """Match :class:`MoEMetricsTracker` DP averaging (include GTP-remat peers)."""
    if pg_collection is None:
        return parallel_state.get_data_parallel_group(
            with_context_parallel=False, partial_data_parallel=False
        )
    dp_group = getattr(pg_collection, "dp_cp_gtp_remat", None)
    if dp_group is None:
        dp_group = getattr(pg_collection, "dp", None)
    if dp_group is None:
        dp_group = parallel_state.get_data_parallel_group(
            with_context_parallel=False, partial_data_parallel=False
        )
    return dp_group


def _pp_group(pg_collection: Optional[ProcessGroupCollection]):
    if pg_collection is None:
        return parallel_state.get_pipeline_model_parallel_group()
    return pg_collection.pp


def _tp_cp_group(pg_collection: Optional[ProcessGroupCollection]):
    if pg_collection is None:
        return parallel_state.get_tensor_and_context_parallel_group()
    return pg_collection.tp_cp


def _ep_group(pg_collection: Optional[ProcessGroupCollection]):
    if pg_collection is None:
        return parallel_state.get_expert_model_parallel_group()
    return pg_collection.ep


@contextmanager
def mask_routed_moe_layer(layer_number: int):
    """Temporarily zero only the routed path of one physical MoE layer."""
    global _MASKED_ROUTED_MOE_LAYER
    previous = _MASKED_ROUTED_MOE_LAYER
    _MASKED_ROUTED_MOE_LAYER = layer_number
    try:
        yield
    finally:
        _MASKED_ROUTED_MOE_LAYER = previous


def should_mask_routed_moe_layer(layer_number: Optional[int]) -> bool:
    """Whether the supplied MoE layer is selected by the temporary mask context."""
    return layer_number is not None and layer_number == _MASKED_ROUTED_MOE_LAYER


class _CachingDataIterator:
    """Wrap a data iterator and replay the microbatches it has already yielded."""

    def __init__(self, data_iterator):
        self._source = data_iterator
        self._cache: list = []
        self._replay = False
        self._pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._replay:
            if self._pos >= len(self._cache):
                raise StopIteration
            item = self._cache[self._pos]
            self._pos += 1
            return item
        item = next(self._source)
        self._cache.append(item)
        return item

    def rewind(self) -> None:
        self._replay = True
        self._pos = 0


def collect_moe_layer_numbers(model_chunks) -> List[int]:
    """Return sorted 1-indexed physical layer numbers for local MoE layers."""
    from megatron.core.transformer.moe.moe_layer import MoELayer

    layer_numbers = []
    for chunk in model_chunks:
        for module in chunk.modules():
            if isinstance(module, MoELayer) and module.layer_number is not None:
                layer_numbers.append(module.layer_number)
    return sorted(set(layer_numbers))


def _aggregate_layer_values(values: List[float]) -> dict[str, float]:
    """Return mean/min/max aggregates over per-layer scalar metrics."""
    if not values:
        return {}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def compute_router_score_distribution(
    logits: torch.Tensor, score_function: str, expert_bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Return a normalized full-expert distribution for router diagnostics."""
    logits = logits.float()
    if score_function == "softmax":
        return torch.softmax(logits, dim=-1)
    if score_function == "sigmoid":
        scores = torch.sigmoid(logits)
        if expert_bias is not None:
            scores = (scores + expert_bias.float()).clamp_min(0.0)
        return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    raise ValueError(f"Invalid score_function: {score_function}")


def compute_normalized_entropy(distribution: torch.Tensor) -> torch.Tensor:
    """Compute entropy over the last dimension, normalized to the range [0, 1]."""
    num_categories = distribution.shape[-1]
    entropy = -(distribution * distribution.clamp(min=1e-12).log()).sum(dim=-1)
    if num_categories > 1:
        return entropy / math.log(num_categories)
    return torch.ones_like(entropy)


def compute_expert_load_metrics(
    tokens_per_expert: torch.Tensor, near_dead_uniform_fraction: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute expert-load health metrics from raw per-expert token counts."""
    counts = tokens_per_expert.float()
    total = counts.sum()
    num_experts = counts.numel()
    has_tokens = (total > 0).float()
    fractions = counts / total.clamp_min(1.0)
    dead_count = (counts == 0).sum().float() * has_tokens
    near_dead_threshold = near_dead_uniform_fraction / num_experts
    near_dead_count = (fractions < near_dead_threshold).sum().float() * has_tokens
    max_fraction = fractions.max() * has_tokens
    entropy = compute_normalized_entropy(fractions) * has_tokens
    mean = counts.mean()
    max_vio = ((counts.max() - mean) / mean.clamp_min(1e-20)) * has_tokens
    coefficient_of_variation = (counts.std(unbiased=False) / mean.clamp_min(1e-20)) * has_tokens
    return (dead_count, near_dead_count, max_fraction, entropy, max_vio, coefficient_of_variation)


def update_expert_near_dead_streaks(
    streaks: torch.Tensor, near_dead_mask: torch.Tensor
) -> torch.Tensor:
    """Update consecutive near-dead optimizer-step counts in place."""
    if streaks.shape != near_dead_mask.shape:
        raise ValueError(
            f"streak and near-dead mask shapes must match, got {streaks.shape} and "
            f"{near_dead_mask.shape}"
        )
    streaks.copy_(torch.where(near_dead_mask, streaks + 1, torch.zeros_like(streaks)))
    return streaks


def get_expert_utilization_tracker() -> dict:
    """Return the expert utilization tracker."""
    return _MOE_EXPERT_UTILIZATION_TRACKER


def save_to_expert_utilization_tracker(
    selected_tokens_per_expert: torch.Tensor,
    dispatched_tokens_per_expert: torch.Tensor,
    layer_number: int,
    num_layers: int,
    reduce_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Accumulate selected and dispatched expert counts for global-step logging."""
    if layer_number is None:
        return
    tracker = get_expert_utilization_tracker()
    if "selected_values" not in tracker:
        num_experts = selected_tokens_per_expert.shape[0]
        device = selected_tokens_per_expert.device
        tracker["selected_values"] = torch.zeros(
            num_layers, num_experts, device=device, dtype=torch.float32
        )
        tracker["dispatched_values"] = torch.zeros_like(tracker["selected_values"])
        tracker["selected_near_dead_streaks"] = torch.zeros(
            num_layers, num_experts, device=device, dtype=torch.int32
        )
        tracker["reduce_group"] = reduce_group
    idx = layer_number - 1
    tracker["selected_values"][idx] += selected_tokens_per_expert.detach().float()
    tracker["dispatched_values"][idx] += dispatched_tokens_per_expert.detach().float()


def clear_expert_utilization_tracker() -> None:
    """Clear per-step counts while preserving cross-step near-dead streaks."""
    tracker = get_expert_utilization_tracker()
    if "selected_values" in tracker:
        tracker["selected_values"].zero_()
        tracker["dispatched_values"].zero_()


def get_router_stats_tracker() -> dict:
    """Return the router stats tracker."""
    return _MOE_ROUTER_STATS_TRACKER


def save_to_router_stats_tracker(
    sum_max_score: torch.Tensor,
    sum_score_entropy: torch.Tensor,
    sum_logits: torch.Tensor,
    token_count: torch.Tensor,
    layer_number: int,
    num_layers: int,
    reduce_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Accumulate per-layer router statistics for logging."""
    if layer_number is None:
        return
    tracker = get_router_stats_tracker()
    if "sum_max_score" not in tracker:
        num_experts = sum_logits.shape[0]
        device = sum_logits.device
        tracker["sum_max_score"] = torch.zeros(num_layers, device=device, dtype=torch.float32)
        tracker["sum_score_entropy"] = torch.zeros(num_layers, device=device, dtype=torch.float32)
        tracker["sum_logits"] = torch.zeros(
            num_layers, num_experts, device=device, dtype=torch.float32
        )
        tracker["token_count"] = torch.zeros(num_layers, device=device, dtype=torch.float32)
        tracker["reduce_group"] = reduce_group
    idx = layer_number - 1
    tracker["sum_max_score"][idx] += sum_max_score.detach().float()
    tracker["sum_score_entropy"][idx] += sum_score_entropy.detach().float()
    tracker["sum_logits"][idx] += sum_logits.detach().float()
    tracker["token_count"][idx] += token_count.detach().float()


def clear_router_stats_tracker() -> None:
    """Zero out the router stats tracker without deallocating the buffers."""
    tracker = get_router_stats_tracker()
    if "sum_max_score" in tracker:
        tracker["sum_max_score"].zero_()
        tracker["sum_score_entropy"].zero_()
        tracker["sum_logits"].zero_()
        tracker["token_count"].zero_()


def record_router_health_metrics(
    logits: torch.Tensor,
    selected_routing_map: torch.Tensor,
    routing_map: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    score_function: str,
    expert_bias: Optional[torch.Tensor],
    layer_number: Optional[int],
    num_layers: int,
    reduce_group: Optional[torch.distributed.ProcessGroup],
) -> None:
    """Record utilization and score-distribution stats for one router forward."""
    if layer_number is None or is_graph_capturing():
        return
    if padding_mask is not None:
        valid = ~padding_mask
        selected_tokens_per_expert = selected_routing_map[valid].sum(dim=0).float()
        dispatched_tokens_per_expert = routing_map[valid].sum(dim=0).float()
    else:
        valid = None
        selected_tokens_per_expert = selected_routing_map.sum(dim=0).float()
        dispatched_tokens_per_expert = routing_map.sum(dim=0).float()
    save_to_expert_utilization_tracker(
        selected_tokens_per_expert,
        dispatched_tokens_per_expert,
        layer_number,
        num_layers,
        reduce_group=reduce_group,
    )
    with torch.no_grad():
        score_distribution = compute_router_score_distribution(
            logits, score_function, expert_bias
        )
        max_scores = score_distribution.max(dim=-1).values
        score_entropy = compute_normalized_entropy(score_distribution)
        if valid is not None:
            max_scores = max_scores[valid]
            score_entropy = score_entropy[valid]
            valid_logits = logits.float()[valid]
        else:
            valid_logits = logits.float()
        save_to_router_stats_tracker(
            max_scores.sum(),
            score_entropy.sum(),
            valid_logits.sum(dim=0),
            torch.tensor(float(valid_logits.shape[0]), device=logits.device),
            layer_number,
            num_layers,
            reduce_group=reduce_group,
        )


def _combined_rms(tensors: List[Optional[torch.Tensor]]) -> torch.Tensor:
    valid = [tensor.detach().float() for tensor in tensors if tensor is not None]
    if not valid:
        return torch.tensor(float("nan"))
    square_sum = sum(tensor.square().sum() for tensor in valid)
    count = sum(tensor.numel() for tensor in valid)
    return torch.sqrt(square_sum / count)


def _parameter_or_gradient(parameter: torch.nn.Parameter, gradients: bool) -> Optional[torch.Tensor]:
    if gradients:
        return getattr(parameter, "main_grad", parameter.grad)
    return parameter


def _grouped_mlp_expert_tensor_groups(
    experts: torch.nn.Module, num_local_experts: int, gradients: bool = False
) -> Optional[List[List[torch.Tensor]]]:
    weight1 = getattr(experts, "weight1", None)
    weight2 = getattr(experts, "weight2", None)
    config = getattr(experts, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if weight1 is None or weight2 is None or hidden_size is None:
        return None

    w1 = _parameter_or_gradient(weight1, gradients)
    w2 = _parameter_or_gradient(weight2, gradients)
    if w1 is None or w2 is None:
        return None

    try:
        w1 = w1.view(num_local_experts, hidden_size, -1)
        w2 = w2.view(num_local_experts, -1, hidden_size)
    except RuntimeError:
        return None

    return [[w1[index].reshape(-1), w2[index].reshape(-1)] for index in range(num_local_experts)]


def _iter_local_expert_tensor_groups(
    experts: torch.nn.Module, num_local_experts: int, gradients: bool = False
) -> Optional[List[List[torch.Tensor]]]:
    local = getattr(experts, "local_experts", None)
    if local is not None:
        groups = []
        for expert in local:
            tensors = [
                tensor
                for tensor in (_parameter_or_gradient(p, gradients) for p in expert.parameters())
                if tensor is not None
            ]
            groups.append(tensors)
        return groups

    grouped = _grouped_mlp_expert_tensor_groups(experts, num_local_experts, gradients)
    if grouped is not None:
        return grouped

    per_expert = [[] for _ in range(num_local_experts)]
    for parameter in experts.parameters():
        tensor = _parameter_or_gradient(parameter, gradients)
        if tensor is not None and tensor.ndim > 0 and tensor.shape[0] == num_local_experts:
            for index in range(num_local_experts):
                per_expert[index].append(tensor[index])
    if not any(per_expert):
        return None
    return per_expert


def local_expert_rms(experts: torch.nn.Module, num_local_experts: int, gradients: bool = False):
    """Return combined RMS values for local routed experts."""
    groups = _iter_local_expert_tensor_groups(experts, num_local_experts, gradients)
    if groups is None:
        return None
    return torch.stack([_combined_rms(tensors) for tensors in groups])


def expert_rms_statistics(
    experts: torch.nn.Module, num_local_experts: int, gradients: bool = False
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return per-local-expert squared sums and element counts for a combined RMS."""
    groups = _iter_local_expert_tensor_groups(experts, num_local_experts, gradients)
    if groups is None:
        return None
    stats = []
    for tensors in groups:
        tensors = [tensor.detach().float() for tensor in tensors]
        stats.append(
            (sum(tensor.square().sum() for tensor in tensors), sum(tensor.numel() for tensor in tensors))
        )
    return torch.stack([s[0] for s in stats]), torch.tensor(
        [s[1] for s in stats], device=stats[0][0].device, dtype=torch.float32
    )


def get_expert_viability_tracker() -> dict:
    """Return the opt-in routed-expert viability tracker."""
    return _MOE_EXPERT_VIABILITY_TRACKER


def clear_expert_viability_tracker() -> None:
    """Clear sufficient statistics after their logging event."""
    for key in (
        "routed_sq_sum",
        "input_sq_sum",
        "output_sq_sum",
        "routed_count",
        "input_count",
        "output_count",
        "weight_sq_sum",
        "weight_count",
        "grad_sq_sum",
        "grad_count",
    ):
        if key in _MOE_EXPERT_VIABILITY_TRACKER:
            _MOE_EXPERT_VIABILITY_TRACKER[key].zero_()


def capture_expert_viability_parameter_stats(model_chunks) -> None:
    """Capture parameter and finalized-gradient sufficient statistics at a log event."""
    tracker = get_expert_viability_tracker()
    for chunk in model_chunks:
        for module in chunk.modules():
            if not hasattr(module, "local_expert_indices") or not hasattr(module, "experts"):
                continue
            if not getattr(module.config, "moe_expert_viability_metrics", False):
                continue
            weight = expert_rms_statistics(module.experts, module.num_local_experts)
            grad = expert_rms_statistics(module.experts, module.num_local_experts, gradients=True)
            if weight is None:
                continue
            if "weight_sq_sum" not in tracker:
                device = weight[0].device
                shape = (module.config.num_layers, module.config.num_moe_experts)
                for name in (
                    "weight_sq_sum",
                    "weight_count",
                    "grad_sq_sum",
                    "grad_count",
                    "initial_rms",
                ):
                    tracker[name] = torch.zeros(shape, device=device, dtype=torch.float32)
            layer = module.layer_number - 1
            indices = torch.tensor(module.local_expert_indices, device=weight[0].device)
            tracker["weight_sq_sum"][layer, indices] = weight[0]
            tracker["weight_count"][layer, indices] = weight[1]
            if hasattr(module, "_initial_expert_rms"):
                tracker["initial_rms"][layer, indices] = module._initial_expert_rms.float()
            if grad is not None:
                tracker["grad_sq_sum"][layer, indices] = grad[0]
                tracker["grad_count"][layer, indices] = grad[1]


def save_routed_expert_output_stats(
    routed_output: torch.Tensor,
    input_tensor: torch.Tensor,
    layer_output: torch.Tensor,
    layer_number: Optional[int],
    num_layers: int,
    include_layer_output: bool = True,
) -> None:
    """Accumulate squared-sum/count routed-path statistics without retaining activations."""
    if layer_number is None:
        return
    tracker = _MOE_EXPERT_VIABILITY_TRACKER
    if "routed_sq_sum" not in tracker:
        device = routed_output.device
        tracker["routed_sq_sum"] = torch.zeros(num_layers, device=device, dtype=torch.float32)
        tracker["input_sq_sum"] = torch.zeros_like(tracker["routed_sq_sum"])
        tracker["output_sq_sum"] = torch.zeros_like(tracker["routed_sq_sum"])
        tracker["routed_count"] = torch.zeros_like(tracker["routed_sq_sum"])
        tracker["input_count"] = torch.zeros_like(tracker["routed_sq_sum"])
        tracker["output_count"] = torch.zeros_like(tracker["routed_sq_sum"])
    index = layer_number - 1
    tracker["routed_sq_sum"][index] += routed_output.detach().float().square().sum()
    tracker["input_sq_sum"][index] += input_tensor.detach().float().square().sum()
    tracker["routed_count"][index] += routed_output.numel()
    tracker["input_count"][index] += input_tensor.numel()
    if include_layer_output:
        tracker["output_sq_sum"][index] += layer_output.detach().float().square().sum()
        tracker["output_count"][index] += layer_output.numel()


def _all_reduce_tensors(tensors, groups, avg_last: bool = False):
    for value in tensors:
        for i, group in enumerate(groups):
            if group is None:
                continue
            if avg_last and i == len(groups) - 1:
                torch.distributed.all_reduce(value, group=group, op=torch.distributed.ReduceOp.AVG)
            else:
                torch.distributed.all_reduce(value, group=group)


def report_moe_health_metrics(
    iteration: int,
    writer=None,
    wandb_writer=None,
    total_loss_dict: Optional[dict] = None,
    per_layer_logging: bool = False,
    expert_viability_metrics: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> None:
    """Reduce and log router-health / viability diagnostics, then clear per-step buffers."""
    util_tracker = get_expert_utilization_tracker()
    if "selected_values" in util_tracker:
        selected_values = util_tracker["selected_values"]
        dispatched_values = util_tracker["dispatched_values"]
        pp_group = _pp_group(pg_collection)
        dp_group = _dp_group(pg_collection)
        reduce_group = util_tracker.get("reduce_group")
        for values in (selected_values, dispatched_values):
            torch.distributed.all_reduce(values, group=pp_group)
            if reduce_group is not None:
                torch.distributed.all_reduce(values, group=reduce_group)
            torch.distributed.all_reduce(values, group=dp_group, op=torch.distributed.ReduceOp.AVG)

        active_layers = selected_values.sum(dim=-1) > 0
        selected_means = selected_values.mean(dim=-1, keepdim=True)
        selected_zero_mask = (selected_values == 0) & active_layers.unsqueeze(-1)
        selected_near_dead_mask = (selected_values < 0.1 * selected_means) & active_layers.unsqueeze(
            -1
        )
        near_dead_streaks = util_tracker.get("selected_near_dead_streaks")
        if near_dead_streaks is None or near_dead_streaks.shape != selected_values.shape:
            near_dead_streaks = torch.zeros_like(selected_values, dtype=torch.int32)
            util_tracker["selected_near_dead_streaks"] = near_dead_streaks
        update_expert_near_dead_streaks(near_dead_streaks, selected_near_dead_mask)

        selected_zero_slots = selected_zero_mask.sum().item()
        selected_near_dead_slots = selected_near_dead_mask.sum().item()
        persistent_near_dead_slots = (near_dead_streaks >= 100).sum().item()

        dead_count_list = []
        expert_entropy_list = []
        selected_max_vio_list = []
        dropped_frac_list = []
        wandb_layer_log: dict = {}
        have_wandb = per_layer_logging and (wandb_writer is not None)

        for i in range(dispatched_values.shape[0]):
            selected = selected_values[i]
            dispatched = dispatched_values[i]
            selected_total = selected.sum().item()
            dispatched_total = dispatched.sum().item()
            if selected_total == 0:
                continue

            selected_max_vio = compute_expert_load_metrics(selected)[4].item()
            if dispatched_total > 0:
                fractions = dispatched / dispatched_total
                dead_count = int((dispatched == 0).sum().item())
                entropy = compute_normalized_entropy(fractions).item()
            else:
                dead_count = dispatched.numel()
                entropy = 0.0
            dropped_frac = 1.0 - (dispatched_total / selected_total)

            dead_count_list.append(dead_count)
            expert_entropy_list.append(entropy)
            selected_max_vio_list.append(selected_max_vio)
            dropped_frac_list.append(dropped_frac)

            if per_layer_logging:
                if writer is not None:
                    writer.add_scalar(
                        f"moe/dispatched_expert_load_entropy_layer_{i}", entropy, iteration
                    )
                if have_wandb:
                    wandb_layer_log[f"router-layers/dispatched_expert_load_entropy_layer_{i}"] = (
                        entropy
                    )

        if dead_count_list:
            aggregate_log = {
                "expert_dead_count": sum(dead_count_list),
                "dispatched_expert_load_entropy_mean": sum(expert_entropy_list)
                / len(expert_entropy_list),
                "dispatched_expert_load_entropy_min": min(expert_entropy_list),
                "dispatched_expert_load_entropy_max": max(expert_entropy_list),
                "selected_zero_load_expert_layer_slots": selected_zero_slots,
                "selected_near_dead_expert_layer_slots": selected_near_dead_slots,
                "persistent_selected_near_dead_expert_layer_slots_100": persistent_near_dead_slots,
                "selected_expert_max_vio_max": max(selected_max_vio_list),
                "dropped_assignment_frac_max": max(dropped_frac_list),
            }
            if writer is not None:
                for name, value in aggregate_log.items():
                    writer.add_scalar(f"moe/{name}", value, iteration)
            if wandb_writer:
                wandb_writer.log(
                    {
                        **wandb_layer_log,
                        **{f"router-aggregates/{name}": value for name, value in aggregate_log.items()},
                    },
                    iteration,
                )

    rs_tracker = get_router_stats_tracker()
    if "sum_max_score" in rs_tracker:
        sum_max_score = rs_tracker["sum_max_score"]
        sum_score_entropy = rs_tracker["sum_score_entropy"]
        sum_logits = rs_tracker["sum_logits"]
        token_count = rs_tracker["token_count"]
        pp_group = _pp_group(pg_collection)
        dp_group = _dp_group(pg_collection)
        for t in (sum_max_score, sum_score_entropy, sum_logits, token_count):
            torch.distributed.all_reduce(t, group=pp_group)
        if rs_tracker.get("reduce_group") is not None:
            for t in (sum_max_score, sum_score_entropy, sum_logits, token_count):
                torch.distributed.all_reduce(t, group=rs_tracker["reduce_group"])
        for t in (sum_max_score, sum_score_entropy, sum_logits, token_count):
            torch.distributed.all_reduce(t, group=dp_group, op=torch.distributed.ReduceOp.AVG)

        have_wandb_rs = per_layer_logging and (wandb_writer is not None)
        logit_spread_list = []
        router_score_entropy_list = []
        wandb_rs_log: dict = {}
        for i in range(sum_max_score.shape[0]):
            cnt = token_count[i].item()
            if cnt == 0:
                continue
            mean_max_score_i = (sum_max_score[i] / cnt).item()
            mean_score_entropy_i = (sum_score_entropy[i] / cnt).item()
            mean_logit_i = (sum_logits[i] / cnt).cpu()
            logit_spread_i = mean_logit_i.std(unbiased=False).item()
            logit_spread_list.append(logit_spread_i)
            router_score_entropy_list.append(mean_score_entropy_i)
            if per_layer_logging:
                if writer is not None:
                    writer.add_scalar(f"moe/router_mean_max_prob_layer_{i}", mean_max_score_i, iteration)
                    writer.add_scalar(
                        f"moe/router_mean_score_entropy_layer_{i}", mean_score_entropy_i, iteration
                    )
                    writer.add_scalar(f"moe/expert_logit_spread_layer_{i}", logit_spread_i, iteration)
                if have_wandb_rs:
                    wandb_rs_log[f"router-layers/router_mean_max_prob_layer_{i}"] = mean_max_score_i
                    wandb_rs_log[f"router-layers/router_mean_score_entropy_layer_{i}"] = (
                        mean_score_entropy_i
                    )
                    wandb_rs_log[f"router-layers/expert_logit_spread_layer_{i}"] = logit_spread_i

        if logit_spread_list:
            agg_max_logit_spread = max(logit_spread_list)
            agg_mean_logit_spread = sum(logit_spread_list) / len(logit_spread_list)
            agg_mean_router_score_entropy = sum(router_score_entropy_list) / len(
                router_score_entropy_list
            )
            agg_max_router_score_entropy = max(router_score_entropy_list)
            agg_min_router_score_entropy = min(router_score_entropy_list)
            if total_loss_dict is not None:
                total_loss_dict["expert_logit_spread_max"] = torch.tensor(agg_max_logit_spread)
            if writer is not None:
                writer.add_scalar("moe/expert_logit_spread_max", agg_max_logit_spread, iteration)
                writer.add_scalar("moe/expert_logit_spread_mean", agg_mean_logit_spread, iteration)
                writer.add_scalar(
                    "moe/router_mean_score_entropy_mean", agg_mean_router_score_entropy, iteration
                )
                writer.add_scalar(
                    "moe/router_mean_score_entropy_min", agg_min_router_score_entropy, iteration
                )
                writer.add_scalar(
                    "moe/router_mean_score_entropy_max", agg_max_router_score_entropy, iteration
                )
            if wandb_writer:
                wandb_writer.log(
                    {
                        **wandb_rs_log,
                        "router-aggregates/expert_logit_spread_max": agg_max_logit_spread,
                        "router-aggregates/expert_logit_spread_mean": agg_mean_logit_spread,
                        "router-aggregates/router_mean_score_entropy_mean": (
                            agg_mean_router_score_entropy
                        ),
                        "router-aggregates/router_mean_score_entropy_min": (
                            agg_min_router_score_entropy
                        ),
                        "router-aggregates/router_mean_score_entropy_max": (
                            agg_max_router_score_entropy
                        ),
                    },
                    iteration,
                )

    viability_tracker = get_expert_viability_tracker()
    if expert_viability_metrics and "routed_sq_sum" in viability_tracker:
        values = [
            viability_tracker[key]
            for key in (
                "routed_sq_sum",
                "input_sq_sum",
                "output_sq_sum",
                "routed_count",
                "input_count",
                "output_count",
            )
        ]
        _all_reduce_tensors(
            values,
            (_tp_cp_group(pg_collection), _pp_group(pg_collection), _dp_group(pg_collection)),
            avg_last=True,
        )
        layer_log = {}
        wandb_layer_log = {}
        routed_rms_list, routed_to_input, routed_to_output = [], [], []
        have_wandb_viability = wandb_writer is not None
        for i in range(viability_tracker["routed_sq_sum"].numel()):
            routed_count = viability_tracker["routed_count"][i]
            if routed_count.item() == 0:
                continue
            routed_rms = torch.sqrt(viability_tracker["routed_sq_sum"][i] / routed_count).item()
            input_rms = torch.sqrt(
                viability_tracker["input_sq_sum"][i] / viability_tracker["input_count"][i]
            ).item()
            input_ratio = routed_rms / max(input_rms, 1.0e-12)
            output_ratio = None
            if viability_tracker["output_count"][i].item() > 0:
                output_rms = torch.sqrt(
                    viability_tracker["output_sq_sum"][i] / viability_tracker["output_count"][i]
                ).item()
                output_ratio = routed_rms / max(output_rms, 1.0e-12)
            if writer is not None:
                writer.add_scalar(f"moe/routed_expert_output_rms_layer_{i}", routed_rms, iteration)
                writer.add_scalar(
                    f"moe/routed_expert_output_to_input_rms_layer_{i}", input_ratio, iteration
                )
                if output_ratio is not None:
                    writer.add_scalar(
                        f"moe/routed_expert_output_to_layer_output_rms_layer_{i}",
                        output_ratio,
                        iteration,
                    )
            if have_wandb_viability:
                wandb_layer_log[f"viability-layers/routed_expert_output_rms_layer_{i}"] = routed_rms
                wandb_layer_log[f"viability-layers/routed_expert_output_to_input_rms_layer_{i}"] = (
                    input_ratio
                )
                if output_ratio is not None:
                    wandb_layer_log[
                        f"viability-layers/routed_expert_output_to_layer_output_rms_layer_{i}"
                    ] = output_ratio
            routed_rms_list.append(routed_rms)
            routed_to_input.append(input_ratio)
            if output_ratio is not None:
                routed_to_output.append(output_ratio)

        routed_aggregate_specs = [
            ("routed_expert_output_rms", routed_rms_list),
            ("routed_expert_output_to_input_rms", routed_to_input),
        ]
        if routed_to_output:
            routed_aggregate_specs.append(
                ("routed_expert_output_to_layer_output_rms", routed_to_output)
            )
        wandb_aggregate_log = {}
        for metric_name, vals in routed_aggregate_specs:
            aggregates = _aggregate_layer_values(vals)
            for stat_name, value in aggregates.items():
                layer_log[f"moe/{metric_name}_{stat_name}"] = value
                wandb_aggregate_log[f"viability-aggregates/{metric_name}_{stat_name}"] = value
        if routed_to_input and total_loss_dict is not None:
            total_loss_dict["routed_expert_output_to_input_rms_min"] = torch.tensor(
                layer_log["moe/routed_expert_output_to_input_rms_min"]
            )
        if writer is not None:
            for name, value in layer_log.items():
                writer.add_scalar(name, value, iteration)
        if wandb_writer and (wandb_layer_log or wandb_aggregate_log):
            wandb_writer.log({**wandb_layer_log, **wandb_aggregate_log}, iteration)

    if expert_viability_metrics and "weight_sq_sum" in viability_tracker:
        groups = (
            _tp_cp_group(pg_collection),
            _pp_group(pg_collection),
            _ep_group(pg_collection),
        )
        for name in ("weight_sq_sum", "weight_count", "grad_sq_sum", "grad_count", "initial_rms"):
            value = viability_tracker[name]
            for group in groups:
                torch.distributed.all_reduce(value, group=group)
        weight_rms = torch.sqrt(
            viability_tracker["weight_sq_sum"] / viability_tracker["weight_count"].clamp_min(1)
        )
        grad_rms = torch.sqrt(
            viability_tracker["grad_sq_sum"] / viability_tracker["grad_count"].clamp_min(1)
        )
        initial_rms = viability_tracker["initial_rms"]
        param_log = {}
        wandb_param_layer_log = {}
        weight_medians, grad_medians, collapsed_fractions, relative_medians = [], [], [], []
        have_wandb_param = wandb_writer is not None
        for i in range(weight_rms.shape[0]):
            active = viability_tracker["weight_count"][i] > 0
            if not active.any():
                continue
            weights = weight_rms[i, active]
            grads = grad_rms[i, active]
            initial = initial_rms[i, active]
            relative = weights / initial.clamp_min(1.0e-12)
            collapsed = (relative < EXPERT_COLLAPSE_RMS_FRACTION).float().mean().item()
            weight_median = weights.median().item()
            grad_median = grads.median().item()
            relative_median = relative.median().item()
            if writer is not None:
                writer.add_scalar(f"moe/expert_weight_rms_median_layer_{i}", weight_median, iteration)
                writer.add_scalar(
                    f"moe/expert_weight_rms_p10_layer_{i}",
                    torch.quantile(weights, 0.1).item(),
                    iteration,
                )
                writer.add_scalar(
                    f"moe/expert_weight_rms_min_layer_{i}", weights.min().item(), iteration
                )
                writer.add_scalar(
                    f"moe/expert_weight_rms_relative_to_init_median_layer_{i}",
                    relative_median,
                    iteration,
                )
                writer.add_scalar(
                    f"moe/expert_weight_collapsed_frac_layer_{i}", collapsed, iteration
                )
                writer.add_scalar(f"moe/expert_grad_rms_median_layer_{i}", grad_median, iteration)
            if have_wandb_param:
                wandb_param_layer_log[f"viability-layers/expert_weight_rms_median_layer_{i}"] = (
                    weight_median
                )
                wandb_param_layer_log[f"viability-layers/expert_weight_rms_p10_layer_{i}"] = (
                    torch.quantile(weights, 0.1).item()
                )
                wandb_param_layer_log[f"viability-layers/expert_weight_rms_min_layer_{i}"] = (
                    weights.min().item()
                )
                wandb_param_layer_log[
                    f"viability-layers/expert_weight_rms_relative_to_init_median_layer_{i}"
                ] = relative_median
                wandb_param_layer_log[f"viability-layers/expert_weight_collapsed_frac_layer_{i}"] = (
                    collapsed
                )
                wandb_param_layer_log[f"viability-layers/expert_grad_rms_median_layer_{i}"] = (
                    grad_median
                )
            weight_medians.append(weight_median)
            grad_medians.append(grad_median)
            collapsed_fractions.append(collapsed)
            relative_medians.append(relative_median)

        param_aggregate_specs = (
            ("expert_weight_rms_median", weight_medians),
            ("expert_grad_rms_median", grad_medians),
            ("expert_weight_collapsed_frac", collapsed_fractions),
            ("expert_weight_rms_relative_to_init_median", relative_medians),
        )
        wandb_param_aggregate_log = {}
        for metric_name, vals in param_aggregate_specs:
            aggregates = _aggregate_layer_values(vals)
            for stat_name, value in aggregates.items():
                param_log[f"moe/{metric_name}_{stat_name}"] = value
                wandb_param_aggregate_log[f"viability-aggregates/{metric_name}_{stat_name}"] = value
        if collapsed_fractions and total_loss_dict is not None:
            total_loss_dict["expert_weight_collapsed_frac_max"] = torch.tensor(
                param_log["moe/expert_weight_collapsed_frac_max"]
            )
        if writer is not None:
            for name, value in param_log.items():
                writer.add_scalar(name, value, iteration)
        if wandb_writer and (wandb_param_layer_log or wandb_param_aggregate_log):
            wandb_writer.log({**wandb_param_layer_log, **wandb_param_aggregate_log}, iteration)

    clear_expert_utilization_tracker()
    clear_router_stats_tracker()
    clear_expert_viability_tracker()


def run_masked_layer_validation(
    forward_step_func,
    data_iterator,
    model,
    config,
    iteration: int,
    writer=None,
    wandb_writer=None,
    pg_collection=None,
    p2p_communicator=None,
) -> None:
    """Run paired validation with one routed MoE layer masked at a time."""
    from megatron.training import get_args, print_rank_0
    from megatron.training.training import evaluate

    args = get_args()
    if not args.moe_masked_layer_validation:
        return

    eval_iters = args.moe_masked_layer_eval_iters
    caching_iterator = _CachingDataIterator(data_iterator)

    def _nll(iterator) -> float:
        loss_dict, _, _ = evaluate(
            forward_step_func,
            iterator,
            model,
            process_non_loss_data_func=None,
            config=config,
            verbose=False,
            eval_iters=eval_iters,
            pg_collection=pg_collection,
            p2p_communicator=p2p_communicator,
        )
        if not loss_dict or "lm loss" not in loss_dict:
            return float("nan")
        value = loss_dict["lm loss"]
        return value.item() if torch.is_tensor(value) else float(value)

    baseline_nll = _nll(caching_iterator)
    baseline_ppl = math.exp(min(20.0, baseline_nll))

    layer_numbers = collect_moe_layer_numbers(model)
    nll_deltas: List[float] = []
    layer_log: dict[str, float] = {}
    wandb_layer_log: dict[str, float] = {}

    for layer_number in layer_numbers:
        caching_iterator.rewind()
        with mask_routed_moe_layer(layer_number):
            masked_nll = _nll(caching_iterator)
        nll_delta = masked_nll - baseline_nll
        ppl_delta = math.exp(min(20.0, masked_nll)) - baseline_ppl
        nll_deltas.append(nll_delta)
        layer_idx = layer_number - 1
        layer_log[f"moe/masked_layer_nll_delta_layer_{layer_idx}"] = nll_delta
        layer_log[f"moe/masked_layer_ppl_delta_layer_{layer_idx}"] = ppl_delta
        wandb_layer_log[f"viability-layers/masked_layer_nll_delta_layer_{layer_idx}"] = nll_delta
        wandb_layer_log[f"viability-layers/masked_layer_ppl_delta_layer_{layer_idx}"] = ppl_delta

    aggregate_log = {}
    wandb_aggregate_log = {}
    if nll_deltas:
        for stat_name, value in _aggregate_layer_values(nll_deltas).items():
            aggregate_log[f"moe/masked_layer_nll_delta_{stat_name}"] = value
            wandb_aggregate_log[f"viability-aggregates/masked_layer_nll_delta_{stat_name}"] = value

    if writer is not None:
        for name, value in {**layer_log, **aggregate_log}.items():
            writer.add_scalar(name, value, iteration)
    if wandb_writer and (wandb_layer_log or wandb_aggregate_log):
        wandb_writer.log({**wandb_layer_log, **wandb_aggregate_log}, iteration)

    if layer_numbers:
        agg = _aggregate_layer_values(nll_deltas)
        print_rank_0(
            " masked-layer validation at iteration {} | baseline nll {:.6E} | "
            "nll delta mean {:.6E} min {:.6E} max {:.6E}".format(
                iteration,
                baseline_nll,
                agg.get("mean", float("nan")),
                agg.get("min", float("nan")),
                agg.get("max", float("nan")),
            )
        )
