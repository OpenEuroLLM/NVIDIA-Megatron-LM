# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Diagnostics of the architecture features: gates, routers, residual streams, memories.

The MoE router has its own health metrics (moe_health.py: score entropy, dead /
near-dead experts, load imbalance). This module covers the rest -- mHC stream usage,
routed-residual stream selection, Block AttnRes depth attention, nGPT LERP rates,
attention output gates, exclusive self attention, Engram gates, product-key memory
usage, shared-expert gates, and the decay / input gates of the native linear mixers.

Cost. Recording only happens on *metric steps*: the training loop turns it on with
:func:`set_active` for the steps it will report (``--arch-metrics-interval``), so the
other steps pay one Python bool check per call site. Records are never taken during
CUDA-graph capture or a no-grad (recompute / eval) forward.

Aggregation. A metric is a per-layer ``(sum, count)`` pair (a scalar or a small
vector per layer). :func:`report` all-reduces both over the whole world and reports
``sum / count``: layers on different pipeline stages are disjoint, and ranks that
compute the same value (tensor-parallel replicas) add to the sum and the count
alike, so the mean is that of all tokens seen. Union metrics (which memory rows were
read) are bitmaps reduced with MAX.
"""

import math
from typing import Callable, Dict, Optional

import torch

_ACTIVE = False
_SUMS: Dict[str, torch.Tensor] = {}
_COUNTS: Dict[str, torch.Tensor] = {}
_UNIONS: Dict[str, torch.Tensor] = {}
# Vector metrics are reported through a summary of each layer's mean vector.
_VECTOR_SUMMARIES: Dict[str, Callable[[torch.Tensor], Dict[str, float]]] = {}


def set_active(active: bool) -> None:
    """Record on the coming forward passes (set by the training loop per step)."""
    global _ACTIVE
    _ACTIVE = bool(active)


def active() -> bool:
    """True when a call site should compute and record its metrics now."""
    if not _ACTIVE or not torch.is_grad_enabled():
        return False
    from megatron.core.transformer.cuda_graphs import is_graph_capturing

    return not is_graph_capturing()


def num_layers_of(config) -> int:
    return config.num_layers + (getattr(config, "mtp_num_layers", None) or 0)


def tag(module, kind: str, layer_number: Optional[int], config) -> None:
    """Tell a metric-recording module where it sits: ``kind`` names the sublayer
    ("attn", "mlp", "mixer"), ``layer_number`` its layer (1-based)."""
    if module is not None:
        module._arch_metrics_tag = (kind, layer_number, num_layers_of(config))


def tag_of(module):
    """``(kind, layer_number, num_layers)`` of a tagged module, or None."""
    return getattr(module, "_arch_metrics_tag", None)


def record(
    name: str,
    layer_number: Optional[int],
    num_layers: int,
    total: torch.Tensor,
    count: float | torch.Tensor,
) -> None:
    """Add ``total`` (a sum over ``count`` items, scalar or vector) to layer
    ``layer_number`` (1-based) of metric ``name``."""
    if layer_number is None or not 1 <= layer_number <= num_layers:
        return
    total = total.detach().float()
    if name not in _SUMS:
        _SUMS[name] = torch.zeros((num_layers,) + tuple(total.shape), device=total.device)
        _COUNTS[name] = torch.zeros(num_layers, device=total.device)
    _SUMS[name][layer_number - 1] += total
    _COUNTS[name][layer_number - 1] += count


def record_mean(name, layer_number, num_layers, values: torch.Tensor) -> None:
    """Record the mean of all elements of ``values``."""
    values = values.detach().float()
    record(name, layer_number, num_layers, values.sum(), values.numel())


def record_share(name, layer_number, num_layers, shares: torch.Tensor) -> None:
    """Record ``shares`` [tokens, n] (each row a distribution over n slots: streams,
    sources); reported as the normalized entropy of the mean distribution and the
    smallest slot's share relative to uniform (0 = a dead slot)."""
    shares = shares.detach().float().reshape(-1, shares.shape[-1])
    _VECTOR_SUMMARIES.setdefault(name, _share_summary)
    record(name, layer_number, num_layers, shares.sum(0), shares.shape[0])


def record_streams(name, layer_number, num_layers, streams: torch.Tensor) -> None:
    """Record the size of each of the ``n`` residual streams ``[..., n, h]`` (mHC / routed
    residual) entering a sublayer: per-stream mean square over all tokens; reported as the RMS of
    stream 0 (the main stream), the mean RMS of the others, the largest RMS and the max / min
    ratio -- where across depth the streams grow, die or stay unused."""
    x = streams.detach().float()
    n = x.shape[-2]
    ms = x.reshape(-1, n, x.shape[-1]).square().mean(-1)  # [tokens, n]
    _VECTOR_SUMMARIES.setdefault(name, _stream_summary)
    record(name, layer_number, num_layers, ms.sum(0), ms.shape[0])


def _stream_summary(mean_square: torch.Tensor) -> Dict[str, float]:
    rms = mean_square.clamp(min=0).sqrt()
    rest = rms[1:] if rms.numel() > 1 else rms
    return {
        "main": float(rms[0]),
        "others": float(rest.mean()),
        "max": float(rms.max()),
        "max_over_min": float(rms.max() / rms.min().clamp(min=1e-12)),
    }


def record_union(name, layer_number, num_layers, size: int, indices: torch.Tensor) -> None:
    """Mark ``indices`` (of ``size`` slots, e.g. memory rows) as used; reported as the
    fraction of slots used since the last report."""
    if layer_number is None or not 1 <= layer_number <= num_layers:
        return
    if name not in _UNIONS:
        _UNIONS[name] = torch.zeros(num_layers, size, dtype=torch.uint8, device=indices.device)
    _UNIONS[name][layer_number - 1].index_fill_(0, indices.reshape(-1).long(), 1)


def _share_summary(mean: torch.Tensor) -> Dict[str, float]:
    p = mean.clamp(min=0)
    p = p / p.sum().clamp(min=1e-12)
    n = p.numel()
    entropy = -(p * p.clamp(min=1e-12).log()).sum() / math.log(n) if n > 1 else p.new_ones(())
    return {"entropy": float(entropy), "min_rel": float(p.min() * n)}


def _sync_names(group) -> None:
    """Create on every rank the entries other ranks have, so the all-reduces match."""
    import torch.distributed as dist

    local = {
        "sums": {k: (tuple(v.shape), str(v.device)) for k, v in _SUMS.items()},
        "unions": {k: tuple(v.shape) for k, v in _UNIONS.items()},
        "vec": sorted(_VECTOR_SUMMARIES),
    }
    gathered = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, local, group=group)
    device = torch.device("cuda", torch.cuda.current_device())
    for other in gathered:
        for name, (shape, _) in other["sums"].items():
            if name not in _SUMS:
                _SUMS[name] = torch.zeros(shape, device=device)
                _COUNTS[name] = torch.zeros(shape[0], device=device)
        for name, shape in other["unions"].items():
            if name not in _UNIONS:
                _UNIONS[name] = torch.zeros(shape, dtype=torch.uint8, device=device)
        for name in other["vec"]:
            summary = _stream_summary if name.endswith("/stream_rms") else _share_summary
            _VECTOR_SUMMARIES.setdefault(name, summary)


def collect(group=None) -> Dict[str, float]:
    """Reduce the recorded metrics over ``group`` (default: the world) and return
    ``{metric: value}``: per-layer values as ``<name>/layer_<i>`` and the mean over
    the layers that recorded as ``<name>``. Clears the buffers."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        _sync_names(group)
        for name in sorted(_SUMS):
            dist.all_reduce(_SUMS[name], group=group)
            dist.all_reduce(_COUNTS[name], group=group)
        for name in sorted(_UNIONS):
            dist.all_reduce(_UNIONS[name], op=dist.ReduceOp.MAX, group=group)
    out: Dict[str, float] = {}
    for name in sorted(_SUMS):
        sums, counts = _SUMS[name].cpu(), _COUNTS[name].cpu()
        layers = [i for i in range(counts.numel()) if counts[i] > 0]
        if not layers:
            continue
        summary = _VECTOR_SUMMARIES.get(name)
        per_layer: Dict[str, list] = {}
        for i in layers:
            mean = sums[i] / counts[i]
            values = summary(mean) if summary is not None else {"": float(mean)}
            for key, value in values.items():
                metric = f"{name}/{key}" if key else name
                out[f"{metric}/layer_{i + 1}"] = value
                per_layer.setdefault(metric, []).append(value)
        for metric, values in per_layer.items():
            out[metric] = sum(values) / len(values)
    for name in sorted(_UNIONS):
        used = _UNIONS[name].float().mean(dim=1).cpu()
        layers = [i for i in range(used.numel()) if bool(_UNIONS[name][i].any())]
        for i in layers:
            out[f"{name}/layer_{i + 1}"] = float(used[i])
        if layers:
            out[name] = float(sum(used[i] for i in layers) / len(layers))
    clear()
    return out


def clear() -> None:
    for buf in list(_SUMS.values()) + list(_COUNTS.values()) + list(_UNIONS.values()):
        buf.zero_()


def reset() -> None:
    """Drop every entry (tests, or a model rebuild)."""
    _SUMS.clear()
    _COUNTS.clear()
    _UNIONS.clear()
    _VECTOR_SUMMARIES.clear()


def report(iteration: int, writer=None, wandb_writer=None, per_layer: bool = False) -> str:
    """Collect and log to TensorBoard / W&B; returns a console string of the
    layer-averaged metrics."""
    metrics = collect()
    shown = {k: v for k, v in metrics.items() if per_layer or "/layer_" not in k}
    if writer is not None:
        for key, value in shown.items():
            writer.add_scalar(f"arch/{key}", value, iteration)
    if wandb_writer:
        wandb_writer.log({f"arch/{key}": value for key, value in shown.items()}, iteration)
    return " | ".join(
        f"{k}: {v:.4g}" for k, v in sorted(metrics.items()) if "/layer_" not in k
    )
