# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""The routed-expert spectral cap measured in the forward pass (``muown_expert_spectral_cap_source
= forward``, Megatron-FSDP).

Under Megatron-FSDP the optimizer holds only row shards of the fp32 expert matrices, so the
``owner`` cap (``TensorParallelMuownClassical._row_sharded_floor``) sends every expert matrix to
an owner rank and back each time it is applied (~0.8 s per step for the v4 main run at 256 nodes).
The full matrices exist anyway in the forward pass: FSDP all-gathers each rank's local experts
(with ``fsdp_fp8_param_gather=experts`` as the dequantized FP8 round trip of the bf16 weights,
accurate to ~1 % in sigma_max). Here:

1. the optimizer ``arm``s the cap at the end of a step for the next optimizer step that is due
   (``muown_expert_bound_interval``), with that step's cap per matrix shape;
2. the experts' forward (``MoonEPGroupedMLP._gathered_weights``: plain forward and the chunked
   path, first forward and recompute alike) ``measure``s sigma_max of the full matrices once per
   armed step -- a few batched, warm-started power iterations -- and stores the scale
   ``c = min(1, cap / sigma_max)`` per matrix, keyed by the forward parameter object;
3. the optimizer step that is due multiplies each rank's local fp32 rows and Muown row
   magnitudes g of the matrix by c (W = g U / |U|: scaling W is scaling g), before the main
   weights are copied to the model weights.

No matrix moves: every rank that gathered a matrix measures the same bytes and gets the same c.
The cap acts on the weights of the previous step (one step later than the owner cap).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch

_WARM_ITERS = 3  # power iterations from the previous step's right singular vector
_COLD_ITERS = 12  # from the column sums of |W| (first measurement of a matrix)
_CHUNK = 32  # matrices per batched power iteration when they are not one contiguous stack


class _State:
    def __init__(self):
        self.step: Optional[int] = None  # the optimizer step the measured scales are for
        self.caps: Dict[tuple, float] = {}  # matrix shape -> absolute sigma_max cap
        self.scales: Dict[int, torch.Tensor] = {}  # id(forward param) -> 0-dim fp32 scale
        self.vectors: Dict[int, torch.Tensor] = {}  # id(forward param) -> fp32 warm start
        self.measured: set = set()

    def reset(self):
        self.__init__()


STATE = _State()


def arm(step: int, caps: Dict[tuple, float]) -> None:
    """Measure in the coming forward(s), for optimizer step ``step`` (caps per matrix shape)."""
    STATE.step = int(step)
    STATE.caps = {tuple(int(d) for d in k): float(v) for k, v in caps.items() if v > 0.0}
    STATE.scales = {}
    STATE.measured = set()


def disarm() -> None:
    STATE.step = None
    STATE.caps = {}
    STATE.scales = {}
    STATE.measured = set()


def armed() -> bool:
    return STATE.step is not None and bool(STATE.caps)


def take_scales(step: int) -> Dict[int, torch.Tensor]:
    """The scales measured for optimizer ``step`` (empty if none were armed for it)."""
    if STATE.step != step:
        return {}
    return STATE.scales


def _stacked(tensors: List[torch.Tensor]) -> Optional[torch.Tensor]:
    """``[n, *shape]`` zero-copy view of same-shape tensors back to back in one storage."""
    t0 = tensors[0]
    nbytes = t0.numel() * t0.element_size()
    base = t0.untyped_storage().data_ptr()
    for i, t in enumerate(tensors):
        if (
            t.dtype != t0.dtype
            or t.shape != t0.shape
            or not t.is_contiguous()
            or t.untyped_storage().data_ptr() != base
            or t.data_ptr() != t0.data_ptr() + i * nbytes
        ):
            return None
    return torch.as_strided(
        t0.detach(), (len(tensors), *t0.shape), (t0.numel(), *t0.stride()), t0.storage_offset()
    )


def _sigma_max(w: torch.Tensor, v: torch.Tensor, iters: int) -> tuple:
    """Batched power iteration on ``w`` [n, m, k] from right vectors ``v`` [n, k] (fp32):
    (sigma_max [n] fp32, the new right vectors [n, k] fp32). The products run in w's dtype
    (bf16) with fp32 accumulation; the vectors are normalized in fp32."""
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    for _ in range(iters):
        u = torch.bmm(w, v.to(w.dtype).unsqueeze(-1)).squeeze(-1).float()
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        v = torch.bmm(w.transpose(1, 2), u.to(w.dtype).unsqueeze(-1)).squeeze(-1).float()
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    sigma = torch.bmm(w, v.to(w.dtype).unsqueeze(-1)).squeeze(-1).float().norm(dim=-1)
    return sigma, v


@torch.no_grad()
def measure(groups: Iterable[List[torch.nn.Parameter]]) -> None:
    """sigma_max -> scale of every not yet measured matrix in ``groups`` (lists of same-shape
    full 2-D weights, e.g. one module's fc1 and fc2 experts) whose shape has an armed cap."""
    if not armed() or torch.cuda.is_current_stream_capturing():
        return
    from megatron.core.transformer import arch_metrics

    record = arch_metrics.step_active()
    for params in groups:
        todo = [p for p in params if id(p) not in STATE.measured]
        if not todo:
            continue
        shape = tuple(int(d) for d in todo[0].shape)
        cap = STATE.caps.get(shape)
        if cap is None:
            continue
        stack = _stacked(todo) if len(todo) == len(params) else None
        chunks = [(todo, stack)] if stack is not None else [
            (todo[i:i + _CHUNK], torch.stack([p.detach() for p in todo[i:i + _CHUNK]]))
            for i in range(0, len(todo), _CHUNK)
        ]
        for ps, w in chunks:
            warm = all(id(p) in STATE.vectors for p in ps)
            if warm:
                v0 = torch.stack([STATE.vectors[id(p)] for p in ps])
            else:  # column sums of |W|: deterministic, and close to the top direction
                v0 = w.abs().float().sum(dim=1)
            sigma, v = _sigma_max(w, v0, _WARM_ITERS if warm else _COLD_ITERS)
            scale = (cap / sigma.clamp_min(1e-30)).clamp_max(1.0)
            for j, p in enumerate(ps):
                STATE.scales[id(p)] = scale[j]
                STATE.vectors[id(p)] = v[j]
                STATE.measured.add(id(p))
            if record:
                name = f"muown/expert_{shape[0]}x{shape[1]}"
                arch_metrics.record(f"{name}/capped", 1, 1, (sigma > cap).float().sum(), len(ps))
                arch_metrics.record(f"{name}/sigma_over_cap", 1, 1, (sigma / cap).sum(), len(ps))
