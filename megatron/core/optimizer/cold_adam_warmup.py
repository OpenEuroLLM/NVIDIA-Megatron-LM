# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Ramp in the updates of Adam-handled parameter elements that resume with a cold (zero) second moment.

Checkpoints written on JUPITER (pt2512 image: torch 2.10 nv25.12, TE 2.10; Megatron-FSDP fsdp_dtensor) hold every
FSDP-sharded Adam state (``exp_avg`` / ``exp_avg_sq``) as one full-size item with only data-parallel rank 0's copy
filled: TE 2.10's FusedAdam creates the states as plain ``torch.empty(param.shape)`` tensors, not DTensors (newer TE
re-wraps them), so DCP treats them as replicated
(docs/recall_surgery_ambitious.md §14.6): 15/16 of the output layer's and embedding's moments and all moments of
most small parameters (mHC coefficients, norms, CKDA conv / gate vectors) load as zeros, while the live optimizer
state is fine. A resume then restarts those elements' Adam statistics from zero at a constant learning rate --
~0.45 lr * sign(g) per element -- and the loss jumped from ~3 to 30-120 after the first update, recovering only
over ~100 steps.

``install_cold_adam_warmup(optimizer)`` finds, after the optimizer state has been loaded, every parameter whose
Adam state is cold on this rank (``cold_parameters``: states saved in the broken layout -- plain tensors of the
parameter's global shape -- are zeroed and wholly cold; otherwise the elements with a zero ``exp_avg_sq``), and wraps
the step of the torch optimizer that holds it: for the next ``OELLM_COLD_ADAM_WARMUP`` steps (default 100; 0 disables) the applied update
of those elements is scaled by k / N (k = 1, 2, ...). Their moments evolve normally, so they warm up meanwhile;
every other element is untouched (its update is exactly the optimizer's). Elements whose second moment is zero
legitimately (embedding rows of never-seen tokens: no gradient yet) are ramped as well, which only matters once
they first get a gradient. A resume whose moments are complete installs nothing.
"""

from __future__ import annotations

import logging
import os
from collections import Counter

import torch

logger = logging.getLogger(__name__)


def _local(t):
    return t.to_local() if hasattr(t, "to_local") else t


def _torch_optimizers(optimizer) -> list:
    """The torch optimizers (objects with ``state`` and ``param_groups``) under a Megatron optimizer."""
    out, seen, todo = [], set(), [optimizer]
    while todo:
        o = todo.pop()
        if o is None or id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, torch.optim.Optimizer):
            out.append(o)
            continue
        found_subs = False
        for name in ("chained_optimizers", "optimizers"):
            subs = getattr(o, name, None)
            if isinstance(subs, (list, tuple)):
                todo.extend(subs)
                found_subs = True
        if found_subs:
            # ChainedOptimizer.optimizer asserts when it holds more than one optimizer.
            continue
        try:
            inner = getattr(o, "optimizer", None)
        except (AssertionError, AttributeError):
            inner = None
        if inner is not None:
            todo.append(inner)
    return out


def _is_dtensor(t) -> bool:
    return hasattr(t, "to_local") and hasattr(t, "placements")


def cold_parameters(torch_optimizer, reset_layout: bool = True) -> list:
    """``(param, mask)`` for the parameters of ``torch_optimizer`` whose loaded Adam state is cold on this rank;
    ``mask`` marks the cold elements of the local shard (None: all of it; "reset": all of it, the state zeroed).

    * The state is a plain tensor of the parameter's GLOBAL shape while the parameter is sharded (TE 2.10 FusedAdam
      under Megatron-FSDP: not a DTensor, so the fsdp_dtensor save keeps only one rank's copy and every rank loads
      that copy): untrustworthy as a whole. With ``reset_layout`` its exp_avg / exp_avg_sq are zeroed here and the
      whole local shard is cold.
    * Otherwise, the elements whose local exp_avg_sq (shaped like the local parameter) is zero.
    States shaped like neither (the Engram tables' row-wise second moment) are skipped."""
    cold = []
    for group in torch_optimizer.param_groups:
        for p in group["params"]:
            state = torch_optimizer.state.get(p)
            if not isinstance(state, dict):
                continue
            v = state.get("exp_avg_sq")
            if not isinstance(v, torch.Tensor):
                continue
            p_local = _local(p.data)
            if p_local.numel() == 0:
                continue
            if (reset_layout and not _is_dtensor(v) and tuple(v.shape) == tuple(p.shape)
                    and tuple(v.shape) != tuple(p_local.shape)):
                with torch.no_grad():
                    for name in ("exp_avg", "exp_avg_sq"):
                        t = state.get(name)
                        if isinstance(t, torch.Tensor):
                            _local(t).zero_()
                cold.append((p, "reset"))
                continue
            v = _local(v)
            if v.shape != p_local.shape:
                continue
            zero = v == 0
            if not bool(zero.any()):
                continue
            cold.append((p, None if bool(zero.all()) else zero))
    return cold


def _wrap_step(torch_optimizer, cold: list, steps: int) -> None:
    original = torch_optimizer.step
    progress = {"k": 0}

    def step(*args, **kwargs):
        progress["k"] += 1
        k = progress["k"]
        if k > steps:
            torch_optimizer.step = original  # done: the plain step from now on
            return original(*args, **kwargs)
        old = [_local(p.data).detach().clone() for p, _ in cold]
        result = original(*args, **kwargs)
        r = k / steps
        with torch.no_grad():
            for (p, mask), before in zip(cold, old):
                now = _local(p.data)
                if mask is None or isinstance(mask, str):
                    now.copy_(before + r * (now - before))
                else:
                    now.copy_(torch.where(mask, before + r * (now - before), now))
        return result

    torch_optimizer.step = step


def install_cold_adam_warmup(optimizer, steps: int | None = None) -> dict:
    """Detect cold-resumed Adam parameters under ``optimizer`` and ramp in their updates (see module doc).
    Returns {'params': n, 'reset': n, 'partial': n, 'elements': n, 'shapes': Counter} of this rank ('reset': states
    zeroed for their broken layout; 'partial': parameters with only some cold elements; 'elements': cold local elements; empty when nothing is cold)."""
    if steps is None:
        steps = int(os.environ.get("OELLM_COLD_ADAM_WARMUP", "100") or 0)
    summary = {"params": 0, "reset": 0, "partial": 0, "elements": 0, "shapes": Counter()}
    if steps <= 0 or optimizer is None:
        return summary
    for opt in _torch_optimizers(optimizer):
        cold = cold_parameters(opt)
        if not cold:
            continue
        _wrap_step(opt, cold, steps)
        summary["params"] += len(cold)
        summary["reset"] += sum(isinstance(mask, str) for _, mask in cold)
        summary["partial"] += sum(isinstance(mask, torch.Tensor) for _, mask in cold)
        summary["elements"] += sum(int(mask.sum()) if isinstance(mask, torch.Tensor) else _local(p.data).numel()
                                   for p, mask in cold)
        summary["shapes"].update(tuple(p.shape) for p, _ in cold)
    if summary["params"]:
        top = ", ".join(f"{s}x{n}" for s, n in summary["shapes"].most_common(6))
        logger.warning(
            "cold Adam state on resume: %d parameters (%d reset: non-DTensor state of a sharded parameter; %d partially "
            "zero; %d local elements); their updates are ramped in over %d steps (OELLM_COLD_ADAM_WARMUP). Shapes: %s",
            summary["params"], summary["reset"], summary["partial"], summary["elements"], steps, top)
    return summary
