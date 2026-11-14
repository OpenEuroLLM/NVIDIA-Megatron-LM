# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Ramp in the updates of Adam-handled parameters that resume with a cold (all-zero) second moment.

Checkpoints written on JUPITER (torch 2.11 DCP, Megatron-FSDP fsdp_dtensor) were found to hold all-zero
``exp_avg`` / ``exp_avg_sq`` for most small Adam-handled parameters (mHC coefficients, norms, CKDA conv / gate
vectors; docs/recall_surgery_ambitious.md §14.6), while the live optimizer state is fine. A resume then restarts
those parameters' Adam statistics from zero at a constant learning rate -- ~0.45 lr * sign(g) on every element of
hundreds of tensors at once -- and the loss jumped from ~3 to 30-120 after the first update, recovering only over
~100 steps.

``install_cold_adam_warmup(optimizer)`` finds, after the optimizer state has been loaded, every parameter whose local
``exp_avg_sq`` shard is non-empty and all zero, and wraps the step of the torch optimizer that holds it: for the next
``OELLM_COLD_ADAM_WARMUP`` steps (default 100; 0 disables) the applied update of those parameters is scaled by
k / N (k = 1, 2, ...). Their moments evolve normally, so they warm up meanwhile; every other parameter is untouched
(its update is exactly the optimizer's). A resume whose moments are complete installs nothing.
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


def cold_parameters(torch_optimizer) -> list:
    """Parameters of ``torch_optimizer`` whose local exp_avg_sq shard is non-empty and all zero."""
    cold = []
    for group in torch_optimizer.param_groups:
        for p in group["params"]:
            state = torch_optimizer.state.get(p)
            if not isinstance(state, dict):
                continue
            v = state.get("exp_avg_sq")
            if not isinstance(v, torch.Tensor):
                continue
            v = _local(v)
            if v.numel() > 0 and not bool(v.any()):
                cold.append(p)
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
        old = [_local(p.data).detach().clone() for p in cold]
        result = original(*args, **kwargs)
        r = k / steps
        with torch.no_grad():
            for p, before in zip(cold, old):
                now = _local(p.data)
                now.copy_(before + r * (now - before))
        return result

    torch_optimizer.step = step


def install_cold_adam_warmup(optimizer, steps: int | None = None) -> dict:
    """Detect cold-resumed Adam parameters under ``optimizer`` and ramp in their updates (see module doc).
    Returns {'params': n, 'elements': n, 'shapes': Counter} of this rank (empty when nothing is cold)."""
    if steps is None:
        steps = int(os.environ.get("OELLM_COLD_ADAM_WARMUP", "100") or 0)
    summary = {"params": 0, "elements": 0, "shapes": Counter()}
    if steps <= 0 or optimizer is None:
        return summary
    for opt in _torch_optimizers(optimizer):
        cold = cold_parameters(opt)
        if not cold:
            continue
        _wrap_step(opt, cold, steps)
        summary["params"] += len(cold)
        summary["elements"] += sum(_local(p.data).numel() for p in cold)
        summary["shapes"].update(tuple(p.shape) for p in cold)
    if summary["params"]:
        top = ", ".join(f"{s}x{n}" for s, n in summary["shapes"].most_common(6))
        logger.warning(
            "cold Adam state on resume: %d parameters (%d local elements) have an all-zero exp_avg_sq; their "
            "updates are ramped in over %d steps (OELLM_COLD_ADAM_WARMUP). Shapes: %s",
            summary["params"], summary["elements"], steps, top)
    return summary
