"""Name the layer that first goes non-finite when a NaN loss fires.

Why this exists (2026-11-25): the v4b main run dies about once per 175 steps with "found NaN in local
forward loss calculation" on ONE rank, on a different node every time (9 events, 9 distinct nodes,
each node once). The check that fires looks at one rank's loss before the DP all-reduce, so the bad
value is local to that rank; the open question is where it first appears. A mask or packing bug would
be deterministic and would recur at the same point, and it does not -- the same iteration passes on
the next attempt -- so the answer has to come from the failing process itself.

``--nan-trace`` installs counting forward hooks: each module's output is counted for non-finite
elements once per forward, and only the counts are kept (never a reference), so autograd retention
and the host-memory profile are unchanged. When a NaN loss is detected, ``report()`` returns lines
that name, in this order:

* the bad module outputs in reverse (loss -> input), i.e. the module closest to the loss first,
* the first bad module in forward order -- the origin, the one whose input was still clean,
* a reverse scan of the parameters, so a corrupted LM head or expert weight is named outright.

If no module output is bad while the loss is NaN, the NaN was produced after the last module (the
fused cross-entropy), which is itself a decisive answer.

Off by default; ``install(..., enabled=False)`` does nothing.
"""

import torch

# Bound the trail to the most recent bad module outputs: a run that never hits a NaN must not grow
# this list without limit.
MAX_TRAIL = 128

_state: dict = {"installed": False, "trail": [], "handles": [], "model": None}


def install(model, enabled: bool = True) -> int:
    """Register the counting hooks once. Returns how many were installed (0 when disabled)."""
    if not enabled or model is None or _state["installed"]:
        return 0
    trail: list = []
    handles = []
    for depth, (name, module) in enumerate(model.named_modules()):
        handles.append(module.register_forward_hook(_hook(name, depth, trail)))
    _state.update(installed=True, trail=trail, handles=handles, model=model)
    return len(handles)


def reset() -> None:
    """Remove the hooks and forget the trail (tests, and a model replaced mid-run)."""
    for handle in _state["handles"]:
        handle.remove()
    _state.update(installed=False, trail=[], handles=[], model=None)


def _hook(name: str, depth: int, trail: list):
    def hook(module, inputs, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        bad = int((~torch.isfinite(tensor)).sum())
        if bad:
            trail.append((depth, name, type(module).__name__, bad, tensor.numel()))
            del trail[: -MAX_TRAIL]

    return hook


def report(model, output_tensor, loss_mask, rank: int = 0) -> list[str]:
    """The lines naming the NaN source. Pure: returns text, prints nothing, so it is testable.

    ``model`` may be None: Megatron calls ``loss_func(output_tensor)`` from the pipeline schedule
    without a model, so the model that ``install()`` saw is used in that case. Without either, the
    parameter scan is skipped and the report says so.
    """
    model = model if model is not None else _state.get("model")
    trail = list(_state["trail"])
    logits_bad = int((~torch.isfinite(output_tensor)).sum())
    lines = [
        f"[rank{rank}] NAN TRACE: loss is NaN; logits {tuple(output_tensor.shape)} "
        f"non-finite {logits_bad}/{output_tensor.numel()}, "
        f"masked tokens {int(loss_mask.sum())}"
    ]
    if trail:
        lines.append(f"[rank{rank}] bad module outputs, reverse (loss -> input), {len(trail)} recorded:")
        for depth, name, cls, bad, numel in reversed(trail):
            lines.append(f"[rank{rank}]   depth={depth} {cls} '{name}' non-finite={bad}/{numel}")
        first = trail[0]
        lines.append(
            f"[rank{rank}] ORIGIN (first bad module in forward order): depth={first[0]} "
            f"{first[2]} '{first[1]}' non-finite={first[3]}/{first[4]}"
        )
    else:
        lines.append(
            f"[rank{rank}] no module output was non-finite: the NaN was produced after the last "
            f"module output (fused cross-entropy / loss kernel), not by a layer"
        )
    bad_params: list[str] = []
    scan_error = None
    if model is None:
        lines.append(f"[rank{rank}] no model reference: parameter scan skipped")
        _state["trail"].clear()
        return lines
    try:
        for name, param in reversed(list(model.named_parameters())):
            if param.numel() and not torch.isfinite(param).all():
                bad_params.append(name)
                if len(bad_params) >= 8:
                    break
    except Exception as err:  # a sharded/DTensor parameter that isfinite cannot reduce over
        scan_error = err
    if bad_params:
        lines.append(
            f"[rank{rank}] non-finite PARAMETERS (reverse order, closest to the loss first): "
            f"{bad_params}"
        )
    elif scan_error is not None:
        lines.append(f"[rank{rank}] parameter scan failed: {scan_error}")
    else:
        lines.append(
            f"[rank{rank}] all parameters finite: the weights are clean, so the NaN entered "
            f"through an activation on this device"
        )
    _state["trail"].clear()
    return lines
