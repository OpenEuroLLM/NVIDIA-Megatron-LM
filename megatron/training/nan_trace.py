"""Name the layer that first went non-finite when a NaN loss fires -- retrospectively, at no cost.

Why this shape (2026-11-25): the first version installed counting forward hooks on every module. That
worked but it ran `(~isfinite(x)).sum()` and `int()` on every module output of every forward, which
forces a host synchronisation at every module boundary: measured pjob .2492 at 39.1 s/step against
19.0 s/step without it (.2470), a 2x slowdown for a diagnostic that is only ever needed once.

The activations are already retained for the backward, so nothing has to be observed while training
runs. On a NaN loss this module walks the autograd graph backwards from the loss, inspects the tensors
each node saved for backward, and reports:

* the bad saved activations in reverse (loss -> input), i.e. the node closest to the loss first,
* the ORIGIN: the bad node furthest from the loss, which is the earliest layer in forward order that
  held a non-finite value,
* a reverse scan of the parameters, so a corrupted LM head or expert weight is named outright.

If no saved activation is bad while the loss is NaN, the NaN was produced after the last saved tensor
(the fused cross-entropy), which is itself a decisive answer.

Steady-state cost: zero. The whole module is only called when the loss check fires.
"""

import torch

MAX_NODES = 4000  # bound the walk; a 64-layer MoE graph is well under this


def _bad_count(tensor) -> int:
    return int((~torch.isfinite(tensor)).sum())


def bad_saved_activations(output_tensor, limit: int = MAX_NODES) -> list[tuple[int, str, int, int]]:
    """(depth, node type, bad elements, elements) of every graph node whose saved tensors are not
    finite. ``depth`` is the distance from the loss, so sorting by it gives the reverse walk order."""
    if not torch.is_tensor(output_tensor) or output_tensor.grad_fn is None:
        return []
    found: list[tuple[int, str, int, int]] = []
    seen: set[int] = set()
    stack = [(output_tensor.grad_fn, 0)]
    while stack:
        node, depth = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if len(seen) > limit:
            break
        bad = numel = 0
        try:
            saved = node.saved_tensors or ()
        except Exception:  # a node whose saved tensors were freed (recompute) or are unavailable
            saved = ()
        for tensor in saved:
            if torch.is_tensor(tensor) and tensor.numel():
                numel += tensor.numel()
                bad += _bad_count(tensor)
        if bad:
            found.append((depth, type(node).__name__, bad, numel))
        for entry in node.next_functions or ():
            stack.append((entry[0], depth + 1))
    found.sort(key=lambda item: item[0])
    return found


def report(output_tensor, loss_mask, model=None, rank: int = 0) -> list[str]:
    """The lines naming the NaN source. Pure: returns text, prints nothing, so it is testable.

    ``model`` may be None (Megatron calls loss_func(output_tensor) from the pipeline schedule without
    one); the parameter scan is then skipped and the report says so.
    """
    lines = [
        f"[rank{rank}] NAN TRACE: loss is NaN; output {tuple(output_tensor.shape)} "
        f"non-finite {_bad_count(output_tensor)}/{output_tensor.numel()}, "
        f"masked tokens {int(loss_mask.sum())}"
    ]
    bad = bad_saved_activations(output_tensor)
    if bad:
        lines.append(f"[rank{rank}] bad saved activations, reverse (loss -> input), {len(bad)} node(s):")
        for depth, node_type, count, numel in bad:
            lines.append(f"[rank{rank}]   depth={depth} {node_type} non-finite={count}/{numel}")
        depth, node_type, count, numel = bad[-1]
        lines.append(
            f"[rank{rank}] ORIGIN (furthest from the loss = earliest in forward order): "
            f"depth={depth} {node_type} non-finite={count}/{numel}"
        )
    else:
        lines.append(
            f"[rank{rank}] no saved activation was non-finite: the NaN was produced after the last "
            f"saved tensor (fused cross-entropy / loss kernel), not by a layer"
        )
    if model is None:
        lines.append(f"[rank{rank}] no model reference: parameter scan skipped")
        return lines
    bad_params: list[str] = []
    scan_error = None
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
            f"[rank{rank}] all parameters finite: the weights are clean, so the NaN entered through "
            f"an activation on this device"
        )
    return lines
