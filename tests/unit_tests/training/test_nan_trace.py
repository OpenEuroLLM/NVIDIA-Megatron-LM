# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for megatron/training/nan_trace.py -- the retrospective NaN source report.

Design constraint that the tests pin: the diagnostic must cost NOTHING while training runs. The first
version installed forward hooks that synchronised at every module boundary and doubled the step time
(39.1 s against 19.0 s, pjob .2492 vs .2470). So the tests check that the module exposes no hook
installation at all, and that the report is built from the activations already saved for the backward.

Written for pytest (every function is ``test_*``) and runnable without it (the training container has
torch but no pytest):

    python submodules/Megatron-LM/tests/unit_tests/training/test_nan_trace.py
"""

import importlib.util
import pathlib
import traceback

import torch

_MEGATRON = pathlib.Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "nan_trace_under_test", _MEGATRON / "megatron" / "training" / "nan_trace.py"
)
nan_trace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nan_trace)


class _PoisonSaved(torch.autograd.Function):
    """Saves a NaN tensor for the backward: the shape of a layer that produced a bad activation."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(torch.full_like(x, float("nan")))
        return x * 2

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out * 2


def test_no_hooks_are_installed():
    """The diagnostic must be retrospective: no forward hooks, no per-step synchronisation."""
    assert not hasattr(nan_trace, "install"), "install() would observe every forward"
    assert not hasattr(nan_trace, "_hook"), "a forward hook would synchronise every module output"


def test_bad_saved_activations_finds_the_poisoned_node():
    x = torch.ones(4, requires_grad=True)
    out = _PoisonSaved.apply(x).sum()
    bad = nan_trace.bad_saved_activations(out)
    assert bad, "the saved NaN must be found"
    depth, node_type, count, numel = bad[0]
    assert "_PoisonSaved" in node_type, bad
    assert count == numel == 4, bad


def test_report_names_the_origin_and_walks_in_reverse():
    x = torch.ones(4, requires_grad=True)
    out = _PoisonSaved.apply(x).sum()
    lines = nan_trace.report(out, torch.ones(4), model=None, rank=3)
    text = "\n".join(lines)

    assert lines and all(line.startswith("[rank3]") for line in lines), text
    assert "bad saved activations, reverse (loss -> input)" in text
    origin = next(line for line in lines if "ORIGIN" in line)
    assert "_PoisonSaved" in origin, origin
    # the parameter scan is skipped, and the report says so rather than pretending
    assert any("no model reference" in line for line in lines), text


def test_report_blames_the_loss_kernel_when_nothing_saved_is_bad():
    x = torch.ones(4, requires_grad=True)
    out = (x * 3).sum()  # every saved tensor is finite
    lines = nan_trace.report(torch.full((2,), float("nan")), torch.ones(2), model=None, rank=0)
    text = "\n".join(lines)
    assert any("after the last saved tensor" in line for line in lines), text
    assert "ORIGIN" not in text


def test_report_names_a_nonfinite_parameter():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with torch.no_grad():
        model[0].weight[0, 0] = float("nan")
    out = torch.zeros(2, requires_grad=True).sum()
    lines = nan_trace.report(out, torch.ones(2), model=model, rank=0)
    text = "\n".join(lines)
    assert "non-finite PARAMETERS" in text
    assert "0.weight" in text
    assert "all parameters finite" not in text


def test_ordering_is_by_distance_from_the_loss():
    """Two poisoned nodes at different depths: the shallow one comes first in the reverse walk."""
    x = torch.ones(4, requires_grad=True)
    shallow = _PoisonSaved.apply(x)
    out = (_PoisonSaved.apply(shallow)).sum()
    bad = nan_trace.bad_saved_activations(out)
    assert len(bad) >= 2, bad
    depths = [entry[0] for entry in bad]
    assert depths == sorted(depths), bad


if __name__ == "__main__":  # runnable in the training container, which has torch but no pytest
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
