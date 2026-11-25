# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for megatron/training/nan_trace.py, the --nan-trace diagnostic.

The run that motivates it: the v4b main run dies about once per 175 steps with "found NaN in local
forward loss calculation" on one rank, on a different node every time. --nan-trace answers *where* the
value first went non-finite, so the tests pin the three answers it can give: the origin module, the
reverse (loss -> input) order, and a non-finite parameter instead of an activation.

Written for pytest (every function is a ``test_*``), and also runnable without it, because the
training container ships torch but no pytest:

    python submodules/Megatron-LM/tests/unit_tests/training/test_nan_trace.py
"""

import argparse
import importlib.util
import pathlib
import traceback

import torch

# Load nan_trace.py by path instead of `from megatron.training import nan_trace`: the package
# __init__ pulls megatron.core's compiled extensions (packed_doc_static and friends), which exist in
# the container and in CI but not on a CPU login node. Loading the file needs torch only, so the same
# test runs in both places.
_MEGATRON = pathlib.Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "nan_trace_under_test", _MEGATRON / "megatron" / "training" / "nan_trace.py"
)
nan_trace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nan_trace)


class _Poison(torch.nn.Module):
    """Identity until poisoned, then adds NaN: a layer that introduces the NaN."""

    def __init__(self):
        super().__init__()
        self.poison = False

    def forward(self, x):
        if self.poison:
            return x + float("nan")
        return x


def test_cli_flag_is_a_real_argument():
    """The diagnostic is switched on by --nan-trace, not by an environment variable."""
    nan_trace.reset()
    source = (_MEGATRON / "megatron" / "training" / "arguments.py").read_text()
    group = source[source.index("def _add_diagnostics_args"):]
    assert "'--nan-trace'" in group, "the flag must be registered in the diagnostics group"
    try:  # the real parser needs megatron.core's compiled extensions
        from megatron.training import arguments
    except Exception:
        return
    parser = argparse.ArgumentParser()
    arguments._add_diagnostics_args(parser)
    assert parser.parse_args([]).nan_trace is False
    assert parser.parse_args(["--nan-trace"]).nan_trace is True


def test_install_disabled_installs_nothing():
    nan_trace.reset()
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    assert nan_trace.install(model, enabled=False) == 0
    lines = nan_trace.report(model, torch.zeros(1, 4), torch.ones(1), rank=0)
    assert any("no module output was non-finite" in line for line in lines)


def test_install_counts_modules_and_is_idempotent():
    nan_trace.reset()
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    assert nan_trace.install(model, enabled=True) >= 2
    assert nan_trace.install(model, enabled=True) == 0


def test_report_names_the_origin_and_walks_in_reverse():
    nan_trace.reset()
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), _Poison(), torch.nn.Linear(4, 2))
    assert nan_trace.install(model, enabled=True) >= 3
    model[1].poison = True
    logits = model(torch.zeros(1, 4))
    assert not torch.isfinite(logits).all()

    lines = nan_trace.report(model, logits, torch.ones(1), rank=7)
    text = "\n".join(lines)

    # only the failing rank reports, and every line is attributed to it
    assert lines and all(line.startswith("[rank7]") for line in lines)
    # the origin is the first bad module in FORWARD order: the _Poison layer, not the linear after it
    origin = next(line for line in lines if "ORIGIN" in line)
    assert "_Poison" in origin and "depth=2" in origin
    # reverse order: the outermost module (the Sequential, whose output is the logits and therefore
    # closest to the loss) is listed first, then '2', and the poison layer last
    trail = [line for line in lines if "non-finite=" in line and "ORIGIN" not in line]
    assert trail, text
    assert "Sequential" in trail[0], trail
    after_poison = next(i for i, line in enumerate(trail) if "'2'" in line)
    at_poison = next(i for i, line in enumerate(trail) if "'1'" in line)
    assert after_poison < at_poison, trail


def test_report_names_a_nonfinite_parameter():
    nan_trace.reset()
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with torch.no_grad():
        model[0].weight[0, 0] = float("nan")
    nan_trace.install(model, enabled=True)

    lines = nan_trace.report(model, torch.full((1, 2), float("nan")), torch.ones(1), rank=0)
    text = "\n".join(lines)
    assert "non-finite PARAMETERS" in text
    assert "0.weight" in text
    assert "all parameters finite" not in text


def test_report_blames_the_loss_kernel_when_no_module_is_bad():
    """A NaN that no module produced came from the fused cross-entropy, and the report says so."""
    nan_trace.reset()
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    nan_trace.install(model, enabled=True)

    lines = nan_trace.report(model, torch.full((1, 2), float("nan")), torch.ones(1), rank=0)
    assert any("after the last module output" in line for line in lines)
    assert any("all parameters finite" in line for line in lines)


def test_report_clears_the_trail():
    nan_trace.reset()
    model = torch.nn.Sequential(_Poison())
    nan_trace.install(model, enabled=True)
    model[0].poison = True
    logits = model(torch.zeros(1, 4))
    nan_trace.report(model, logits, torch.ones(1), rank=0)
    # a second report cannot repeat the previous iteration's modules
    again = nan_trace.report(model, logits, torch.ones(1), rank=0)
    assert any("no module output was non-finite" in line for line in again)


def test_trail_is_bounded():
    """A long run without a NaN must not grow the trail without limit."""
    nan_trace.reset()
    model = torch.nn.Sequential(_Poison())
    nan_trace.install(model, enabled=True)
    model[0].poison = True
    for _ in range(nan_trace.MAX_TRAIL + 20):
        model(torch.zeros(1, 4))
    lines = nan_trace.report(model, torch.zeros(1, 4), torch.ones(1), rank=0)
    recorded = next(line for line in lines if "recorded:" in line)
    assert f"{nan_trace.MAX_TRAIL} recorded" in recorded


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
