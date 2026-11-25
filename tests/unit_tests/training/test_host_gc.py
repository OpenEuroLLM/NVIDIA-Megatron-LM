# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for megatron/training/host_gc.py and its two diagnostics flags.

The run that motivates it: the v4b main run was killed by the host OOM killer at the 57000 and 59000
checkpoint saves, so --gc-before-save / --gc-before-val call gc.collect() + malloc_trim(0) before the
allocations that tipped it over.

Written for pytest (every function is ``test_*``) and runnable without it (the training container has
torch but no pytest):

    python submodules/Megatron-LM/tests/unit_tests/training/test_host_gc.py
"""

import importlib.util
import pathlib
import traceback

_MEGATRON = pathlib.Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "host_gc_under_test", _MEGATRON / "megatron" / "training" / "host_gc.py"
)
host_gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(host_gc)


def test_cli_flags_are_real_arguments():
    """Both switches are CLI flags, not environment variables."""
    source = (_MEGATRON / "megatron" / "training" / "arguments.py").read_text()
    group = source[source.index("def _add_diagnostics_args"):]
    assert "'--gc-before-save'" in group
    assert "'--gc-before-val'" in group


def test_gc_before_val_runs_after_args_are_bound():
    """Regression for 2026-11-25: the hook was inserted before `args = get_args()`, so `args` was a
    local referenced before assignment and the first validation killed the run (pjob .2502, rc=143)."""
    source = (_MEGATRON / "megatron" / "training" / "training.py").read_text()
    body = source[source.index("def evaluate_and_print_results("):]
    body = body[: body.index("\ndef ", 1)] if "\ndef " in body[1:] else body
    bound = body.find("args = get_args()")
    used = body.find("gc_before_val")
    assert bound != -1, "evaluate_and_print_results must bind args"
    assert used != -1, "the gc-before-val hook is missing"
    assert bound < used, "gc_before_val must be read AFTER args = get_args()"


def test_gc_and_trim_reports_what_it_did():
    report = host_gc.gc_and_trim("before_save")
    assert "host_gc[before_save]" in report
    assert "collected" in report
    assert "malloc_trim=" in report
    assert report.rstrip().endswith("s"), report


def test_gc_and_trim_defaults_its_label():
    assert "host_gc[gc]" in host_gc.gc_and_trim()


def test_gc_and_trim_collects_a_reference_cycle():
    """The collector really runs: a cycle that only gc can free is reported as collected."""
    class Node:
        pass

    a, b = Node(), Node()
    a.peer, b.peer = b, a
    del a, b
    report = host_gc.gc_and_trim("cycle")
    collected = int(report.split("collected ")[1].split(" ")[0])
    assert collected >= 1, report


def test_gc_and_trim_is_cheap_and_repeatable():
    first = host_gc.gc_and_trim("a")
    second = host_gc.gc_and_trim("b")
    assert "host_gc[a]" in first and "host_gc[b]" in second


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
