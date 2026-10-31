# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""MEM_BREAKDOWN=1: where the GPU memory outside PyTorch's caching allocator goes.

The OOM message only gives the total ("non-PyTorch memory" = in use - reserved: CUDA context
and kernel images, NCCL communicators, MoonEP's VMM buffers and pools). This logs that number at
the start and end of the first train steps, and the delta of every MoonEP allocation, on the
ranks in MEM_BREAKDOWN_RANKS (comma list, default 0,8,16,24: one per stage at PP 4 x DP 8).
"""

import contextlib
import os

import torch

ENABLED = os.environ.get("MEM_BREAKDOWN", "0") == "1"
_RANKS = {int(r) for r in os.environ.get("MEM_BREAKDOWN_RANKS", "0,8,16,24").split(",")}
# MEM_BREAKDOWN_STEPS=N: report at the first N iterations (default 3). The allocator retries
# (num_alloc_retries: a failed cudaMalloc at the cap -> free the cached blocks -> retry) show
# allocator thrash directly, which sets in a few steps after the start
STEPS = int(os.environ.get("MEM_BREAKDOWN_STEPS", "3"))
_totals = {}


def _rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def non_torch_bytes() -> int:
    """Device memory in use that the caching allocator did not reserve (this process, if it
    is the only one on the GPU)."""
    free, total = torch.cuda.mem_get_info()
    return total - free - torch.cuda.memory_reserved()


def report(tag: str) -> None:
    if not ENABLED or _rank() not in _RANKS:
        return
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    g = 2**30
    parts = ", ".join(f"{k} {v / g:.2f}" for k, v in sorted(_totals.items()))
    print(
        f"[mem_breakdown] rank {_rank()} {tag}: in use {(total - free) / g:.2f} of {total / g:.2f} GiB, reserved "
        f"{torch.cuda.memory_reserved() / g:.2f}, allocated {torch.cuda.memory_allocated() / g:.2f}, "
        f"non-PyTorch {non_torch_bytes() / g:.2f}, max reserved {torch.cuda.max_memory_reserved() / g:.2f}, "
        f"alloc retries {torch.cuda.memory_stats().get('num_alloc_retries', 0)} "
        f"(measured: {parts or '-'})",
        flush=True,
    )


def baseline(name: str) -> None:
    """Record the non-PyTorch memory in use now as ``name`` (first call: the CUDA context, which
    this very query creates if it does not exist yet)."""
    if ENABLED:
        torch.cuda.synchronize()
        _totals[name] = non_torch_bytes()


@contextlib.contextmanager
def measure(name: str):
    """Add the non-PyTorch memory the block allocates to ``name``'s total."""
    if not ENABLED:
        yield
        return
    torch.cuda.synchronize()
    before = non_torch_bytes()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        _totals[name] = _totals.get(name, 0) + non_torch_bytes() - before
