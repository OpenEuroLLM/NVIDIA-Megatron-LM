"""Release host memory before a big allocation.

Why: the v4b main run was killed twice by the host OOM killer at a 1000-step checkpoint save
(pjob .2407 at 57000, pjob .2470 at 59000: 13 and 3+ tasks OOM-killed, exitcode -9), while its steady
resident memory is ~310 GB against a ~500 GB limit. `gc.collect()` drops unreachable Python objects;
`malloc_trim(0)` then returns the freed glibc arenas to the OS, which `gc.collect()` alone never does.

This is headroom, not a fix: it cannot free memory held live by the dataloader worker processes, and a
page-cache burst from writing the checkpoint is invisible to it. It is cheap (a few seconds) and gated
by `--gc-before-save` / `--gc-before-val`, both off by default.
"""

import ctypes
import gc
import time


def gc_and_trim(label: str = "") -> str:
    """gc.collect() then malloc_trim(0). Returns a one-line report (the caller prints it on rank 0)."""
    started = time.time()
    collected = gc.collect()
    trimmed = False
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        trimmed = True
    except Exception:  # not glibc, or trim unavailable: the save must go on
        pass
    return (
        f"host_gc[{label or 'gc'}]: collected {collected} unreachable object(s), "
        f"malloc_trim={'done' if trimmed else 'unavailable'}, "
        f"{time.time() - started:.2f} s"
    )
