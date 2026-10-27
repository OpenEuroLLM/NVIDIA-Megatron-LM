# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Index caches shared by concurrent jobs (datasets/utils.py): one builder per entry at a time,
atomic writes, hard-linked copies left intact."""

import multiprocessing as mp
import os
import time

import numpy

from megatron.core.datasets.utils import atomic_numpy_save, atomic_write_text, cache_entry_lock


def _hold(path, base, started, go, hold_s, out):
    with cache_entry_lock(path, base):
        started.set()
        go.wait(60)  # the waiter is up and about to take the lock
        time.sleep(hold_s)
        atomic_numpy_save(os.path.join(path, f"{base}-x.npy"), numpy.arange(10))
        out.put(("released", time.time()))


def _wait(path, base, go, out):
    go.set()
    with cache_entry_lock(path, base):
        out.put(("acquired", time.time(), os.path.isfile(os.path.join(path, f"{base}-x.npy"))))


def test_lock_serializes_builders(tmp_path):
    ctx = mp.get_context("spawn")
    started, go, out = ctx.Event(), ctx.Event(), ctx.Queue()
    a = ctx.Process(target=_hold, args=(str(tmp_path), "h", started, go, 1.0, out))
    a.start()
    assert started.wait(60)
    b = ctx.Process(target=_wait, args=(str(tmp_path), "h", go, out))
    b.start()
    ev = dict((m[0], m[1:]) for m in (out.get(timeout=60), out.get(timeout=60)))
    a.join(30)
    b.join(30)
    # the second builder got the lock only after the first had written its entry
    assert ev["acquired"][0] >= ev["released"][0] and ev["acquired"][1]


def test_other_entries_do_not_wait(tmp_path):
    ctx = mp.get_context("spawn")
    started, go, out = ctx.Event(), ctx.Event(), ctx.Queue()
    a = ctx.Process(target=_hold, args=(str(tmp_path), "h1", started, go, 3.0, out))
    a.start()
    assert started.wait(60)
    b = ctx.Process(target=_wait, args=(str(tmp_path), "h2", go, out))
    b.start()
    ev = dict((m[0], m[1:]) for m in (out.get(timeout=60), out.get(timeout=60)))
    a.join(30)
    b.join(30)
    # a different entry is not held up by the first builder
    assert ev["acquired"][0] < ev["released"][0]


def test_atomic_writes_leave_hard_links_intact(tmp_path):
    p = tmp_path / "a.npy"
    atomic_numpy_save(str(p), numpy.arange(5))
    link = tmp_path / "b.npy"
    os.link(p, link)
    atomic_numpy_save(str(p), numpy.arange(7))
    assert numpy.load(link).tolist() == list(range(5))
    assert numpy.load(p).tolist() == list(range(7))
    atomic_write_text(str(tmp_path / "d.txt"), "desc")
    assert (tmp_path / "d.txt").read_text() == "desc"
    assert not [f for f in os.listdir(tmp_path) if ".tmp-" in f]
