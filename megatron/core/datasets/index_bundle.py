# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""A bundle of cached dataset index arrays (--data-cache-path ``.npy`` files) in one file.

A blend's start opens every cached index of every dataset on every rank: ~3,200 files per rank
for the 531-dataset v4b blend (GPTDataset document / sample / shuffle index per split, the
BlendedDataset indices), all ranks at once, 12-25 min on GPFS at 256 nodes. ``safe_numpy_load``
also serializes the loads of a rank's builder threads behind one lock.

A bundle holds exactly those arrays, byte for byte, in ``<prefix>.bin`` (each array 4 KiB aligned)
with ``<prefix>.json`` naming them by their cache file name (the name carries the description
hash, so a name identifies the content). A rank maps the one file and takes read-only views; an
array the bundle lacks is loaded from its own file. Nothing about the indices changes: the same
arrays, in the same order, reach the same datasets.

Built ahead on one node: ``tools/prepare_cache.py --write-index-bundle <prefix>`` records every
cached index the run's datasets load, writes the bundle and checks it (byte equality per array,
then the datasets built from the bundle against those built from the files).
"""

import json
import logging
import os
import threading
from typing import Dict, List, Optional

import numpy

logger = logging.getLogger(__name__)

_FORMAT = 1
_ALIGN = 4096

_bundles: Dict[str, "IndexBundle"] = {}
_bundles_lock = threading.Lock()
_recorded: Optional[List[str]] = None
_recorded_lock = threading.Lock()
_counts = {"bundle": 0, "file": 0}


class IndexBundle:
    """The arrays of one bundle, as read-only views of one memory map."""

    def __init__(self, prefix: str):
        with open(prefix + ".json") as f:
            meta = json.load(f)
        assert meta.get("format") == _FORMAT, f"{prefix}.json: unknown bundle format {meta.get('format')}"
        self.prefix = prefix
        self.cache_dir = os.path.normpath(meta["cache_dir"])
        self.entries: Dict[str, dict] = meta["entries"]
        self._map = numpy.memmap(prefix + ".bin", mode="r", dtype=numpy.uint8)
        assert self._map.shape[0] == meta["nbytes"], f"{prefix}.bin: size differs from {prefix}.json"

    def get(self, path: str) -> Optional[numpy.ndarray]:
        """The array of cache file ``path``, or None if the bundle lacks it."""
        if os.path.normpath(os.path.dirname(path)) != self.cache_dir:
            return None
        e = self.entries.get(os.path.basename(path))
        if e is None:
            return None
        return numpy.ndarray(
            tuple(e["shape"]),
            dtype=numpy.dtype(e["dtype"]),
            buffer=self._map,
            offset=e["offset"],
            order="F" if e["fortran_order"] else "C",
        )


def _bundle(prefix: str) -> IndexBundle:
    b = _bundles.get(prefix)
    if b is None:
        with _bundles_lock:
            b = _bundles.get(prefix)
            if b is None:
                b = _bundles[prefix] = IndexBundle(prefix)
    return b


def in_bundle(prefix: Optional[str], paths: List[str]) -> bool:
    """Whether bundle ``prefix`` holds every one of ``paths`` (a cache hit without a file check)."""
    if not prefix:
        return False
    b = _bundle(prefix)
    return all(b.get(p) is not None for p in paths)


def load_cached_index(path: str, prefix: Optional[str] = None) -> numpy.ndarray:
    """A cached index array: from bundle ``prefix`` if it holds it, else memory-mapped from its file.

    The file path loads numeric arrays without the pickle patch (and its process-wide lock);
    an index written with pickled objects falls back to ``safe_numpy_load``.
    """
    if _recorded is not None:
        with _recorded_lock:
            _recorded.append(path)
    if prefix:
        a = _bundle(prefix).get(path)
        if a is not None:
            with _recorded_lock:
                _counts["bundle"] += 1
            return a
    with _recorded_lock:
        _counts["file"] += 1
    try:
        return numpy.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError:
        from megatron.core.safe_globals import safe_numpy_load

        return safe_numpy_load(path, allow_pickle=True, mmap_mode="r")


def load_counts() -> Dict[str, int]:
    """How many cached indices this process took from a bundle and from files."""
    return dict(_counts)


def start_recording() -> None:
    """Record the path of every cached index loaded from now on (see ``recorded_paths``)."""
    global _recorded
    _recorded = []


def recorded_paths() -> List[str]:
    """The cached indices loaded since ``start_recording``, each once, in first-load order."""
    return list(dict.fromkeys(_recorded or []))


def write_bundle(prefix: str, paths: List[str], cache_dir: str) -> dict:
    """Write the arrays of ``paths`` (cache files in ``cache_dir``) as bundle ``prefix``.

    Written under temporary names and renamed (the .json last), then every array of the bundle is
    compared byte for byte with its file. Returns the bundle's metadata.
    """
    cache_dir = os.path.normpath(cache_dir)
    entries, offset = {}, 0
    tmp_bin, tmp_json = f"{prefix}.bin.tmp{os.getpid()}", f"{prefix}.json.tmp{os.getpid()}"
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    with open(tmp_bin, "wb") as f:
        for p in paths:
            assert os.path.normpath(os.path.dirname(p)) == cache_dir, f"{p} is not in {cache_dir}"
            name = os.path.basename(p)
            assert name not in entries, f"{name} twice"
            a = numpy.load(p, mmap_mode="r", allow_pickle=False)
            pad = (-offset) % _ALIGN
            f.write(b"\0" * pad)
            offset += pad
            fortran = bool(a.flags.f_contiguous and not a.flags.c_contiguous)
            raw = numpy.asarray(a).tobytes(order="F" if fortran else "C")
            f.write(raw)
            entries[name] = {
                "offset": offset,
                "dtype": a.dtype.str,
                "shape": list(a.shape),
                "fortran_order": fortran,
                "nbytes": len(raw),
            }
            offset += len(raw)
    meta = {"format": _FORMAT, "cache_dir": cache_dir, "nbytes": offset, "entries": entries}
    with open(tmp_json, "w") as f:
        json.dump(meta, f)
    os.replace(tmp_bin, prefix + ".bin")
    os.replace(tmp_json, prefix + ".json")
    with _bundles_lock:
        _bundles.pop(prefix, None)
    b = _bundle(prefix)
    for p in paths:
        a, v = numpy.load(p, mmap_mode="r", allow_pickle=False), b.get(p)
        assert v is not None and v.dtype == a.dtype and v.shape == a.shape, f"{p}: bundle entry differs"
        assert numpy.array_equal(v, a), f"{p}: bundle bytes differ from the file"
    return meta
