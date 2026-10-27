# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
import logging
import os
import socket
from enum import Enum
from typing import List, Optional, Tuple

import numpy

from ..utils import log_single_rank

logger = logging.getLogger(__name__)


class Split(Enum):
    train = 0
    valid = 1
    test = 2


def compile_helpers():
    """Compile C++ helper functions at runtime. Make sure this is invoked on a single process."""
    import os
    import subprocess

    command = ["make", "-C", os.path.abspath(os.path.dirname(__file__))]
    if subprocess.run(command).returncode != 0:
        import sys

        log_single_rank(logger, logging.ERROR, "Failed to compile the C++ dataset helper functions")
        sys.exit(1)


def normalize(weights: List[float]) -> List[float]:
    """Do non-exponentiated normalization

    Args:
        weights (List[float]): The weights

    Returns:
        List[float]: The normalized weights
    """

    w = numpy.array(weights, dtype=numpy.float64)
    w_sum = numpy.sum(w)
    w = (w / w_sum).tolist()
    return w


def get_blend_from_list(
    blend: Optional[List[str]],
) -> Optional[Tuple[List[str], Optional[List[float]]]]:
    # pylint: disable=line-too-long
    """Get the blended_megatron_dataset_config.BlendedMegatronDatasetConfig blend
    from the blend list

    Args:
        blend (Optional[List[str]]): The blend list, which can be either
            (1) a list of prefixes, e.g. ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], or
            (2) a flattened, zipped list of weights and prefixes, e.g. ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]

    Returns:
        Optional[Tuple[List[str], Optional[List[float]]]]: The blend, consisting of a list of dataset prefixes and optionally a list of dataset weights, e.g. [["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], [30.0, 70.0]].
    """
    # pylint: enable=line-too-long
    if blend is None:
        return None

    if len(blend) % 2 == 1:
        weight_per_dataset = None
        raw_prefix_per_dataset = blend
    else:
        raw_weight_per_dataset, raw_prefix_per_dataset = zip(
            *[(blend[i], blend[i + 1]) for i in range(0, len(blend), 2)]
        )

        weight_per_dataset = []
        for rwpd in raw_weight_per_dataset:
            try:
                weight = float(rwpd)
            except ValueError:
                weight = None
            weight_per_dataset.append(weight)

        is_none = map(lambda _: _ is None, weight_per_dataset)
        if any(is_none):
            assert all(is_none)
            weight_per_dataset = None
            raw_prefix_per_dataset = blend

    prefix_per_dataset = [rppd.strip() for rppd in raw_prefix_per_dataset]

    return prefix_per_dataset, weight_per_dataset


# Index caches shared by concurrent jobs (one cache directory per data recipe): the entries are
# addressed by the description hash, so jobs with different settings never collide, and jobs with
# the same settings need the same files. Two jobs that miss the same entry at once used to write
# the same files in place, and a third (or the slower writer itself) could load a half-written
# array -- runs died silently. Now one job builds an entry at a time (an exclusive POSIX lock per
# entry; GPFS honours it across nodes; the next job re-checks and loads), and every file is
# written to a temporary name and renamed into place, the description last (it completes the hit
# check), so a reader sees either no entry or a complete one. A rename also leaves a hard-linked
# copy of a cache untouched.


@contextlib.contextmanager
def cache_entry_lock(path_to_cache: str, base: str):
    """Hold an exclusive lock on the cache entry ``base`` while it is checked and built."""
    os.makedirs(path_to_cache, exist_ok=True)
    fd = os.open(os.path.join(path_to_cache, f"{base}.lock"), os.O_CREAT | os.O_RDWR, 0o664)
    locked = False
    try:
        import fcntl

        try:
            fcntl.lockf(fd, fcntl.LOCK_EX)
            locked = True
        except OSError as e:  # a filesystem without POSIX locks: atomic writes only
            log_single_rank(
                logger, logging.WARNING, f"cache lock unavailable ({e}); building unlocked"
            )
        yield
    finally:
        if locked:
            fcntl.lockf(fd, fcntl.LOCK_UN)
        os.close(fd)


def _tmp_path(path: str) -> str:
    return f"{path}.tmp-{socket.gethostname()}-{os.getpid()}"


def atomic_numpy_save(path: str, array: numpy.ndarray) -> None:
    """numpy.save to ``path`` through a temporary file and a rename."""
    tmp = _tmp_path(path)
    with open(tmp, "wb") as f:
        numpy.save(f, array, allow_pickle=True)
    os.replace(tmp, path)


def atomic_write_text(path: str, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file and a rename."""
    tmp = _tmp_path(path)
    with open(tmp, "wt") as f:
        f.write(text)
    os.replace(tmp, path)
