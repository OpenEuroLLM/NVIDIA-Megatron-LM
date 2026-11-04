# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Full validation: every pass reads its set from the start, and release() ends the loader's
(persistent) worker processes, so they hold no host memory between evaluations."""

import torch

from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.training.training import _FullValidationIterator, _release_valid_iterator


def _loader(n=6, workers=2):
    return torch.utils.data.DataLoader(
        list(range(n)), batch_size=1, num_workers=workers, persistent_workers=workers > 0
    )


def _take(it, k):
    return [int(next(it)) for _ in range(k)]


def test_each_pass_starts_the_set_anew_and_release_ends_the_workers():
    dl = _loader()
    it = RerunDataIterator(_FullValidationIterator(dl))
    assert _take(it, 6) == list(range(6))
    workers = list(dl._iterator._workers)
    assert workers and all(w.is_alive() for w in workers)
    _release_valid_iterator(it)
    for w in workers:
        w.join(timeout=10)
    assert not any(w.is_alive() for w in workers)
    assert dl._iterator is None
    # the next evaluation: the set from its start, new workers
    assert _take(it, 3) == [0, 1, 2]
    assert all(w.is_alive() for w in dl._iterator._workers)
    _release_valid_iterator(it)


def test_a_short_rank_wraps_within_a_pass():
    """eval_iters is the max over DP ranks: a rank with fewer samples wraps around."""
    it = _FullValidationIterator(_loader(n=4))
    assert _take(it, 6) == [0, 1, 2, 3, 0, 1]
    it.release()
    assert _take(it, 2) == [0, 1]
    it.release()


def test_without_workers_and_release_without_a_pass():
    it = _FullValidationIterator(_loader(n=3, workers=0))
    it.release()  # nothing started yet
    assert _take(it, 3) == [0, 1, 2]
    it.release()
    assert _take(it, 1) == [0]
