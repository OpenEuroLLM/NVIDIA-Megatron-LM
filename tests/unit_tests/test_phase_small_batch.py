# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Small-batch arms from a variable-batch phase checkpoint (recall surgery, docs/recall_surgery_ambitious.md §10).

The ambitious v4 main run is in phase 2 (64k, global batch 1024 from iteration 23691; phase 1 ran at 8192).
An arm resumed from its checkpoint at global batch 128 must (1) build the main run's training dataset (sized
at the reference batch 1024, not 128 -> same index / cache entry / samples) and (2) start at the main run's
data position in the phase -- the consumed samples minus the phase start, NOT (iteration - 23691) x 128,
which would replay the phase's first eighth. Back at 1024 after k x 8 small steps it continues exactly.

    python -m pytest tests/unit_tests/test_phase_small_batch.py
"""

from types import SimpleNamespace

import pytest

import megatron.training.training as tr

P1_ITERS, P1_GBS, P2_GBS, TRAIN_ITERS = 23691, 8192, 1024, 236733
P2_START = P1_ITERS * P1_GBS  # 194,076,672: consumed samples at the phase-2 start


def _args(iteration, gbs, consumed, ref_gbs=None, phase_start=None):
    return SimpleNamespace(
        phase_transition_iterations=[P1_ITERS],
        phase_transition_variable_batch=True,
        iteration=iteration,
        global_batch_size=gbs,
        consumed_train_samples=consumed,
        data_sampler_reference_global_batch_size=ref_gbs,
        phase_start_consumed_samples=phase_start,
        train_samples=None,
        train_iters=TRAIN_ITERS,
        full_validation=True,
        skip_train=False,
        eval_iters=10,
        eval_interval=1000,
        start_eval_at_iter=None,
        eval_global_batch_size=gbs,
    )


def test_main_run_unchanged():
    it = 32928
    consumed = P2_START + (it - P1_ITERS) * P2_GBS  # 203,535,360 as logged by the main run
    assert consumed == 203535360
    a = _args(it, P2_GBS, consumed)
    assert tr.consumed_train_samples_in_phase(a) == (it - P1_ITERS) * P2_GBS
    # the flag gives the same position for the main run itself
    a = _args(it, P2_GBS, consumed, ref_gbs=P2_GBS, phase_start=P2_START)
    assert tr.consumed_train_samples_in_phase(a) == (it - P1_ITERS) * P2_GBS


def test_small_batch_position_and_hand_back():
    it0 = 33200
    consumed0 = P2_START + (it0 - P1_ITERS) * P2_GBS
    # without the flag: (iteration - phase start) x 128 -- an eighth of the way: replays seen samples
    wrong = tr.consumed_train_samples_in_phase(_args(it0, 128, consumed0))
    assert wrong == (it0 - P1_ITERS) * 128
    # with it: the main run's position at its checkpoint
    a = _args(it0, 128, consumed0, ref_gbs=P2_GBS, phase_start=P2_START)
    assert tr.consumed_train_samples_in_phase(a) == (it0 - P1_ITERS) * P2_GBS
    # the arm's own checkpoint 800 steps later (consumed advanced by 800 x 128)
    a = _args(it0 + 800, 128, consumed0 + 800 * 128, ref_gbs=P2_GBS, phase_start=P2_START)
    pos = tr.consumed_train_samples_in_phase(a)
    assert pos == (it0 - P1_ITERS) * P2_GBS + 800 * 128
    # hand back to the main run's batch at iteration it0 + 800: the next unseen sample = 100 main steps on
    b = _args(it0 + 800, P2_GBS, consumed0 + 800 * 128, ref_gbs=P2_GBS, phase_start=P2_START)
    assert tr.consumed_train_samples_in_phase(b) == (it0 + 100 - P1_ITERS) * P2_GBS
    assert tr.consumed_train_samples_in_phase(b) % P2_GBS == 0


def test_bad_phase_start_is_loud():
    it0 = 33200
    consumed0 = P2_START + (it0 - P1_ITERS) * P2_GBS
    with pytest.raises(AssertionError):
        tr.consumed_train_samples_in_phase(_args(it0, 128, consumed0, P2_GBS, phase_start=consumed0 + 1))
    with pytest.raises(AssertionError):  # inside the phase but nothing consumed
        tr.consumed_train_samples_in_phase(_args(it0, 128, consumed0, P2_GBS, phase_start=consumed0))


def test_dataset_sized_at_reference_batch(monkeypatch):
    it0 = 33200
    consumed0 = P2_START + (it0 - P1_ITERS) * P2_GBS
    main = _args(it0, P2_GBS, consumed0)
    arm = _args(it0, 128, consumed0, ref_gbs=P2_GBS, phase_start=P2_START)
    arm_unfixed = _args(it0, 128, consumed0)
    sizes = {}
    for name, a in (("main", main), ("arm", arm), ("arm_unfixed", arm_unfixed)):
        monkeypatch.setattr(tr, "get_args", lambda a=a: a)
        sizes[name] = tr.get_train_valid_test_num_samples()[0]
    assert sizes["main"] == (TRAIN_ITERS - P1_ITERS) * P2_GBS
    assert sizes["arm"] == sizes["main"]  # the main run's dataset (index, cache entry, samples)
    assert sizes["arm_unfixed"] == (TRAIN_ITERS - P1_ITERS) * 128  # another dataset
