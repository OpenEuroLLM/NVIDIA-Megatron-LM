"""MegatronPretrainingReferenceLayoutSampler: the reference layout's samples per step, any layout."""

import itertools
import sys
import types

import pytest
import torch

# the samplers module imports megatron.training (args); only the sampler classes are tested here
from megatron.training.datasets.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingReferenceLayoutSampler,
)


def _steps_reference(total, gbs, dp, mbs, sharding, consumed, n_steps):
    """Per step: per rank the list of samples, from the original cyclic sampler (re-created at
    every epoch end, as Megatron's cyclic iterator does)."""
    m_per_step = gbs // (mbs * dp)
    iters = [None] * dp
    state = [consumed] * dp
    out = []
    for _ in range(n_steps):
        step = []
        for r in range(dp):
            got = []
            for _ in range(m_per_step):
                while True:
                    if iters[r] is None:
                        s = MegatronPretrainingRandomSampler(
                            None, total, state[r], mbs, r, dp, sharding
                        )
                        iters[r] = (s, iter(s))
                    try:
                        mb = next(iters[r][1])
                        state[r] = iters[r][0].consumed_samples
                        break
                    except StopIteration:
                        iters[r] = None
                got.append(mb)
            step.append(got)
        out.append(step)
    return out


def _steps_new(total, gbs, dp, mbs, sharding, consumed, n_steps, ref_dp, ref_mbs):
    m_per_step = gbs // (mbs * dp)
    its = [
        iter(
            MegatronPretrainingReferenceLayoutSampler(
                None, total, consumed, mbs, gbs, r, dp, sharding, ref_dp, ref_mbs
            )
        )
        for r in range(dp)
    ]
    return [[[next(it) for _ in range(m_per_step)] for it in its] for _ in range(n_steps)]


def _flat(step):
    return sorted(x for rank in step for mb in rank for x in mb)


CASES = [
    # (total, gbs, ref_dp, ref_mbs) -- totals chosen so steps straddle epoch ends
    (1000, 32, 8, 2),
    (997, 48, 4, 4),
    (640, 64, 16, 1),
]


@pytest.mark.parametrize("sharding", [True, False])
@pytest.mark.parametrize("total,gbs,ref_dp,ref_mbs", CASES)
def test_same_layout_is_exact(total, gbs, ref_dp, ref_mbs, sharding):
    for consumed in (0, gbs * 3):
        ref = _steps_reference(total, gbs, ref_dp, ref_mbs, sharding, consumed, 40)
        new = _steps_new(total, gbs, ref_dp, ref_mbs, sharding, consumed, 40, ref_dp, ref_mbs)
        assert new == ref


@pytest.mark.parametrize("sharding", [True, False])
@pytest.mark.parametrize("total,gbs,ref_dp,ref_mbs", CASES)
def test_other_layouts_same_samples_per_step(total, gbs, ref_dp, ref_mbs, sharding):
    ref = _steps_reference(total, gbs, ref_dp, ref_mbs, sharding, 0, 40)
    layouts = [
        (dp, mbs)
        for dp, mbs in itertools.product([1, 2, 4, 8, 16, 32, 64], [1, 2, 4, 8])
        if gbs % (dp * mbs) == 0 and (dp, mbs) != (ref_dp, ref_mbs)
    ]
    assert layouts
    for dp, mbs in layouts:
        new = _steps_new(total, gbs, dp, mbs, sharding, 0, 40, ref_dp, ref_mbs)
        for s in range(40):
            assert _flat(new[s]) == _flat(ref[s]), (dp, mbs, s)
            assert all(len(mb) == mbs for rank in new[s] for mb in rank)


def test_resume_mid_run_matches_continuous():
    total, gbs, ref_dp, ref_mbs = 1000, 32, 8, 2
    full = _steps_new(total, gbs, 16, 1, True, 0, 30, ref_dp, ref_mbs)
    resumed = _steps_new(total, gbs, 16, 1, True, gbs * 12, 18, ref_dp, ref_mbs)
    assert resumed == full[12:]
    # switch layouts at step 12: 8 x 2 before, 16 x 1 after -- same samples per step as the
    # reference run throughout
    ref = _steps_reference(total, gbs, ref_dp, ref_mbs, True, 0, 30)
    for s in range(12, 30):
        assert _flat(resumed[s - 12]) == _flat(ref[s])


def test_rejects_non_step_boundary_and_bad_gbs():
    with pytest.raises(AssertionError):
        MegatronPretrainingReferenceLayoutSampler(None, 1000, 0, 1, 30, 0, 16, True, 8, 2)
    s = MegatronPretrainingReferenceLayoutSampler(None, 1000, 5, 1, 32, 0, 16, True, 8, 2)
    with pytest.raises(AssertionError):
        next(iter(s))


# ---- reference_global_batch_size: a smaller actual global batch on the reference run's stream ----


def _steps_small(total, gbs, dp, mbs, sharding, consumed, n_steps, ref_dp, ref_mbs, ref_gbs):
    m_per_step = gbs // (mbs * dp)
    its = [
        iter(
            MegatronPretrainingReferenceLayoutSampler(
                None, total, consumed, mbs, gbs, r, dp, sharding, ref_dp, ref_mbs, ref_gbs
            )
        )
        for r in range(dp)
    ]
    return [[[next(it) for _ in range(m_per_step)] for it in its] for _ in range(n_steps)]


def _ordered(step):
    """A step's samples in stream order: rank 0's micro-batches, then rank 1's, ..."""
    return [x for rank in step for mb in rank for x in mb]


SMALL_CASES = [
    # (total, ref_gbs, ref_dp, ref_mbs, gbs, dp, mbs): the v4 main run's shape scaled down
    # (reference 512 ranks x mbs 1, gbs 1024 -> arms 128 ranks x mbs 1, gbs 128) and others
    (5000, 64, 32, 1, 8, 8, 1),
    (5000, 64, 32, 1, 16, 8, 1),
    (997, 48, 4, 4, 12, 3, 2),
    (1000, 32, 8, 2, 8, 4, 2),
    (1000, 32, 8, 2, 32, 16, 1),  # same gbs, other layout: the old behaviour
    # phase 1 of the v4 main run scaled down: reference 512 ranks x mbs 8, gbs 8192 (2 reference
    # micro-batches per rank and step) -> arms 256 ranks x mbs 4, gbs 1024 (one micro-batch, 1/8)
    (3000, 64, 16, 2, 8, 8, 1),
    (3000, 64, 16, 2, 8, 2, 4),
]


@pytest.mark.parametrize("sharding", [True, False])
@pytest.mark.parametrize("total,ref_gbs,ref_dp,ref_mbs,gbs,dp,mbs", SMALL_CASES)
def test_small_batch_follows_reference_stream(total, ref_gbs, ref_dp, ref_mbs, gbs, dp, mbs, sharding):
    """k = ref_gbs / gbs small steps train on exactly the samples of one reference step, in the
    reference step's order (rank slices of the reference layout concatenated); across epoch ends too."""
    n_ref = 30
    ref = _steps_new(total, ref_gbs, ref_dp, ref_mbs, sharding, 0, n_ref, ref_dp, ref_mbs)
    k = ref_gbs // gbs
    small = _steps_small(total, gbs, dp, mbs, sharding, 0, n_ref * k, ref_dp, ref_mbs, ref_gbs)
    for s in range(n_ref):
        got = [x for t in range(k) for x in _ordered(small[s * k + t])]
        assert got == _ordered(ref[s]), (s,)
        assert all(len(mb) == mbs for t in range(k) for rank in small[s * k + t] for mb in rank)


def test_small_batch_resume_and_hand_back():
    """Resume from the reference run's checkpoint (consumed = 5 reference steps) at 1/8 of the batch,
    resume again from the small run's own checkpoint mid reference step, then switch back to the
    reference batch at a reference-step boundary: the stream continues the reference run exactly."""
    total, ref_gbs, ref_dp, ref_mbs = 5000, 64, 32, 1
    gbs, dp, mbs = 8, 8, 1
    ref = _steps_new(total, ref_gbs, ref_dp, ref_mbs, True, 0, 12, ref_dp, ref_mbs)
    c0 = 5 * ref_gbs
    a = _steps_small(total, gbs, dp, mbs, True, c0, 11, ref_dp, ref_mbs, ref_gbs)  # 11 small steps
    b = _steps_small(total, gbs, dp, mbs, True, c0 + 11 * gbs, 5, ref_dp, ref_mbs, ref_gbs)  # resume
    stream = [x for st in a + b for x in _ordered(st)]
    assert stream == _ordered(ref[5]) + _ordered(ref[6])
    # hand back: the reference layout and batch from consumed = 7 reference steps
    back = _steps_new(total, ref_gbs, ref_dp, ref_mbs, True, c0 + 16 * gbs, 3, ref_dp, ref_mbs)
    assert back == ref[7:10]


def test_small_batch_rejects_bad_reference():
    with pytest.raises(AssertionError):  # reference batch not a multiple of the reference layout
        MegatronPretrainingReferenceLayoutSampler(None, 1000, 0, 1, 8, 0, 8, True, 32, 1, 48)
    s = MegatronPretrainingReferenceLayoutSampler(None, 1000, 4, 1, 8, 0, 8, True, 32, 1, 64)
    with pytest.raises(AssertionError):  # consumed not a step boundary of the actual batch
        next(iter(s))
