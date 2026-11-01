# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""--phase-transition-iterations with --phase-transition-variable-batch: a sequence-length curriculum at
constant tokens per step (another global batch size in every phase)."""

from types import SimpleNamespace

import pytest

from megatron.training import checkpointing
from megatron.training.global_vars import set_args
from megatron.training.training import get_train_valid_test_num_samples


def _sample_args(**kw):
    args = SimpleNamespace(iteration=0, train_samples=None, train_iters=1000, eval_interval=250, eval_iters=10,
                           global_batch_size=16, skip_train=False, start_eval_at_iter=None, full_validation=False,
                           phase_transition_iterations=None)
    for k, v in kw.items():
        setattr(args, k, v)
    return args


@pytest.mark.parametrize("iteration,gbs,expect", [
    (0, 256, 20 * 256),          # phase 1 (iterations 0-20) at 4k: gbs 256
    (20, 64, 80 * 64),           # phase 2 (20-100) at 16k: gbs 64
    (100, 16, 900 * 16),         # phase 3 (100-1000) at 64k: gbs 16
])
def test_train_samples_in_current_phase_iteration_based(iteration, gbs, expect):
    """--train-iters (train_samples None): the phase's sample count, in this phase's global batch size."""
    set_args(_sample_args(iteration=iteration, global_batch_size=gbs, phase_transition_iterations=[20, 100]))
    train, _, _ = get_train_valid_test_num_samples()
    assert train == expect


def _ckpt_args(**kw):
    a = SimpleNamespace(num_layers=4, hidden_size=64, num_attention_heads=4, add_position_embedding=False,
                        vocab_file=None, data_parallel_random_init=False, global_batch_size=256,
                        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, use_dist_ckpt=True)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_check_checkpoint_args_variable_batch(monkeypatch):
    monkeypatch.setattr(checkpointing, "get_checkpoint_version", lambda: 3.0)
    ckpt = _ckpt_args(global_batch_size=256)
    set_args(_ckpt_args(global_batch_size=16, phase_transition_iterations=[20, 100],
                        phase_transition_variable_batch=False))
    with pytest.raises(AssertionError, match="global_batch_size"):
        checkpointing.check_checkpoint_args(ckpt)
    set_args(_ckpt_args(global_batch_size=16, phase_transition_iterations=[20, 100],
                        phase_transition_variable_batch=True))
    checkpointing.check_checkpoint_args(ckpt)          # another batch size per phase: allowed


class _Sched:
    def __init__(self, num_steps):
        self.num_steps, self.increments = num_steps, []

    def step(self, increment):
        self.num_steps += increment
        self.increments.append(increment)


def test_rescale_scheduler_for_phase():
    s = _Sched(num_steps=20 * 256)                     # saved at iteration 20 of the 4k phase (gbs 256)
    checkpointing.rescale_scheduler_for_phase(s, iteration=20, global_batch_size=64)
    assert s.num_steps == 20 * 64 and s.increments == [0]   # same iteration, this phase's samples; lr re-set
    s = _Sched(num_steps=100 * 16)                     # nothing to change
    checkpointing.rescale_scheduler_for_phase(s, iteration=100, global_batch_size=16)
    assert s.num_steps == 1600


@pytest.mark.parametrize("iteration,gbs,expect", [
    (0, 16384, 3 * 10 * 16384),        # v3 phase 1 (0-4737, 4k): evals at 2000, 4000 + the exit one
    (4737, 4096, 10 * 10 * 4096),      # phase 2 (4737-23691, 16k): 6000 ... 22000 + the exit one
    (23691, 1024, 108 * 10 * 1024),    # phase 3 (23691-236733, 64k): 24000 ... 236000 + the final one
    (10000, 4096, 10 * 10 * 4096),     # a restart within phase 2 sizes the same index
])
def test_valid_samples_per_variable_batch_phase(iteration, gbs, expect):
    """Variable-batch phases size the validation index for the phase's own evaluations, not for the
    whole run's at the phase's batch size (119 x 10 x 16384 = 19.5M 4k samples in phase 1)."""
    set_args(_sample_args(iteration=iteration, global_batch_size=gbs, train_iters=236733,
                          eval_interval=2000, phase_transition_iterations=[4737, 23691],
                          phase_transition_variable_batch=True))
    _, valid, _ = get_train_valid_test_num_samples()
    assert valid == expect


def test_valid_samples_fixed_batch_phases_unchanged():
    set_args(_sample_args(iteration=20, global_batch_size=64, phase_transition_iterations=[20, 100]))
    _, valid, _ = get_train_valid_test_num_samples()
    assert valid == (1000 // 250 + 1) * 10 * 64


def test_phase_eval_points_resume():
    from megatron.training.training import _phase_eval_points

    args = _sample_args(eval_interval=2000, start_eval_at_iter=None)
    # evaluations done in phase 2 when resuming at 10000: 6000 and 8000 and 10000
    assert _phase_eval_points(args, 4737, 10000) - 1 == 3
    assert _phase_eval_points(args, 4737, 4737) == 0  # an empty range (the loader starts at 0)
