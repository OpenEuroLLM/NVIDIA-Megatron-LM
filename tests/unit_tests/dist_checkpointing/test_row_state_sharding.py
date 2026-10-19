# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Per-row optimizer states (angular MuOwn's g / m_g / v_g, (..., 1)) map onto their weight's
ShardedTensor with the last axis collapsed -- compact checkpoints (ANGULAR_MUOWN_COMPACT_CKPT)."""

import torch

from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.dist_checkpointing.optimizer import make_sharded_optimizer_tensor


def test_plain_weight():
    w = ShardedTensor.from_rank_offsets("decoder.w", torch.zeros(32, 64))
    st = make_sharded_optimizer_tensor(w, torch.zeros(32, 1), prefix="optimizer.state.g")
    assert st.key == "optimizer.state.g.decoder.w"
    assert st.local_shape == (32, 1) and st.global_shape == (32, 1) and st.global_offset == (0, 0)


def test_expert_weight_with_prepended_axis_and_row_sharding():
    # expert 3 of 8 (prepended axis 0), rows sharded 2-way (rank 1)
    w = ShardedTensor.from_rank_offsets(
        "experts.w", torch.zeros(16, 64), (0, 3, 8), (1, 1, 2), prepend_axis_num=1
    )
    st = make_sharded_optimizer_tensor(w, torch.zeros(16, 1), prefix="optimizer.state.m_g")
    assert st.global_shape == (8, 32, 1)
    assert st.global_offset == (3, 16, 0)
    assert st.axis_fragmentations == (8, 2, 1)


def test_last_axis_fragmented_falls_back():
    """Column-sharded weight (TP along the last axis): the rows are replicated over the shards, so
    no compact mapping -- the shape check then requires a weight-shaped state."""
    w = ShardedTensor.from_rank_offsets("decoder.w", torch.zeros(32, 64), (1, 1, 2))
    try:
        make_sharded_optimizer_tensor(w, torch.zeros(32, 1), prefix="optimizer.state.g")
    except AssertionError:
        return
    raise AssertionError("expected the weight-shape assertion")


def test_full_shape_state_unchanged():
    w = ShardedTensor.from_rank_offsets("decoder.w", torch.zeros(32, 64))
    st = make_sharded_optimizer_tensor(w, torch.zeros(32, 64), prefix="optimizer.state.m_u")
    assert st.local_shape == (32, 64) and st.global_shape == (32, 64)
