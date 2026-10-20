# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Per-row optimizer states (angular MuOwn's g / m_g / v_g, (..., 1)) map onto their weight's
ShardedTensor with the last axis collapsed -- compact checkpoints (ANGULAR_MUOWN_COMPACT_CKPT)."""

import torch

from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.dist_checkpointing.optimizer import make_sharded_optimizer_tensor
from megatron.core.ssm.utils import _split_tensor_factory


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


def test_factory_weight_gets_expanded_row_state():
    """The split GDN / Mamba in_proj is a ShardedTensorFactory: a compact (rows, 1) state is
    expanded to the weight shape so the factory's narrow + metadata checks hold (qwen35 s2 save
    crashed with 'Data shape should match local_shape')."""
    w = ShardedTensor.from_rank_offsets("decoder.in_proj.weight", torch.zeros(48, 64))
    fac = _split_tensor_factory(w, [16, 32], ["a", "b"], 0)
    g = torch.arange(48.0).reshape(48, 1)
    opt = make_sharded_optimizer_tensor(fac, g, prefix="optimizer.state.g")
    assert opt.data.shape == (48, 64) and torch.equal(opt.data[:, 63:], g)
    chunks = opt.build()
    assert [c.local_shape for c in chunks] == [(16, 64), (32, 64)]
    assert torch.equal(chunks[1].data[:, :1], g[16:])
