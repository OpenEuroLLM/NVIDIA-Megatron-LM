# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Muown under tensor parallelism reproduces the single-process update."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS
from megatron.core.optimizer.muown import Muown
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(Utils.world_size < 2, reason="needs 2 GPUs")
@pytest.mark.skipif(not HAVE_EMERGING_OPTIMIZERS, reason="needs emerging_optimizers")
@pytest.mark.parametrize("partition_dim", [0, 1])
def test_sharded_step_matches_full_matrix(partition_dim):
    from megatron.core.optimizer.emerging_optimizers import _muown_config_to_kwargs

    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        tp = parallel_state.get_tensor_model_parallel_group()
        rank, size = tp.rank(), tp.size()
        gen = torch.Generator(device="cuda").manual_seed(0)
        full_w = torch.randn(64, 96, device="cuda", generator=gen)
        full_g = torch.randn(64, 96, device="cuda", generator=gen)

        from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp

        # The reference uses the same Newton-Schulz as the TP path, on the full matrix.
        def full_ns(g, steps=5, **_):
            return newton_schulz_tp(
                g,
                steps=steps,
                coefficient_type="quintic",
                tp_group=None,
                partition_dim=None,
                tp_mode="duplicated",
            )

        ref = torch.nn.Parameter(full_w.clone())
        ref.grad = full_g.clone()
        Muown([ref], lr=0.02, weight_decay=0.1, orthogonalize_fn=full_ns).step()

        shard = torch.nn.Parameter(full_w.chunk(size, dim=partition_dim)[rank].clone())
        shard.grad = full_g.chunk(size, dim=partition_dim)[rank].clone()
        shard.tensor_model_parallel = True
        shard.partition_dim = partition_dim
        config = OptimizerConfig(optimizer="muown", lr=0.02, weight_decay=0.1)
        kwargs = _muown_config_to_kwargs(config, None, ProcessGroupCollection(tp=tp, expt_tp=tp))
        kwargs.update(lr=0.02, weight_decay=0.1)
        Muown([shard], **kwargs).step()

        expected = ref.data.chunk(size, dim=partition_dim)[rank]
        # Newton-Schulz runs in bf16; the sharded and full paths differ in its rounding.
        torch.testing.assert_close(shard.data, expected, atol=2e-3, rtol=2e-2)
    finally:
        Utils.destroy_model_parallel()
