# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""moe_global_aux_counts_from_forward: under activation recompute the global aux loss reuses the token counts the
checkpointed first forward all-reduced asynchronously -- same outputs, gradients and accumulated counts as the
synchronous all-reduce in the recompute, and the forward's counts are consumed by its recompute."""

from typing import cast

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (NCCL)")


def _router(from_forward: bool):
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    cfg = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_moe_experts=8,
        use_cpu_initialization=True,
        moe_router_load_balancing_type="global_aux_loss",
        moe_router_topk=2,
        moe_aux_loss_coeff=1e-2,
        moe_router_score_function="sigmoid",
        add_bias_linear=False,
        moe_global_aux_counts_from_forward=from_forward,
    )
    submodules = get_submodules(
        get_gpt_layer_local_submodules(num_experts=8, moe_grouped_gemm=False).mlp
    )
    assert isinstance(submodules, MoESubmodules)
    router = cast(Router, MoELayer(cfg, submodules).router).cuda()
    router.train()
    return router


def _run(from_forward: bool):
    router = _router(from_forward)
    torch.manual_seed(7 + torch.distributed.get_rank())
    x = torch.randn(64, 1, 32, device="cuda", requires_grad=True)
    w = torch.randn(64, 8, device="cuda")

    stashed = []

    def fn(h):
        probs, routing_map = router(h)
        if not torch.is_grad_enabled():  # the checkpoint's first forward
            stashed.append(len(router._fwd_global_counts))
        return probs * routing_map

    out = tensor_parallel.checkpoint(fn, False, x)
    (out * w).sum().backward()
    assert stashed == [1 if from_forward else 0], stashed
    assert not router._fwd_global_counts, "the recompute did not consume the forward's counts"
    return (out.detach(), x.grad.detach(), router.weight.grad.detach().clone(),
            router.global_tokens_per_expert.detach().clone())


class TestGlobalCountsFromForward:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_matches_the_synchronous_recompute(self):
        ref = _run(False)
        got = _run(True)
        for name, a, b in zip(("out", "dx", "dweight", "global_counts"), ref, got):
            assert torch.equal(a, b), name
        assert got[3].sum() > 0  # the aux loss ran (counts accumulated)
