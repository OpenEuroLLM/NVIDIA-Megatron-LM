# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import cast

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import qb_dual_update
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class TestQBDualUpdate:
    """Pure-tensor tests for the quantile-balancing dual update (CPU, no distributed)."""

    @pytest.mark.internal
    @pytest.mark.parametrize("m,n,k", [(64, 8, 2), (40, 8, 1), (12, 4, 1), (64, 27, 5), (100, 12, 3)])
    def test_column_quantile_contract(self, m, n, k):
        """qb_beta_local is the (col_target+1)-th largest score minus alpha per expert."""
        torch.manual_seed(123)
        scores = torch.randn(m, n)
        beta = torch.zeros(n)

        _, beta_local = qb_dual_update(scores, k, beta, update_beta=True)

        alpha = (scores - beta).topk(k + 1, dim=1).values[:, -1:]
        adjusted = scores - alpha
        col_target = m * k // n
        expected = adjusted.sort(dim=0, descending=True).values[col_target]
        torch.testing.assert_close(beta_local, expected)


    @pytest.mark.internal
    @pytest.mark.parametrize("m,n,k", [(4096, 216, 5), (4096, 112, 5), (4096, 64, 4)])
    def test_dual_update_balances_non_divisible_loads(self, m, n, k):
        """n need not divide m * k (216 or 112 experts at 2^j tokens, top-5): routing with the
        bias the update returns gives every expert about m * k / n tokens; the bias from
        imbalanced scores (a per-expert offset) cancels the offset."""
        torch.manual_seed(0)
        scores = torch.randn(m, n) + torch.linspace(-2.0, 2.0, n)
        beta = torch.zeros(n)
        for _ in range(30):
            _, beta = qb_dual_update(scores, k, beta, update_beta=True)
            beta = beta - beta.mean()
        indices, _ = qb_dual_update(scores, k, beta, update_beta=False)
        load = torch.bincount(indices.flatten(), minlength=n).float()
        share = m * k / n
        assert load.sum() == m * k
        # bias-free routing puts ~0 tokens on the lowest-offset experts; balanced: within 15 %
        assert (load - share).abs().max() < 0.15 * share, (load.min(), load.max(), share)


class TestQuantileBalancingRouter:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.num_moe_experts = 8
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=self.num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="quantile_balancing",
            moe_router_score_function="softmax",
            moe_router_topk=2,
            moe_aux_loss_coeff=0,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )
        self.submodules = get_submodules(
            get_gpt_layer_local_submodules(
                num_experts=self.num_moe_experts, moe_grouped_gemm=False
            ).mlp
        )
        assert isinstance(self.submodules, MoESubmodules)
        self.moe_layer = MoELayer(self.transformer_config, self.submodules)
        self.router = cast(Router, self.moe_layer.router)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_non_qb_router_has_no_qb_buffers(self):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=self.num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )
        router = MoELayer(config, self.submodules).router
        assert router.qb_beta is None
        assert router.qb_beta_accum is None
        assert router.qb_beta_count is None

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("moe_router_pre_softmax", [True, False])
    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    def test_qb_router_forward(self, score_function, moe_router_pre_softmax):
        self.router = self.router.cuda()
        self.router.config.moe_router_score_function = score_function
        self.router.score_function = score_function
        self.router.config.moe_router_pre_softmax = moe_router_pre_softmax

        num_tokens = 32 * 2
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size)).cuda().bfloat16()
        with torch.no_grad():
            probs, routing_map = self.router(hidden_states)

        assert probs.shape == (num_tokens, self.num_moe_experts)
        assert routing_map.shape == (num_tokens, self.num_moe_experts)
        # Each token selects exactly topk distinct experts.
        assert routing_map.sum().item() == num_tokens * self.router.topk

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_qb_beta_accumulates_in_training(self):
        self.router = self.router.cuda()
        self.router.train()
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size)).cuda().bfloat16()

        assert self.router.qb_beta_count.item() == 0
        self.router(hidden_states)
        assert self.router.qb_beta_count.item() == 1
        assert self.router.qb_beta_accum.abs().sum().item() > 0
        self.router(hidden_states)
        assert self.router.qb_beta_count.item() == 2

        # No accumulation outside the training path (eval / recompute).
        accum_before = self.router.qb_beta_accum.clone()
        with torch.no_grad():
            self.router(hidden_states)
        assert self.router.qb_beta_count.item() == 2
        torch.testing.assert_close(self.router.qb_beta_accum, accum_before)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_qb_router_rejects_padding_mask(self):
        self.router = self.router.cuda()
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size)).cuda().bfloat16()
        padding_mask = torch.zeros((32, 2), dtype=torch.bool, device=hidden_states.device)
        padding_mask[-2:] = True

        with pytest.raises(AssertionError, match="does not support padding masks"):
            self.router(hidden_states, padding_mask=padding_mask)


class TestQuantileBalancingWithAuxLoss:
    """[quantile_balancing, seq_aux_loss]: QB routes, the listed auxiliary loss is added."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _config(load_balancing_type, coeff, **kw):
        return TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=8,
            use_cpu_initialization=True,
            moe_router_load_balancing_type=load_balancing_type,
            moe_router_score_function="sigmoid",
            moe_router_topk=2,
            moe_aux_loss_coeff=coeff,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
            **kw,
        )

    def _router(self, load_balancing_type, coeff):
        config = self._config(load_balancing_type, coeff)
        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=8, moe_grouped_gemm=False).mlp
        )
        torch.manual_seed(0)
        return cast(Router, MoELayer(config, submodules).router).cuda()

    @pytest.mark.internal
    def test_config_validation(self):
        self._config(["quantile_balancing", "seq_aux_loss"], [0.0, 1e-4])  # valid
        with pytest.raises(ValueError, match="must be 0"):
            self._config(["quantile_balancing", "seq_aux_loss"], [1e-4, 1e-4])
        with pytest.raises(ValueError, match="only with auxiliary losses"):
            self._config(["quantile_balancing", "sinkhorn"], [0.0, 0.0])
        with pytest.raises(ValueError, match="qb_beta"):
            self._config("quantile_balancing", 0.0, moe_router_enable_expert_bias=True)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_qb_routes_and_the_aux_loss_adds_gradient(self):
        hidden = torch.randn((32, 2, 12), device="cuda").bfloat16()
        grads = {}
        for name, lb, coeff in (
            ("qb", "quantile_balancing", 0.0),
            ("qb_seq", ["quantile_balancing", "seq_aux_loss"], [0.0, 1e-1]),
        ):
            router = self._router(lb, coeff)
            router.train()
            assert router.uses_qb and router.qb_beta_count.item() == 0
            probs, routing_map = router(hidden)
            assert router.qb_beta_count.item() == 1  # QB routed and accumulated its quantile
            assert routing_map.sum().item() == 64 * 2
            probs.sum().backward()
            grads[name] = router.weight.grad.float().clone()
        # the same QB routing; the aux loss adds its gradient to the router weight
        assert not torch.allclose(grads["qb"], grads["qb_seq"])
