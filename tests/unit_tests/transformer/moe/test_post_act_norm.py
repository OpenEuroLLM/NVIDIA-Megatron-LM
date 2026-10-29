# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Post-activation RMS norm of the experts / MLPs (moe_expert_post_act_norm, mlp_post_act_norm):
a = p * h / rms(h) with h = SwiGLU(fc1 x), no scale vector -- the fused kernel against autograd,
MoonEP's activation, and the MoE layer (TEGroupedMLP / SequentialMLP, shared expert)."""

import copy

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions.fused_bias_swiglu import NormedSwiGLUFunction, normed_swiglu_impl
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

EPS = 1e-5


def _reference(h, probs, limit, floor=0.0):
    y1, y2 = torch.chunk(h.float(), 2, -1)
    if limit is not None:
        y1, y2 = y1.clamp(max=limit), y2.clamp(min=-limit, max=limit)
    a = F.silu(y1) * y2
    a = a * torch.rsqrt((a.square().mean(-1, keepdim=True) + EPS).clamp(min=floor * floor))
    return a if probs is None else a * probs


@pytest.mark.parametrize("limit", [None, 10.0, 0.7])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("floor", [0.0, 1.0])
def test_fused_normed_swiglu_matches_autograd(limit, weighted, floor):
    torch.manual_seed(0)
    # rows of very different scale: with floor 1 some are held at the floor, some pass through
    h = (torch.randn(64, 2 * 48, dtype=torch.float64) * torch.logspace(-1, 1, 64)[:, None])
    h.requires_grad_()
    probs = torch.rand(64, 1, dtype=torch.float64).requires_grad_() if weighted else None
    ref = _reference(h, probs, limit, floor)
    g = torch.randn_like(ref)
    inputs = (h, probs) if weighted else (h,)
    grads_ref = torch.autograd.grad(ref, inputs, g)
    out = normed_swiglu_impl(h, probs, False, limit, EPS, floor)
    grads = torch.autograd.grad(out, inputs, g)
    # the kernel reduces in fp32 (fp64 in, fp64 out): fp32 tolerances
    torch.testing.assert_close(out, ref.to(out.dtype), atol=1e-6, rtol=1e-5)
    for a, b in zip(grads, grads_ref):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_output_has_unit_rms_times_the_weight_and_bf16_runs():
    torch.manual_seed(1)
    h = torch.randn(32, 2 * 64, device="cuda", dtype=torch.bfloat16) * 50
    probs = torch.rand(32, 1, device="cuda")
    out = normed_swiglu_impl(h, probs, False, None, EPS)
    assert out.dtype == torch.bfloat16
    rms = (out.float() / probs).square().mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=2e-2, rtol=0)


def test_bounded_norm_passes_small_rows_and_caps_large_ones():
    torch.manual_seed(4)
    small = torch.randn(8, 2 * 64) * 0.3   # rms(SwiGLU) well below 1
    large = torch.randn(8, 2 * 64) * 30.0
    for h, passes in ((small, True), (large, False)):
        out = normed_swiglu_impl(h, None, False, None, EPS, 1.0)
        y1, y2 = torch.chunk(h, 2, -1)
        act = F.silu(y1) * y2
        if passes:
            torch.testing.assert_close(out, act)
        else:
            rms = out.square().mean(-1).sqrt()
            torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-4, rtol=0)


@pytest.mark.parametrize("limit", [None, 10.0, 0.7])
def test_moonep_activation_matches_the_fused_function(limit):
    moonep = pytest.importorskip("megatron.core.transformer.moe.moonep")
    torch.manual_seed(2)
    h = (torch.randn(64, 2 * 48) * 4).requires_grad_()
    probs = torch.rand(64, 1).requires_grad_()
    lim = float("inf") if limit is None else limit
    ref = NormedSwiGLUFunction.apply(h, probs, False, lim, EPS)
    g = torch.randn_like(ref)
    dh_ref, dp_ref = torch.autograd.grad(ref, (h, probs), g)
    out = moonep._weighted_act(h.detach(), probs.detach(), limit, EPS)
    dh, dp = moonep._weighted_act_back(g, h.detach(), probs.detach(), limit, EPS)
    torch.testing.assert_close(out, ref.detach())
    torch.testing.assert_close(dh, dh_ref)
    torch.testing.assert_close(dp, dp_ref)


class TestMoELayerPostActNorm:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _layer(grouped, fused, shared, **kw):
        cfg = TransformerConfig(
            num_layers=1, hidden_size=32, num_attention_heads=4, num_moe_experts=4,
            moe_router_topk=2, moe_ffn_hidden_size=48, ffn_hidden_size=48,
            moe_shared_expert_intermediate_size=48 if shared else None,
            activation_func=F.silu, gated_linear_unit=True, add_bias_linear=False,
            bias_activation_fusion=fused, moe_grouped_gemm=grouped,
            moe_token_dispatcher_type="alltoall", moe_router_load_balancing_type="none",
            # a tiny eps: this model's activations are ~3e-3 (hidden 32, init std 0.02), so
            # mean(h^2) ~ 1e-5 would be of the order of the usual eps and break the invariance
            use_cpu_initialization=True, layernorm_epsilon=1e-12,
            moe_expert_post_act_norm=True, mlp_post_act_norm=shared, **kw,
        )
        spec = (
            get_gpt_layer_with_transformer_engine_submodules(num_experts=4, moe_grouped_gemm=True)
            if grouped
            else get_gpt_layer_local_submodules(num_experts=4, moe_grouped_gemm=False)
        )
        return MoELayer(cfg, get_submodules(spec.mlp)).cuda()

    @pytest.mark.parametrize("grouped", [True, False])
    @pytest.mark.parametrize("shared", [False, True])
    def test_fused_and_unfused_agree_and_the_scale_of_fc1_drops_out(self, grouped, shared):
        torch.manual_seed(3)
        fused = self._layer(grouped, True, shared)
        unfused = self._layer(grouped, False, shared)
        unfused.load_state_dict(copy.deepcopy(fused.state_dict()))
        x = torch.randn(8, 2, 32, device="cuda", requires_grad=True)
        outs, grads = [], []
        for layer in (fused, unfused):
            xi = x.detach().clone().requires_grad_()
            out, _ = layer(xi)
            out.sum().backward()
            outs.append(out.detach())
            grads.append(xi.grad)
        torch.testing.assert_close(outs[0], outs[1], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(grads[0], grads[1], atol=1e-5, rtol=1e-4)
        # scaling every fc1's linear (up) projection scales h; the norm removes it (fc1 is one
        # [gate; up] matrix, or split into weight*_w (gate) / weight*_v (up) for MuOwn)
        with torch.no_grad():
            n_up = 0
            for name, p in fused.named_parameters():
                if "linear_fc1" not in name or p.dim() != 2 or name.endswith("_w"):
                    continue
                (p if name.endswith("_v") else p[p.shape[0] // 2:]).mul_(3.0)
                n_up += 1
            assert n_up > 0
            scaled, _ = fused(x.detach())
        torch.testing.assert_close(scaled, outs[0], atol=1e-5, rtol=1e-4)

    def test_rejects_te_activation_modules(self):
        with pytest.raises(ValueError, match="post-activation"):
            TransformerConfig(
                num_layers=1, hidden_size=32, num_attention_heads=4,
                moe_expert_post_act_norm=True, use_te_activation_func=True,
            )
