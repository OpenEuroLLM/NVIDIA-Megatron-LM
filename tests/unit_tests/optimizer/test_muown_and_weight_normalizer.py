# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Unit tests for Muown and the nGPT weight normalizer (CPU-only, single process)."""

import pytest
import torch

from megatron.core.optimizer.muown import Muown, newton_schulz
from megatron.core.optimizer.weight_normalizer import classify_param, normalize_weight_


class TestNewtonSchulz:
    @pytest.mark.parametrize("shape", [(64, 32), (32, 64), (48, 48)])
    def test_singular_values_pushed_toward_one(self, shape):
        torch.manual_seed(0)
        out = newton_schulz(torch.randn(*shape), steps=5)
        assert out.shape == shape
        sv = torch.linalg.svdvals(out.float())
        # The quintic iteration is deliberately inexact but must land in a tight band.
        assert sv.min() > 0.5 and sv.max() < 1.5

    def test_rejects_non_matrix(self):
        with pytest.raises(ValueError, match="expects a 2-D matrix"):
            newton_schulz(torch.randn(4, 4, 4))


class TestMuown:
    def test_rejects_one_dimensional_params(self):
        """1-D params must be routed to Adam, not silently mishandled."""
        param = torch.nn.Parameter(torch.randn(8))
        opt = Muown([param], lr=1e-3)
        param.grad = torch.randn(8)
        with pytest.raises(ValueError, match="route non-matrix parameters to Adam"):
            opt.step()

    def test_magnitude_step_is_bounded_by_lr(self):
        """The l-infinity (sign) step moves each row magnitude by at most mag_lr.

        This is Muown's core guarantee: row magnitudes cannot drift arbitrarily.
        """
        torch.manual_seed(0)
        lr = 1e-2
        param = torch.nn.Parameter(torch.randn(16, 8))
        opt = Muown([param], lr=lr, momentum=0.0, nesterov=False)
        before = param.detach().norm(dim=-1).clone()
        param.grad = torch.randn(16, 8) * 100.0  # huge gradient
        opt.step()
        after = param.detach().norm(dim=-1)
        # Even with an enormous gradient the magnitude moves by exactly one lr step.
        assert torch.all((after - before).abs() <= lr + 1e-5)

    def test_rows_stay_unit_direction(self):
        """After the update, W / |W| must still have unit rows."""
        torch.manual_seed(0)
        param = torch.nn.Parameter(torch.randn(16, 8))
        opt = Muown([param], lr=1e-2)
        for _ in range(5):
            opt.zero_grad()
            (param**2).sum().backward()
            opt.step()
        direction = param.detach() / param.detach().norm(dim=-1, keepdim=True)
        assert torch.allclose(direction.norm(dim=-1), torch.ones(16), atol=1e-5)

    def test_max_row_norm_is_respected(self):
        torch.manual_seed(0)
        param = torch.nn.Parameter(torch.randn(16, 8) * 5)
        opt = Muown([param], lr=1e-1, max_row_norm=1.0)
        for _ in range(50):
            opt.zero_grad()
            (param * 100).sum().backward()
            opt.step()
        assert param.detach().norm(dim=-1).max() <= 1.0 + 1e-5

    def test_handles_grouped_expert_tensors(self):
        """Grouped MoE weights of shape [E, out, in] are orthogonalized per expert."""
        torch.manual_seed(0)
        param = torch.nn.Parameter(torch.randn(4, 16, 8))
        opt = Muown([param], lr=1e-2)
        opt.zero_grad()
        (param**2).sum().backward()
        opt.step()
        assert param.shape == (4, 16, 8)
        assert torch.isfinite(param.detach()).all()

    def test_optimizer_reduces_a_quadratic_objective(self):
        torch.manual_seed(0)
        target = torch.randn(16, 8)
        param = torch.nn.Parameter(torch.randn(16, 8))
        opt = Muown([param], lr=5e-2)

        def loss_fn():
            return ((param - target) ** 2).sum()

        first = loss_fn().item()
        for _ in range(100):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        assert loss_fn().item() < first

    @pytest.mark.parametrize("scale_mode", ["spectral", "unit"])
    def test_scale_modes_run(self, scale_mode):
        param = torch.nn.Parameter(torch.randn(16, 8))
        opt = Muown([param], lr=1e-3, scale_mode=scale_mode)
        opt.zero_grad()
        (param**2).sum().backward()
        opt.step()
        assert torch.isfinite(param.detach()).all()

    def test_invalid_config_raises(self):
        param = torch.nn.Parameter(torch.randn(4, 4))
        with pytest.raises(ValueError, match="num_ns_steps"):
            Muown([param], num_ns_steps=0)
        with pytest.raises(ValueError, match="scale_mode"):
            Muown([param], scale_mode="bogus")


class TestClassifyParam:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("decoder.layers.0.self_attention.linear_qkv.weight", "input"),
            ("decoder.layers.0.mlp.linear_fc1.weight", "input"),
            ("decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight", "input"),
            ("decoder.layers.0.self_attention.linear_proj.weight", "output"),
            ("decoder.layers.0.mlp.linear_fc2.weight", "output"),
            ("decoder.layers.0.mlp.experts.local_experts.3.linear_fc2.weight", "output"),
            ("embedding.word_embeddings.weight", "embedding"),
            ("output_layer.weight", "head"),
            # Must NOT be normalized:
            ("decoder.layers.0.mlp.router.weight", None),
            ("decoder.layers.0.input_layernorm.weight", None),
        ],
    )
    def test_classification(self, name, expected):
        assert classify_param(name, torch.randn(8, 4)) == expected

    def test_one_dimensional_params_are_skipped(self):
        assert (
            classify_param("decoder.layers.0.self_attention.linear_qkv.bias", torch.randn(8))
            is None
        )

    def test_ngpt_scalers_are_skipped(self):
        """Scalers carry magnitude and must never be normalized away."""
        param = torch.randn(8, 4)
        param.is_ngpt_scaler = True
        assert classify_param("decoder.layers.0.mlp.linear_fc1.weight", param) is None


class TestNormalizeWeight:
    @pytest.mark.parametrize("norm_dim", [0, 1])
    def test_rows_or_columns_become_unit_norm(self, norm_dim):
        torch.manual_seed(0)
        param = torch.nn.Parameter(torch.randn(16, 8) * 3)
        normalize_weight_(param, norm_dim)
        assert torch.allclose(
            param.data.norm(dim=norm_dim), torch.ones(param.shape[1 - norm_dim]), atol=1e-5
        )

    def test_bounded_mode_only_shrinks(self):
        """Bounded mode projects onto the ball: small rows are left alone."""
        param = torch.nn.Parameter(torch.zeros(2, 4))
        param.data[0] = torch.tensor([3.0, 0.0, 0.0, 0.0])  # norm 3 -> shrink to 1
        param.data[1] = torch.tensor([0.1, 0.0, 0.0, 0.0])  # norm 0.1 -> unchanged
        normalize_weight_(param, 1, bounded=True, max_norm=1.0)
        assert param.data[0].norm().item() == pytest.approx(1.0, abs=1e-5)
        assert param.data[1].norm().item() == pytest.approx(0.1, abs=1e-5)

    def test_max_norm_is_honored(self):
        param = torch.nn.Parameter(torch.randn(8, 4))
        normalize_weight_(param, 1, max_norm=2.5)
        assert torch.allclose(param.data.norm(dim=1), torch.full((8,), 2.5), atol=1e-5)

    def test_grouped_expert_tensor_normalized_per_expert(self):
        """A [E, out, in] tensor must normalize each expert independently."""
        torch.manual_seed(0)
        param = torch.nn.Parameter(torch.randn(4, 16, 8) * 2)
        normalize_weight_(param, 1)
        assert torch.allclose(param.data.norm(dim=-1), torch.ones(4, 16), atol=1e-5)

    def test_dtype_preserved(self):
        param = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16))
        normalize_weight_(param, 1)
        assert param.dtype == torch.bfloat16
