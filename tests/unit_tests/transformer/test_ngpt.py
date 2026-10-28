# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Unit tests for the nGPT geometry primitives (CPU-only)."""

import math

import pytest
import torch

from megatron.core.transformer.ngpt import (
    ALPHA_ACTIVATIONS,
    LERP_NORMALIZATION_MODES,
    EigenLRScaler,
    NGPTResidual,
    l2_normalize,
    resolve_scale_factor,
)

HIDDEN = 64
# Modes whose forward output is exactly on the unit sphere. "approx" and
# "approx_fwd" deliberately let the residual norm drift (nGPT's orthogonality
# approximation), so they are checked separately.
EXACT_SPHERE_MODES = ("analytic", "direct", "both", "direct_ste", "both_ste")


def _unit(*shape: int) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.randn(*shape), dim=-1)


class TestEigenLRScaler:
    def test_effective_value_matches_init(self):
        """weight * scaler must equal init_value regardless of base_scale."""
        scaler = EigenLRScaler(
            HIDDEN, init_value=math.sqrt(HIDDEN), base_scale=1 / math.sqrt(HIDDEN)
        )
        assert torch.allclose(
            scaler.effective(), torch.full((HIDDEN,), math.sqrt(HIDDEN)), atol=1e-5
        )

    def test_forward_scales_last_dim(self):
        scaler = EigenLRScaler(HIDDEN, init_value=2.0)
        x = torch.ones(3, HIDDEN)
        assert torch.allclose(scaler(x), torch.full((3, HIDDEN), 2.0), atol=1e-6)

    def test_marked_as_scaler(self):
        """The optimizer/normalizer rely on this attribute to skip the param."""
        assert EigenLRScaler(HIDDEN).weight.is_ngpt_scaler is True

    def test_rejects_nonpositive_base_scale(self):
        with pytest.raises(ValueError, match="base_scale must be positive"):
            EigenLRScaler(HIDDEN, base_scale=0.0)


class TestResolveScaleFactor:
    def test_known_modes(self):
        assert resolve_scale_factor("one", HIDDEN, name="t") == 1.0
        assert resolve_scale_factor("sqrt_dim", HIDDEN, name="t") == math.sqrt(HIDDEN)
        assert resolve_scale_factor("inv_sqrt_dim", HIDDEN, name="t") == 1 / math.sqrt(HIDDEN)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown t="):
            resolve_scale_factor("bogus", HIDDEN, name="t")


class TestL2Normalize:
    def test_unit_norm_and_dtype_preserved(self):
        x = torch.randn(2, 3, HIDDEN, dtype=torch.bfloat16)
        out = l2_normalize(x)
        assert out.dtype == torch.bfloat16
        assert torch.allclose(out.float().norm(dim=-1), torch.ones(2, 3), atol=1e-2)


class TestNGPTResidual:
    @pytest.mark.parametrize("mode", EXACT_SPHERE_MODES)
    def test_output_is_on_unit_sphere(self, mode):
        """With unit-norm inputs these modes must land exactly on the sphere."""
        torch.manual_seed(0)
        residual = NGPTResidual(HIDDEN, lerp_normalization=mode)
        out = residual(torch.randn(4, 2, HIDDEN), _unit(4, 2, HIDDEN))
        assert torch.allclose(out.norm(dim=-1), torch.ones(4, 2), atol=1e-5)

    @pytest.mark.parametrize("mode", ("approx", "approx_fwd"))
    def test_approx_modes_are_close_to_sphere(self, mode):
        """The orthogonality approximation stays near, but not exactly on, the sphere."""
        torch.manual_seed(0)
        residual = NGPTResidual(HIDDEN, lerp_normalization=mode)
        out = residual(torch.randn(4, 2, HIDDEN), _unit(4, 2, HIDDEN))
        assert torch.allclose(out.norm(dim=-1), torch.ones(4, 2), atol=5e-2)

    @pytest.mark.parametrize("activation", ALPHA_ACTIVATIONS)
    def test_effective_alpha_at_init_matches_target(self, activation):
        """Init solves the pre-activation so alpha_eff(0) == alpha_init_value."""
        target = 0.05
        residual = NGPTResidual(HIDDEN, alpha_init_value=target, alpha_activation=activation)
        alpha_eff = residual._alpha_activation_fn(residual.alpha.float() * residual.alpha_scaler)
        assert torch.allclose(alpha_eff, torch.full((HIDDEN,), target), atol=1e-5)

    def test_zero_alpha_is_identity_on_residual(self):
        """alpha == 0 must leave a unit-norm residual untouched."""
        residual = NGPTResidual(HIDDEN, alpha_init_value=1e-12, lerp_normalization="analytic")
        torch.nn.init.zeros_(residual.alpha)
        x = _unit(2, 3, HIDDEN)
        assert torch.allclose(residual(torch.randn(2, 3, HIDDEN), x), x, atol=1e-5)

    def test_alpha_one_selects_sublayer_output(self):
        """alpha == 1 must return the normalized sublayer output."""
        residual = NGPTResidual(HIDDEN, lerp_normalization="analytic")
        with torch.no_grad():
            residual.alpha.copy_(torch.ones(HIDDEN) / residual.alpha_scaler)
        a = torch.randn(2, 3, HIDDEN)
        out = residual(a, _unit(2, 3, HIDDEN))
        assert torch.allclose(out, l2_normalize(a), atol=1e-4)

    @pytest.mark.parametrize("mode", LERP_NORMALIZATION_MODES)
    def test_gradients_flow_to_alpha_and_input(self, mode):
        residual = NGPTResidual(HIDDEN, lerp_normalization=mode)
        a = torch.randn(2, 2, HIDDEN, requires_grad=True)
        residual(a, _unit(2, 2, HIDDEN)).sum().backward()
        assert residual.alpha.grad is not None and residual.alpha.grad.abs().sum() > 0
        assert a.grad is not None and a.grad.abs().sum() > 0

    def test_analytic_factor_matches_closed_form(self):
        """The analytic mode must reproduce 1/sqrt((1-a)^2 + 2a(1-a)cos + a^2)."""
        torch.manual_seed(0)
        residual = NGPTResidual(HIDDEN, lerp_normalization="analytic", post_layer_norm=True)
        x, a_raw = _unit(1, 1, HIDDEN), torch.randn(1, 1, HIDDEN)
        out = residual(a_raw, x)

        alpha = residual._alpha_activation_fn(residual.alpha.float() * residual.alpha_scaler)
        a = l2_normalize(a_raw).float()
        cos = (x * a).sum(dim=-1, keepdim=True)
        denom = (1 - alpha) ** 2 + 2 * alpha * (1 - alpha) * cos + alpha**2
        expected = (x + alpha * (a - x)) * torch.rsqrt(denom)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_post_layer_norm_disabled_skips_normalization(self):
        """Without post_layer_norm the raw (unnormalized) output is interpolated."""
        residual = NGPTResidual(HIDDEN, lerp_normalization="direct", post_layer_norm=False)
        with torch.no_grad():
            residual.alpha.copy_(torch.ones(HIDDEN) / residual.alpha_scaler)
        a = torch.randn(1, 1, HIDDEN) * 5.0
        assert torch.allclose(residual(a, _unit(1, 1, HIDDEN)), l2_normalize(a), atol=1e-4)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="lerp_normalization must be one of"):
            NGPTResidual(HIDDEN, lerp_normalization="nonsense")

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="alpha_activation must be one of"):
            NGPTResidual(HIDDEN, alpha_activation="nonsense")

    @pytest.mark.parametrize(
        "activation,bad_alpha",
        [("half_sigmoid", 0.7), ("half_tanh_sq", 0.9), ("tanh_smooth_bound", 0.99)],
    )
    def test_out_of_range_alpha_init_raises(self, activation, bad_alpha):
        with pytest.raises(ValueError):
            NGPTResidual(HIDDEN, alpha_init_value=bad_alpha, alpha_activation=activation)

    def test_base_scale_minus_one_uses_inv_sqrt_hidden(self):
        residual = NGPTResidual(HIDDEN, base_scale=-1.0)
        assert residual.base_scale == pytest.approx(1 / math.sqrt(HIDDEN))

    def test_dtype_preserved(self):
        residual = NGPTResidual(HIDDEN, lerp_normalization="analytic")
        x = _unit(2, 2, HIDDEN).bfloat16()
        assert residual(torch.randn(2, 2, HIDDEN).bfloat16(), x).dtype == torch.bfloat16


@pytest.mark.parametrize("mode", ["approx", "analytic", "direct"])
def test_post_norm_floor_keeps_short_outputs_and_bounds_their_gradient(mode):
    """post_norm_floor: a / max(|a|, floor) -- an output above the floor is normalized as before, a
    shorter one is kept (a sublayer can switch off), and its gradient does not grow as 1 / |a|."""
    from megatron.core.transformer.ngpt import NGPTResidual

    torch.manual_seed(0)
    h = 64
    x = torch.nn.functional.normalize(torch.randn(4, 2, h), dim=-1)
    plain = NGPTResidual(h, alpha_init_value=0.3, lerp_normalization=mode)
    floored = NGPTResidual(h, alpha_init_value=0.3, lerp_normalization=mode, post_norm_floor=1.0)
    big = torch.nn.functional.normalize(torch.randn(4, 2, h), dim=-1) * 3.0
    torch.testing.assert_close(floored(big, x), plain(big, x))
    grads, w = [], torch.randn(4, 2, h)
    for res in (plain, floored):
        torch.manual_seed(1)
        tiny = (torch.randn(4, 2, h) * 1e-4).requires_grad_(True)
        (res(tiny, x) * w).sum().backward()  # a linear readout (|out| is fixed for exact modes)
        grads.append(tiny.grad.norm())
    assert grads[0] > 1e3 * grads[1], grads  # plain: ~1 / |a|; floored: bounded
    with pytest.raises(ValueError, match="post_norm_floor"):
        NGPTResidual(h, post_norm_floor=-1.0)


def test_fixed_alpha_is_not_trained_and_keeps_its_init():
    """alpha_trainable=False: alpha stays at alpha_init_value (no gradient, skipped by DDP and the
    optimizer), the update still takes the LERP step of that size."""
    from megatron.core.transformer.ngpt import NGPTResidual

    torch.manual_seed(0)
    h = 64
    x = torch.nn.functional.normalize(torch.randn(4, 2, h), dim=-1)
    a = torch.randn(4, 2, h, requires_grad=True)
    learned = NGPTResidual(h, alpha_init_value=0.05, alpha_activation="half_sigmoid")
    fixed = NGPTResidual(h, alpha_init_value=0.05, alpha_activation="half_sigmoid",
                         alpha_trainable=False)
    assert learned.alpha.requires_grad and not fixed.alpha.requires_grad
    torch.testing.assert_close(fixed(a, x), learned(a, x))
    fixed(a, x).sum().backward()
    assert fixed.alpha.grad is None and a.grad is not None
