# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""MoonEP's expert activation honours activation_func_clamp_value (as WeightedSwiGLUFunction)."""

import pytest
import torch

from megatron.core.fusions.fused_bias_swiglu import WeightedSwiGLUFunction

moonep = pytest.importorskip("megatron.core.transformer.moe.moonep")


@pytest.mark.parametrize("limit", [None, 10.0, 0.5])
def test_moonep_activation_matches_weighted_swiglu(limit):
    torch.manual_seed(0)
    h = (torch.randn(64, 2 * 48) * 6).requires_grad_()
    probs = torch.rand(64, 1).requires_grad_()
    ref = WeightedSwiGLUFunction.apply(h, probs, False, limit)
    g = torch.randn_like(ref)
    dh_ref, dp_ref = torch.autograd.grad(ref, (h, probs), g)
    out = moonep._weighted_act(h.detach(), probs.detach(), limit)
    dh, dp = moonep._weighted_act_back(g, h.detach(), probs.detach(), limit)
    torch.testing.assert_close(out, ref.detach())
    torch.testing.assert_close(dh, dh_ref)
    torch.testing.assert_close(dp, dp_ref)
    if limit is not None and limit < 1:
        # the clamp is active somewhere (else this would not test it)
        assert not torch.allclose(out, moonep._weighted_act(h.detach(), probs.detach(), None))
