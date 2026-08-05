# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Unit tests for mHC and Block Attention Residuals (CPU-only)."""

import pytest
import torch

from megatron.core.transformer.attn_res import (
    BlockAttnRes,
    accumulate_block,
    resolve_layers_per_block,
)
from megatron.core.transformer.depth_residual import DepthResidualState
from megatron.core.transformer.mhc import (
    MHCMixer,
    collapse_streams,
    expand_to_streams,
    sinkhorn_knopp,
)
from megatron.core.transformer.ngpt import NGPTResidual

S, B, C, N = 4, 2, 32, 4


class TestSinkhornKnopp:
    def test_result_is_doubly_stochastic(self):
        torch.manual_seed(0)
        m = sinkhorn_knopp(torch.randn(3, 5, 5))
        assert torch.allclose(m.sum(-1), torch.ones(3, 5), atol=1e-5)
        assert torch.allclose(m.sum(-2), torch.ones(3, 5), atol=1e-5)
        assert (m >= 0).all()

    def test_is_non_expansive(self):
        """Doubly stochastic => spectral norm <= 1, the property mHC relies on."""
        torch.manual_seed(0)
        m = sinkhorn_knopp(torch.randn(8, 8))
        assert torch.linalg.matrix_norm(m, 2).item() <= 1.0 + 1e-4

    def test_composition_stays_norm_preserving(self):
        """The key mHC claim: products stay on the manifold at any depth.

        Unconstrained Hyper-Connections blow up to ~3000x over a 27B model's depth.
        """
        torch.manual_seed(0)
        prod = torch.eye(4)
        for _ in range(64):
            prod = prod @ sinkhorn_knopp(torch.randn(4, 4))
        assert torch.allclose(prod.sum(-1), torch.ones(4), atol=1e-3)
        assert torch.linalg.matrix_norm(prod, 2).item() <= 1.0 + 1e-3

    def test_extreme_logits_do_not_produce_nan(self):
        m = sinkhorn_knopp(torch.tensor([[1e4, -1e4], [-1e4, 1e4]]))
        assert torch.isfinite(m).all()


class TestMHCMixer:
    def test_init_gives_uniform_read_and_unit_write(self):
        """At init H_pre reads uniformly (1/n) and H_post writes with weight 1."""
        mixer = MHCMixer(C, expansion_rate=N)
        streams = expand_to_streams(torch.randn(S, B, C), N)
        h_pre, h_post, _ = mixer(streams)
        assert torch.allclose(h_pre, torch.full_like(h_pre, 1.0 / N), atol=1e-4)
        assert torch.allclose(h_post, torch.ones_like(h_post), atol=1e-4)

    def test_identity_init_keeps_streams_independent(self):
        mixer = MHCMixer(C, expansion_rate=N, res_init="identity")
        _, _, h_res = mixer(expand_to_streams(torch.randn(S, B, C), N))
        assert h_res.diagonal(dim1=-2, dim2=-1).mean().item() > 0.95

    def test_uniform_init_mixes_evenly(self):
        mixer = MHCMixer(C, expansion_rate=N, res_init="uniform")
        _, _, h_res = mixer(expand_to_streams(torch.randn(S, B, C), N))
        assert torch.allclose(h_res, torch.full_like(h_res, 1.0 / N), atol=1e-4)

    def test_mixer_output_is_doubly_stochastic_after_perturbation(self):
        """Rows are exact; columns converge asymptotically (the sweep ends on rows).

        Strongly perturbed logits give near-permutation matrices, the slowest case
        for Sinkhorn, so the column residual is checked at a looser tolerance.
        """
        torch.manual_seed(0)
        mixer = MHCMixer(C, expansion_rate=N)
        with torch.no_grad():
            mixer.phi.normal_(std=0.5)
            mixer.alpha_res.fill_(1.0)
        _, _, h_res = mixer(expand_to_streams(torch.randn(S, B, C), N))
        assert torch.allclose(h_res.sum(-1), torch.ones(S, B, N), atol=1e-4)
        assert torch.allclose(h_res.sum(-2), torch.ones(S, B, N), atol=5e-2)
        assert (h_res >= 0).all()

    def test_more_sinkhorn_iters_tighten_double_stochasticity(self):
        """The iteration count is a real knob on the column-sum residual."""
        torch.manual_seed(0)
        logits = torch.randn(6, 6) * 4.0
        err = [
            (sinkhorn_knopp(logits, num_iters=k).sum(-2) - 1).abs().max().item()
            for k in (2, 20, 200)
        ]
        assert err[0] > err[1] > err[2]

    def test_read_and_write_shapes(self):
        mixer = MHCMixer(C, expansion_rate=N)
        streams = expand_to_streams(torch.randn(S, B, C), N)
        h_pre, h_post, h_res = mixer(streams)
        assert mixer.read(streams, h_pre).shape == (S, B, C)
        out = mixer.write(streams, torch.randn(S, B, C), h_post, h_res)
        assert out.shape == (S, B, N, C)
        assert collapse_streams(out).shape == (S, B, C)

    def test_depth_does_not_explode(self):
        """Stacking many mHC updates must not blow the stream magnitude up."""
        torch.manual_seed(0)
        mixer = MHCMixer(C, expansion_rate=N)
        streams = expand_to_streams(torch.randn(S, B, C), N)
        start = streams.norm(dim=-1).mean().item()
        for _ in range(64):
            _, h_post, h_res = mixer(streams)
            streams = mixer.write(streams, torch.randn(S, B, C) * 0.1, h_post, h_res)
        grown = streams.norm(dim=-1).mean().item()
        assert torch.isfinite(streams).all()
        assert grown < 5 * start  # bounded, not the O(3000x) blow-up of plain HC

    def test_write_with_lerp_stays_on_sphere(self):
        """With the nGPT LERP composed in, streams stay unit-norm."""
        mixer = MHCMixer(C, expansion_rate=N)
        lerp = NGPTResidual(C, lerp_normalization="analytic")
        streams = torch.nn.functional.normalize(torch.randn(S, B, N, C), dim=-1)
        _, h_post, h_res = mixer(streams)
        out = mixer.write(streams, torch.randn(S, B, C), h_post, h_res, lerp=lerp)
        assert torch.allclose(out.norm(dim=-1), torch.ones(S, B, N), atol=1e-4)

    def test_gradients_flow(self):
        mixer = MHCMixer(C, expansion_rate=N)
        streams = expand_to_streams(torch.randn(S, B, C), N)
        h_pre, h_post, h_res = mixer(streams)
        mixer.write(streams, torch.randn(S, B, C), h_post, h_res).sum().backward()
        assert mixer.phi.grad is not None and torch.isfinite(mixer.phi.grad).all()
        assert mixer.b_res.grad is not None

    def test_expansion_rate_one_is_a_plain_residual(self):
        """n=1 must reduce the doubly stochastic mixer to the scalar 1."""
        mixer = MHCMixer(C, expansion_rate=1)
        streams = expand_to_streams(torch.randn(S, B, C), 1)
        _, _, h_res = mixer(streams)
        assert torch.allclose(h_res, torch.ones_like(h_res), atol=1e-5)

    def test_scaler_params_are_marked(self):
        """Connectivity params must be excluded from the nGPT weight normalizer."""
        mixer = MHCMixer(C, expansion_rate=N)
        for name in ("b_pre", "b_post", "b_res", "alpha_pre", "alpha_post", "alpha_res"):
            assert getattr(getattr(mixer, name), "is_ngpt_scaler", False)

    def test_rejects_bad_config(self):
        with pytest.raises(ValueError, match="expansion_rate"):
            MHCMixer(C, expansion_rate=0)
        with pytest.raises(ValueError, match="res_init"):
            MHCMixer(C, res_init="bogus")


class TestBlockAttnRes:
    def test_uniform_attention_at_init(self):
        """A zero pseudo-query gives uniform depth weights: the standard residual."""
        module = BlockAttnRes(C)
        sources = [torch.randn(S, B, C) for _ in range(4)]
        out = module(sources)
        expected = torch.stack(sources).float().mean(0)
        assert torch.allclose(out, expected, atol=1e-4)

    def test_weights_are_convex(self):
        torch.manual_seed(0)
        module = BlockAttnRes(C, query_init_std=1.0)
        sources = [torch.randn(S, B, C) for _ in range(5)]
        v = torch.stack(sources).float()
        k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + module.eps)
        w = torch.softmax(torch.einsum("d,ksbd->ksb", module.query.float(), k), dim=0)
        assert torch.allclose(w.sum(0), torch.ones(S, B), atol=1e-5)
        assert (w >= 0).all()

    def test_magnitude_stays_bounded(self):
        """Convex weights bound the output by the largest source magnitude."""
        torch.manual_seed(0)
        module = BlockAttnRes(C, query_init_std=1.0)
        sources = [torch.randn(S, B, C) * (i + 1) for i in range(6)]
        out = module(sources)
        assert out.norm(dim=-1).max() <= max(s.norm(dim=-1).max() for s in sources) + 1e-3

    def test_key_normalization_prevents_scale_domination(self):
        """A hugely-scaled source must not dominate purely because of its norm."""
        torch.manual_seed(0)
        module = BlockAttnRes(C, query_init_std=1.0)
        a, b = torch.randn(S, B, C), torch.randn(S, B, C)
        w_small = _depth_weights(module, [a, b])
        w_big = _depth_weights(module, [a, b * 1000.0])
        assert torch.allclose(w_small, w_big, atol=1e-3)

    def test_gradients_flow(self):
        module = BlockAttnRes(C, query_init_std=1.0)
        module([torch.randn(S, B, C) for _ in range(3)]).sum().backward()
        assert module.query.grad is not None and torch.isfinite(module.query.grad).all()

    def test_resolve_layers_per_block(self):
        assert resolve_layers_per_block(32, 8) == 4
        assert resolve_layers_per_block(30, 8) == 3  # remainder joins the last block
        assert resolve_layers_per_block(4, 8) == 1  # never zero
        with pytest.raises(ValueError):
            resolve_layers_per_block(32, 0)


def _depth_weights(module: BlockAttnRes, sources: list[torch.Tensor]) -> torch.Tensor:
    v = torch.stack(sources).float()
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + module.eps)
    k = k * module.key_norm_weight.float()
    return torch.softmax(torch.einsum("d,ksbd->ksb", module.query.float(), k), dim=0)


class TestDepthResidualState:
    def test_mhc_streams_travel_as_hidden_states(self):
        """mHC needs no side state: expand/collapse bracket the layer loop."""
        h = torch.randn(S, B, C)
        streams = expand_to_streams(h, N)
        assert streams.shape == (S, B, N, C)
        # every stream starts as a copy of the embedding
        assert torch.allclose(streams[:, :, 0], h)
        assert collapse_streams(streams).shape == (S, B, C)

    def test_attn_res_state_accumulates_and_seals(self):
        h = torch.randn(S, B, C)
        state = DepthResidualState(h, layers_per_block=2)
        assert state.blocks == [h] or torch.allclose(state.blocks[0], h)
        accumulate_block(state, torch.randn(S, B, C))
        assert len(state.sources()) == 2  # embedding + partial
        state.seal_block()
        assert len(state.blocks) == 2 and state.partial is None

    def test_should_seal_respects_block_size(self):
        state = DepthResidualState(torch.randn(S, B, C), layers_per_block=3)
        assert [state.should_seal(i) for i in range(1, 7)] == [
            False,
            False,
            True,
            False,
            False,
            True,
        ]

    def test_payload_roundtrip(self):
        """The pipeline payload must reconstruct the state exactly."""
        h = torch.randn(S, B, C)
        state = DepthResidualState(h, layers_per_block=2)
        accumulate_block(state, torch.randn(S, B, C))
        payload = state.to_payload()
        before = state.collapse().clone()

        restored = DepthResidualState(torch.zeros_like(h), layers_per_block=2)
        restored.load_payload(payload, has_partial=True)
        assert torch.allclose(restored.collapse(), before, atol=1e-6)

    def test_payload_slots_matches_expansion(self):
        """AttnRes block count plays the same role as the mHC expansion rate."""
        h = torch.randn(S, B, C)
        assert expand_to_streams(h, N).permute(2, 0, 1, 3).shape == (N, S, B, C)
        ar = DepthResidualState(h, layers_per_block=2)
        accumulate_block(ar, torch.randn(S, B, C))
        assert ar.to_payload().shape[1:] == (S, B, C)

    def test_accumulate_with_lerp_stays_on_sphere(self):
        lerp = NGPTResidual(C, lerp_normalization="analytic")
        state = DepthResidualState(
            torch.nn.functional.normalize(torch.randn(S, B, C), dim=-1), layers_per_block=2
        )
        # The LERP variant must be used consistently for the whole block: the
        # seeding contribution is normalized so the running summary starts on the
        # sphere, and every later step keeps it there.
        for _ in range(4):
            accumulate_block(state, torch.randn(S, B, C), lerp=lerp)
        assert torch.allclose(state.partial.norm(dim=-1), torch.ones(S, B), atol=1e-4)
