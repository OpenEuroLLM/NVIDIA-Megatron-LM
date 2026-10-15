# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Routed residual streams (mhc_routed_topk): the fused kernels and the models."""

import pytest
import torch

from megatron.core.fusions import fused_routed_residual as rr
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.routed_residual import RoutedStreamMixer
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not (rr.HAVE_TRITON and torch.cuda.is_available()), reason="needs Triton and a GPU"
)


def _stack(fused, x0, weights, routers, n, k):
    """A small stack of routed sublayers, fused (in place) or out of place."""
    s = x0.unsqueeze(-2).expand(-1, -1, n, -1).contiguous()
    c = x0.shape[-1]
    for w_f, w_r in zip(weights, routers):
        stash = rr.GradStash()
        s0 = rr.stream_tap(s, 0, stash) if fused else s.select(-2, 0)
        logits = s0.reshape(-1, c).float() @ w_r
        idx, r = rr.route_streams(logits, k)
        w = 2 * torch.sigmoid(torch.gather(logits, 1, idx.long()))
        if fused:
            x = rr.routed_read(s, idx, r, stash)
        else:
            x = rr.routed_read_reference(s.reshape(-1, n, c), idx, r).view(x0.shape)
        f = torch.tanh(x.float() @ w_f).to(s.dtype)
        if fused:
            s = rr.routed_write(s, f, idx, w, stash)
        else:
            s = rr.routed_write_reference(s.reshape(-1, n, c), f.reshape(-1, c), idx, w)
            s = s.view(x0.shape[:-1] + (n, c))
    return s.mean(-2)  # the collapse: an expanded gradient arrives at the last write


@pytest.mark.parametrize("n,k", [(2, 1), (2, 2), (4, 2), (4, 3), (8, 4)])
@pytest.mark.parametrize("c", [64, 200, 2304])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_stack_matches_reference(n, k, c, dtype):
    gen = torch.Generator(device="cuda").manual_seed(0)
    layers = 3
    weights = [torch.randn(c, c, device="cuda", generator=gen) / c**0.5 for _ in range(layers)]
    routers = [torch.randn(c, n, device="cuda", generator=gen) for _ in range(layers)]
    x0 = torch.randn(5, 3, c, device="cuda", generator=gen).to(dtype)
    dy = torch.randn(5, 3, c, device="cuda", generator=gen).to(dtype)
    results = []
    for fused in (True, False):
        x = x0.clone().requires_grad_()
        params = [p.clone().requires_grad_() for p in weights + routers]
        y = _stack(fused, x, params[:layers], params[layers:], n, k)
        y.backward(dy)
        results.append([y.detach(), x.grad] + [p.grad for p in params])
    for a, b in zip(*results):
        if dtype == torch.float32:
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
        else:  # bf16 gradient buffers: summation order differs, compare in norm
            assert (a.float() - b.float()).norm() <= 5e-2 * b.float().norm()


def test_write_is_in_place_and_touches_only_the_routed_streams():
    s = torch.randn(4, 2, 4, 32, device="cuda")
    before = s.clone()
    idx = torch.tensor([[0, 2]] * 8, device="cuda", dtype=torch.int32)
    w = torch.ones(8, 2, device="cuda")
    f = torch.randn(4, 2, 32, device="cuda")
    out = rr.routed_write(s, f, idx, w, rr.GradStash())
    assert out.data_ptr() == s.data_ptr()
    torch.testing.assert_close(out[:, :, 1], before[:, :, 1])
    torch.testing.assert_close(out[:, :, 3], before[:, :, 3])
    torch.testing.assert_close(out[:, :, 2], before[:, :, 2] + f)


def test_route_streams_always_keeps_the_main_stream():
    logits = torch.randn(100, 6, device="cuda")
    logits[:, 0] = -1e4  # the main stream is selected regardless of its logit
    idx, r = rr.route_streams(logits, 3)
    assert (idx[:, 0] == 0).all()
    assert all(len(set(row.tolist())) == 3 for row in idx)
    expected = logits[:, 1:].topk(2, dim=-1).indices + 1
    assert (idx[:, 1:].long().sort(-1).values == expected.sort(-1).values).all()
    torch.testing.assert_close(r.sum(-1), torch.ones(100, device="cuda"))


def test_single_stream_is_the_plain_residual():
    mixer = RoutedStreamMixer(hidden_size=32, expansion_rate=1, topk=1).cuda()
    h = torch.randn(6, 2, 1, 32, device="cuda")
    pre, post, res = mixer(h)
    x = mixer.read(h, pre)
    torch.testing.assert_close(x, h[:, :, 0])
    f = torch.randn(6, 2, 32, device="cuda")
    expected = h[:, :, 0] + f  # write weight 2 sigmoid(0) = 1 at init
    out = mixer.write(h, f, post, res)
    torch.testing.assert_close(out[:, :, 0], expected)


def _hybrid_model(pattern, **kw):
    config = TransformerConfig(
        num_layers=len(pattern),
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        normalization="RMSNorm",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        attention_backend=AttnBackend.unfused,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        is_hybrid_model=True,
        mhc=True,
        mhc_expansion_rate=4,
        mhc_routed_topk=2,
        pkm_heads=2,
        pkm_num_keys=16,
        pkm_key_dim=16,
        pkm_topk=4,
        **kw,
    )
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=32,
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    ).cuda()


class TestRoutedHybridModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def test_fused_model_matches_out_of_place(self, monkeypatch):
        """The whole model -- in-place streams and the shared gradient buffer -- gives
        the gradients of the out-of-place torch definition."""
        torch.manual_seed(0)
        model = _hybrid_model("*-P-C-*-")
        ids = torch.randint(0, 128, (2, 16), device="cuda")
        grads = []
        for fused in (True, False):
            if not fused:
                monkeypatch.setattr(rr, "_fused", lambda streams: False)
            model.zero_grad(set_to_none=True)
            logits = model(ids, position_ids=None, attention_mask=None)
            logits.float().logsumexp(-1).mean().backward()
            grads.append({n: p.grad.clone() for n, p in model.named_parameters()})
        assert set(grads[0]) == set(grads[1])
        for name in grads[0]:
            torch.testing.assert_close(grads[0][name], grads[1][name], rtol=1e-4, atol=1e-5)
        routers = [n for n in grads[0] if n.endswith("mhc_attn_mixer.phi")]
        assert routers and all(grads[0][n].abs().sum() > 0 for n in routers)

    def test_trains(self):
        model = _hybrid_model("*-C-*-")
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        ids = torch.randint(0, 128, (4, 16), device="cuda")
        losses = []
        for _ in range(8):
            logits = model(ids[:, :-1], position_ids=None, attention_mask=None)
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, 128), ids[:, 1:].reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]

    def test_config_validation(self):
        with pytest.raises(ValueError, match="requires mhc"):
            TransformerConfig(
                num_layers=2, hidden_size=64, num_attention_heads=4, mhc_routed_topk=2
            )
        with pytest.raises(ValueError, match="must be in"):
            TransformerConfig(
                num_layers=2,
                hidden_size=64,
                num_attention_heads=4,
                mhc=True,
                mhc_expansion_rate=2,
                mhc_routed_topk=3,
            )
        with pytest.raises(ValueError, match="CUDA graphs"):
            TransformerConfig(
                num_layers=2,
                hidden_size=64,
                num_attention_heads=4,
                mhc=True,
                mhc_routed_topk=2,
                cuda_graph_impl="local",
            )
