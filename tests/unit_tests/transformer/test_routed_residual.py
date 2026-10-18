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


def _stack(fused, x0, weights, routers, n, k, mixers=None):
    """A small stack of routed sublayers, fused (in place) or out of place; with
    ``mixers`` ([c, k*k] projections) the write also mixes the routed streams."""
    s = x0.unsqueeze(-2).expand(-1, -1, n, -1).contiguous()
    c = x0.shape[-1]
    for layer, (w_f, w_r) in enumerate(zip(weights, routers)):
        stash = rr.GradStash()
        s0 = rr.stream_tap(s, 0, stash) if fused else s.select(-2, 0)
        # The test's own dense maps run in float64: no TF32 (on by default in NGC
        # containers), which would round the two paths' inputs differently.
        logits = (s0.reshape(-1, c).double() @ w_r.double()).float()
        # Select from the fixed input, so both paths take the same streams: selecting
        # from the stream values would let a rounding difference flip a top-k choice
        # and change the gradients discontinuously. The router still learns through
        # the read / write weights, which are computed from the streams.
        idx, _ = rr.route_streams(x0.reshape(-1, c).detach().float() @ w_r.detach(), k)
        r = torch.softmax(torch.gather(logits, 1, idx.long()), dim=-1)
        w = 2 * torch.sigmoid(torch.gather(logits, 1, idx.long()))
        if fused:
            x = rr.routed_read(s, idx, r, stash)
        else:
            x = rr.routed_read_reference(s.reshape(-1, n, c), idx, r).view(x0.shape)
        f = torch.tanh(x.double() @ w_f.double()).to(s.dtype)
        if mixers is not None:
            from megatron.core.transformer.mhc import sinkhorn_knopp

            mix_logits = (s0.reshape(-1, c).double() @ mixers[layer].double()).float()
            mix_logits = mix_logits.view(-1, k, k)
            mix = sinkhorn_knopp(mix_logits + torch.eye(k, device=s.device), 6)
            if fused:
                s = rr.routed_write_mix(s, f, idx, w, mix, stash)
            else:
                s = rr.routed_write_mix_reference(
                    s.reshape(-1, n, c), f.reshape(-1, c), idx, w, mix
                ).view(x0.shape[:-1] + (n, c))
        elif fused:
            s = rr.routed_write(s, f, idx, w, stash)
        else:
            s = rr.routed_write_reference(s.reshape(-1, n, c), f.reshape(-1, c), idx, w)
            s = s.view(x0.shape[:-1] + (n, c))
    return s.mean(-2)  # the collapse: an expanded gradient arrives at the last write


@pytest.mark.parametrize("n,k", [(2, 1), (2, 2), (4, 2), (4, 3), (8, 4)])
@pytest.mark.parametrize("c", [64, 200, 2304])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mix", [False, True])
def test_fused_stack_matches_reference(n, k, c, dtype, mix):
    if mix and k == 1:
        pytest.skip("nothing to mix with one routed stream")
    gen = torch.Generator(device="cuda").manual_seed(0)
    layers = 3
    weights = [torch.randn(c, c, device="cuda", generator=gen) / c**0.5 for _ in range(layers)]
    # O(1) router logits: unscaled ones (O(sqrt(c))) saturate the softmax / sigmoid,
    # leaving near-zero, cancellation-dominated router gradients.
    routers = [torch.randn(c, n, device="cuda", generator=gen) / c**0.5 for _ in range(layers)]
    mixers = [torch.randn(c, k * k, device="cuda", generator=gen) / c**0.5 for _ in range(layers)]
    x0 = torch.randn(5, 3, c, device="cuda", generator=gen).to(dtype)
    dy = torch.randn(5, 3, c, device="cuda", generator=gen).to(dtype)

    def run(fused, dt):
        x = x0.to(dt).clone().requires_grad_()
        params = [p.clone().requires_grad_() for p in weights + routers + mixers]
        y = _stack(
            fused, x, params[:layers], params[layers : 2 * layers], n, k,
            mixers=params[2 * layers :] if mix else None,
        )
        y.backward(dy.to(dt))
        return [y.detach().float(), x.grad.float()] + [
            p.grad.float() for p in params if p.grad is not None
        ]

    fused, ref = run(True, dtype), run(False, dtype)
    if dtype == torch.float32:
        for a, b in zip(fused, ref):
            # In norm: large reductions (C = 2304) reorder sums.
            assert (a - b).norm() <= 1e-4 * b.norm() + 1e-4
        return
    # bf16: a 3-layer bf16 chain can itself be far from fp32 in both paths (on H100,
    # n = 8, k = 4, C = 2304 drifted by O(1) in both), so measure the fused kernels
    # against an fp32 ground truth on the same bf16-rounded inputs and routing, and
    # require them to be no worse than the out-of-place bf16 definition.
    truth = run(False, torch.float32)
    for a, r, t in zip(fused, ref, truth):
        # In norm, with an absolute floor: some gradients vanish in exact arithmetic
        # (the first mixer sees n identical streams), leaving only rounding in both.
        err_fused, err_ref = (a - t).norm(), (r - t).norm()
        assert err_fused <= 1.5 * err_ref + 2e-2 * t.norm() + 1e-4, (
            float(err_fused), float(err_ref), float(t.norm())
        )


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

    @pytest.mark.parametrize("mix", [False, True])
    def test_fused_model_matches_out_of_place(self, monkeypatch, mix):
        """The whole model -- in-place streams and the shared gradient buffer -- gives
        the gradients of the out-of-place torch definition."""
        torch.manual_seed(0)
        model = _hybrid_model("*-P-C-*-", mhc_routed_mix=mix)
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
        # per-layer "local" CUDA graphs are verified against eager
        # (test_depth_residual_cuda_graphs.py); other graph implementations stay opt-in
        TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            mhc=True,
            mhc_routed_topk=2,
            cuda_graph_impl="local",
        )
        with pytest.raises(ValueError, match="cuda_graph_impl='local' only"):
            TransformerConfig(
                num_layers=2,
                hidden_size=64,
                num_attention_heads=4,
                mhc=True,
                mhc_routed_topk=2,
                cuda_graph_impl="transformer_engine",
            )


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("c", [64, 2304])
def test_bf16_mixing_layer_against_fp32(k, c):
    """One routed read + mixing write in bf16 (in place, fused) against the fp32
    out-of-place definition on the same bf16 inputs: every gradient within 1%."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    t, n = 40, 6
    s0 = torch.randn(t, n, c, device="cuda", generator=gen).bfloat16()
    idx = torch.stack([torch.randperm(n, generator=torch.Generator().manual_seed(i))[:k]
                       for i in range(t)]).int().cuda()
    r0 = torch.softmax(torch.randn(t, k, device="cuda", generator=gen), -1)
    w0 = torch.rand(t, k, device="cuda", generator=gen) + 0.5
    m0 = torch.softmax(torch.randn(t, k, k, device="cuda", generator=gen), -1)
    wf = torch.randn(c, c, device="cuda", generator=gen) / c**0.5
    dy = torch.randn(t, n, c, device="cuda", generator=gen)
    grads = []
    for fused in (True, False):
        s = s0.clone() if fused else s0.float().clone()
        s.requires_grad_()
        base = s * 1.0  # non-leaf, so the fused write may update it in place
        r, w, m = (v.clone().requires_grad_() for v in (r0, w0, m0))
        stash = rr.GradStash()
        if fused:
            x = rr.routed_read(base, idx, r, stash)
        else:
            x = rr.routed_read_reference(base, idx, r)
        f = torch.tanh(x.double() @ wf.double()).to(base.dtype)
        if fused:
            out = rr.routed_write_mix(base, f, idx, w, m, stash)
        else:
            out = rr.routed_write_mix_reference(base, f, idx, w, m)
        out.float().backward(dy)
        grads.append([s.grad, r.grad, w.grad, m.grad])
    for a, b in zip(*grads):
        assert (a.float() - b.float()).norm() <= 1e-2 * b.float().norm(), (a, b)


def test_mixing_write_matches_reference_and_keeps_other_streams():
    s = torch.randn(6, 5, 64, device="cuda")
    before = s.clone()
    idx = torch.tensor([[0, 3, 1]] * 6, device="cuda", dtype=torch.int32)
    w = torch.rand(6, 3, device="cuda")
    mix = torch.softmax(torch.randn(6, 3, 3, device="cuda"), -1)
    f = torch.randn(6, 64, device="cuda")
    expected = rr.routed_write_mix_reference(before, f, idx, w, mix)
    stash = rr.GradStash()
    rr.routed_read(s, idx, torch.full((6, 3), 1 / 3, device="cuda"), stash)
    out = rr.routed_write_mix(s, f, idx, w, mix, stash)
    assert out.data_ptr() == s.data_ptr()
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[:, [2, 4]], before[:, [2, 4]])


class TestRoutedMixAndBalance:
    @pytest.fixture(autouse=True)
    def setup(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def test_balancing_bias_moves_toward_underpicked_streams(self):
        from megatron.core.transformer.routed_residual import update_routed_residual_bias

        model = _hybrid_model("*-*-", mhc_routed_balance_rate=1e-2)
        mixer = model.decoder.layers[0].mhc_attn_mixer
        assert mixer.balance and mixer.route_bias.dtype == torch.float32
        model.bfloat16()
        assert mixer.route_bias.dtype == torch.float32  # kept fp32 under a bf16 cast
        model.float()
        with torch.no_grad():  # make stream 1 the router's favourite
            mixer.b_route[1] = 10.0
        ids = torch.randint(0, 128, (2, 16), device="cuda")
        model(ids, position_ids=None, attention_mask=None).float().mean().backward()
        picks = mixer.local_picks_per_stream.clone()
        assert picks[0] == 0 and picks[1] == picks.sum() == 32  # k-1 = 1 pick per token
        update_routed_residual_bias([model], model.config)
        bias = mixer.route_bias
        assert bias[1] < 0 and (bias[2:] > 0).all() and bias[0] == 0
        assert mixer.local_picks_per_stream.sum() == 0


    def test_mixing_starts_near_identity_and_trains(self):
        model = _hybrid_model("*-C-*-", mhc_routed_mix=True, mhc_routed_balance_rate=1e-3)
        mixer = model.decoder.layers[0].mhc_attn_mixer
        assert mixer.mix
        s = torch.randn(4, 2, 4, 64, device="cuda")
        _, (idx, w, mix, _), _ = mixer(s)
        assert torch.diagonal(mix, dim1=-2, dim2=-1).min() > 0.95
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        ids = torch.randint(0, 128, (4, 16), device="cuda")
        losses = []
        for _ in range(6):
            logits = model(ids[:, :-1], position_ids=None, attention_mask=None)
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, 128), ids[:, 1:].reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]
        assert mixer.b_res.grad is not None and mixer.alpha_res.grad is not None


class TestReversibleRoutedResidual:
    """mhc_routed_reversible: per sublayer only the change is kept; the backward undoes
    the writes to recover the streams the reads and routers saw."""

    @pytest.fixture(autouse=True)
    def setup(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def _grads(self, reversible, dtype):
        torch.manual_seed(0)
        model_parallel_cuda_manual_seed(0)
        model = _hybrid_model("*-P-C-*-", mhc_routed_reversible=reversible)
        if dtype == torch.float32:
            model = model.float()
        ids = torch.randint(0, 128, (2, 16), device="cuda",
                            generator=torch.Generator("cuda").manual_seed(3))
        logits = model(ids, position_ids=None, attention_mask=None)
        logits.float().logsumexp(-1).mean().backward()
        return logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters()}

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_matches_the_stored_backward(self, dtype):
        out_a, ga = self._grads(False, dtype)
        out_b, gb = self._grads(True, dtype)
        torch.testing.assert_close(out_a, out_b)
        assert set(ga) == set(gb)
        for name in ga:
            # the residual path does not read the restored state: exact; the routers
            # do, up to the undo's rounding
            tol = 1e-4 if "mixer" in name else 1e-5
            torch.testing.assert_close(gb[name], ga[name], rtol=tol, atol=tol, msg=name)

    def test_streams_are_restored_after_backward(self):
        from megatron.core.transformer.routed_residual import RoutedStreamMixer

        mixer = RoutedStreamMixer(64, 4, 2, reversible=True).cuda()
        s = torch.randn(8, 2, 4, 64, device="cuda")
        start = s.clone()
        base = s.clone().requires_grad_()
        streams = base * 1.0
        pre, post, _ = mixer(streams)
        x = mixer.read(streams, pre)
        out = mixer.write(streams, torch.tanh(x), post)
        out.sum().backward()
        torch.testing.assert_close(streams.detach(), start, rtol=1e-6, atol=1e-6)

    def test_rejects_mixing(self):
        with pytest.raises(ValueError, match="mhc_routed_reversible"):
            from megatron.core.transformer.routed_residual import RoutedStreamMixer

            RoutedStreamMixer(64, 4, 2, mix=True, reversible=True)


# ----------------------------------------------------------------- LERP write (nGPT)


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def _lerp_layer(fused, reversible, s_in, a0, idx, r0, w0, al0, mode, dy):
    """One routed read + LERP write on ``s_in`` [T, n, C]; returns the output and the
    gradients of (streams, a, r, w, alpha)."""
    s = s_in.clone().requires_grad_()
    base = s * 1.0  # non-leaf, so the fused write may update it in place
    a, r, w, al = (v.clone().requires_grad_() for v in (a0, r0, w0, al0))
    stash = rr.GradStash()
    if not fused:
        x = rr.routed_read_reference(base, idx, r)
        out = rr.routed_write_lerp_reference(base, a, idx, w, al, mode, 1e-6)
    else:
        read = rr.routed_read_reversible if reversible else rr.routed_read
        x = read(base, idx, r, stash)
        out = rr.routed_write_lerp(base, a, idx, w, al, mode, 1e-6, stash, reversible=reversible)
    written = out.detach().clone()  # a reversible backward restores the buffer in place
    # the read feeds the loss too, so its backward runs after the write's
    ((out.double() * dy).sum() + (x.double() * dy[:, 0]).sum()).backward()
    return [written] + [v.grad for v in (s, a, r, w, al)], base


@pytest.mark.parametrize("mode", rr.LERP_WRITE_MODES)
@pytest.mark.parametrize("n,k", [(2, 1), (4, 2), (8, 3)])
@pytest.mark.parametrize("c", [64, 200, 2304])
@pytest.mark.parametrize("reversible", [False, True])
def test_lerp_write_matches_reference(mode, n, k, c, reversible):
    """The fused LERP write (and its reversible form) against the float64 out-of-place
    definition: output and every gradient, on unit-norm streams."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    t = 37
    s0 = _unit(torch.randn(t, n, c, device="cuda", generator=gen, dtype=torch.float64))
    a0 = _unit(torch.randn(t, c, device="cuda", generator=gen, dtype=torch.float64))
    idx = torch.stack(
        [torch.randperm(n, generator=torch.Generator().manual_seed(i))[:k] for i in range(t)]
    ).int().cuda()
    r0 = torch.softmax(torch.randn(t, k, device="cuda", generator=gen, dtype=torch.float64), -1)
    w0 = 2 * torch.sigmoid(torch.randn(t, k, device="cuda", generator=gen, dtype=torch.float64))
    al0 = 0.05 + 0.2 * torch.rand(c, device="cuda", generator=gen, dtype=torch.float64)
    dy = torch.randn(t, n, c, device="cuda", generator=gen, dtype=torch.float64)

    ref, _ = _lerp_layer(False, False, s0, a0, idx, r0, w0, al0, mode, dy)
    fused, base = _lerp_layer(
        True, reversible, s0.float(), a0.float(), idx, r0.float(), w0.float(), al0.float(),
        mode, dy,
    )
    names = ["out", "d_streams", "d_a", "d_r", "d_w", "d_alpha"]
    for name, x, y in zip(names, fused, ref):
        err = (x.double() - y).norm() / y.norm().clamp(min=1e-12)
        assert err < 1e-5, (name, float(err))
    # direct lands exactly on the sphere (approx / analytic only for a scalar step)
    if mode == "direct":
        torch.testing.assert_close(
            fused[0].norm(dim=-1), torch.ones(t, n, device="cuda"), rtol=1e-5, atol=1e-5
        )
    if reversible:  # the backward restored the streams it wrote
        torch.testing.assert_close(base.detach().double(), s0, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode", rr.LERP_WRITE_MODES)
def test_lerp_write_bf16_against_fp32(mode):
    """bf16 streams, in place: within 1% of the fp32 definition on the same inputs."""
    gen = torch.Generator(device="cuda").manual_seed(1)
    t, n, k, c = 40, 6, 3, 2304
    s0 = _unit(torch.randn(t, n, c, device="cuda", generator=gen)).bfloat16()
    a0 = _unit(torch.randn(t, c, device="cuda", generator=gen)).bfloat16()
    idx = torch.stack(
        [torch.randperm(n, generator=torch.Generator().manual_seed(i))[:k] for i in range(t)]
    ).int().cuda()
    r0 = torch.softmax(torch.randn(t, k, device="cuda", generator=gen), -1)
    w0 = 2 * torch.sigmoid(torch.randn(t, k, device="cuda", generator=gen))
    al0 = 0.05 + 0.2 * torch.rand(c, device="cuda", generator=gen)
    dy = torch.randn(t, n, c, device="cuda", generator=gen, dtype=torch.float64)
    fused, _ = _lerp_layer(True, False, s0, a0, idx, r0, w0, al0, mode, dy)
    truth, _ = _lerp_layer(False, False, s0.float(), a0.float(), idx, r0, w0, al0, mode, dy)
    for x, y in zip(fused, truth):
        assert (x.float() - y.float()).norm() <= 1e-2 * y.float().norm()


@pytest.mark.parametrize("mode", rr.LERP_WRITE_MODES)
def test_lerp_write_with_every_stream_is_the_ngpt_residual(mode):
    """k = n and write weight 1: every stream takes exactly NGPTResidual's step."""
    from megatron.core.transformer.ngpt import NGPTResidual

    torch.manual_seed(0)
    n, c = 4, 64
    mixer = RoutedStreamMixer(c, n, n).cuda()
    with torch.no_grad():
        mixer.alpha_post.zero_()  # w = 2 sigmoid(0) = 1 for every row
    lerp = NGPTResidual(c, alpha_init_value=0.1, lerp_normalization=mode).cuda()
    with torch.no_grad():
        lerp.alpha.mul_(1.0 + 0.5 * torch.rand(c, device="cuda"))  # per-channel steps
    s = _unit(torch.randn(6, 2, n, c, device="cuda"))
    f = torch.randn(6, 2, c, device="cuda")
    expected = torch.stack([lerp(f, s[:, :, i]) for i in range(n)], dim=2)
    pre, post, res = mixer(s * 1.0)
    streams = s.clone()
    mixer.read(streams, pre)
    out = mixer.write(streams, f, post, res, lerp=lerp)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


class TestRoutedLerpAndKeys:
    @pytest.fixture(autouse=True)
    def setup(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "kw",
        [
            dict(mhc_routed_key_dim=8),
            dict(mhc_lerp=True, ngpt=True, ngpt_lerp_normalization="analytic"),
            dict(mhc_lerp=True, ngpt=True, ngpt_lerp_normalization="direct",
                 mhc_routed_key_dim=8),
        ],
        ids=["keys", "lerp", "keys_lerp"],
    )
    def test_fused_model_matches_out_of_place(self, monkeypatch, kw):
        torch.manual_seed(0)
        model = _hybrid_model("*-P-C-*-", **kw)
        if "mhc_routed_key_dim" in kw:
            with torch.no_grad():  # a live content score, so its path is checked too
                for m in model.modules():
                    if isinstance(m, RoutedStreamMixer):
                        m.key_gain.fill_(0.5)
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
            torch.testing.assert_close(
                grads[0][name], grads[1][name], rtol=1e-4, atol=1e-5, msg=name
            )
        if "mhc_routed_key_dim" in kw:
            keys = [n for n in grads[0] if n.endswith("w_key")]
            assert keys and all(grads[0][n].abs().sum() > 0 for n in keys)

    @pytest.mark.parametrize("mode", rr.LERP_WRITE_MODES)
    def test_reversible_lerp_matches_the_stored_backward(self, mode):
        def run(reversible):
            torch.manual_seed(0)
            model_parallel_cuda_manual_seed(0)
            model = _hybrid_model(
                "*-P-C-*-", mhc_lerp=True, ngpt=True, ngpt_lerp_normalization=mode,
                mhc_routed_key_dim=8, mhc_routed_reversible=reversible,
            ).float()
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, RoutedStreamMixer):
                        m.key_gain.fill_(0.5)
            ids = torch.randint(0, 128, (2, 16), device="cuda",
                                generator=torch.Generator("cuda").manual_seed(3))
            logits = model(ids, position_ids=None, attention_mask=None)
            logits.float().logsumexp(-1).mean().backward()
            return logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters()}

        out_a, ga = run(False)
        out_b, gb = run(True)
        torch.testing.assert_close(out_a, out_b)
        for name in ga:
            torch.testing.assert_close(gb[name], ga[name], rtol=1e-4, atol=1e-4, msg=name)

    def test_zero_key_gain_is_the_positional_router(self):
        """At init (gain 0) the content router routes, reads and writes exactly as the
        positional one with the same weights, and the gain still gets a gradient."""
        torch.manual_seed(0)
        plain = RoutedStreamMixer(64, 4, 2).cuda()
        keyed = RoutedStreamMixer(64, 4, 2, key_dim=8).cuda()
        with torch.no_grad():
            keyed.phi[:, : plain.phi.shape[1]].copy_(plain.phi)
        s = torch.randn(8, 2, 4, 64, device="cuda")
        (i0, r0, _), (_, w0, _, _), _ = plain(s.clone())
        streams = s.clone()
        (i1, r1, st), (_, w1, _, _), _ = keyed(streams)
        assert torch.equal(i0, i1)
        # equal up to fp32 summation order: the positional router runs the fused projection
        # kernel (fused_router_coeffs), the content router the torch ops
        torch.testing.assert_close(r1, r0, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(w1, w0, rtol=1e-6, atol=1e-6)
        keyed.read(streams, (i1, r1, st)).sum().backward()
        assert keyed.key_gain.grad is not None and keyed.key_gain.grad.abs() > 0

    def test_lerp_trains(self):
        model = _hybrid_model(
            "*-C-*-", mhc_lerp=True, ngpt=True, ngpt_lerp_normalization="analytic",
            mhc_routed_key_dim=8,
        )
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
        mixer = model.decoder.layers[0].mhc_attn_mixer
        assert mixer.key_gain.grad is not None
        assert model.decoder.layers[0].ngpt_attn_residual.alpha.grad is not None

    def test_config_validation(self):
        base = dict(num_layers=2, hidden_size=64, num_attention_heads=4)
        with pytest.raises(ValueError, match="_key_dim need mhc_routed_topk"):
            TransformerConfig(**base, mhc=True, mhc_routed_key_dim=8)
        with pytest.raises(ValueError, match="mhc_routed_key_dim must be in"):
            TransformerConfig(**base, mhc=True, mhc_routed_topk=2, mhc_routed_key_dim=65)
        with pytest.raises(ValueError, match="does not support mhc_routed_mix"):
            TransformerConfig(
                **base, mhc=True, mhc_routed_topk=2, mhc_routed_mix=True, mhc_lerp=True,
                ngpt=True,
            )
        with pytest.raises(ValueError, match="supports ngpt_lerp_normalization"):
            TransformerConfig(
                **base, mhc=True, mhc_routed_topk=2, mhc_lerp=True, ngpt=True,
                ngpt_lerp_normalization_ffn="direct_ste",
            )
