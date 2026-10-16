# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Triadic delta rule (megatron/core/ssm/triadic.py): the KDA-kernel expansion against
the third-order recurrence, and the layer."""

import pytest
import torch
import torch.nn.functional as F

triadic = pytest.importorskip("megatron.core.ssm.triadic")
if not triadic.HAVE_FLA:
    pytest.skip("needs flash-linear-attention (ComplexKDA)", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)


def _inputs(b, t, h, d, e, signed, gen):
    def r(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    q, k = F.normalize(r(b, t, h, d), dim=-1), F.normalize(r(b, t, h, d), dim=-1)
    q2 = F.normalize(F.softplus(r(b, t, h, e)), dim=-1)
    k2 = F.normalize(F.softplus(r(b, t, h, e)), dim=-1)
    v = r(b, t, h, d)
    g2 = -F.softplus(r(b, t, h, e)) * 0.3
    g = -F.softplus(r(b, t, h, d)) * 0.3
    sign = torch.where(r(b, t, h, d) < -0.5, -1, 1).to(torch.int8) if signed else None
    beta_logits = r(b, t, h)
    return q, k, v, q2, k2, g2, g, sign, beta_logits


@pytest.mark.parametrize("e,d", [(1, 64), (2, 64), (4, 64), (2, 128), (4, 128), (8, 64), (8, 128)])
@pytest.mark.parametrize("base", ["gdn", "kda", "ckda"])
def test_kernel_expansion_matches_third_order_recurrence(e, d, base):
    gen = torch.Generator(device="cuda").manual_seed(0)
    b, t, h = 2, 96, 2
    q, k, v, q2, k2, g2, g, sign, bl = _inputs(b, t, h, d, e, base == "ckda", gen)
    if base == "gdn":
        g = None
    neg = base == "ckda"
    beta = torch.sigmoid(bl) * (2.0 if neg else 1.0)
    ref = triadic.triadic_reference(q, k, v, q2, k2, g2, beta, g=g, sign=sign)
    out = triadic.triadic_chunk(
        q, k, v.to(torch.bfloat16), q2, k2, g2, bl, g=g, sign=sign, allow_neg_eigval=neg
    ).float()
    err = (out - ref).norm() / ref.norm()
    assert err < 2e-2, err


def test_expansion_structure():
    """kappa = k2 (x) k is unit norm and the expanded decay is g2 (+) g."""
    gen = torch.Generator(device="cuda").manual_seed(1)
    q, k, v, q2, k2, g2, g, sign, _ = _inputs(1, 5, 2, 8, 3, True, gen)
    qq, kk, gg, ss = triadic.expand_triadic(q, k, q2, k2, g2, g, sign)
    torch.testing.assert_close(kk.norm(dim=-1), torch.ones(1, 5, 2, device="cuda"))
    torch.testing.assert_close(gg.view(1, 5, 2, 3, 8), g2[..., None] + g[..., None, :])
    assert (ss.view(1, 5, 2, 3, 8) == sign[..., None, :]).all()


@pytest.mark.parametrize("base", ["gdn", "kda", "ckda"])
@pytest.mark.parametrize("e", [1, 2, 4])
def test_layer_trains(base, e):
    torch.manual_seed(0)
    layer = triadic.TriadicDeltaAttention(hidden_size=128, head_dim=64, num_heads=2, e=e,
                                          base=base).cuda().to(torch.bfloat16)
    x = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    o, _, _ = layer(x)
    assert o.shape == x.shape and torch.isfinite(o).all()
    o.float().pow(2).mean().backward()
    missing = [n for n, p in layer.named_parameters() if p.grad is None]
    assert not missing, missing


@pytest.mark.parametrize("base", ["gdn", "kda", "ckda"])
def test_hybrid_model_with_triadic_layers_trains(base):
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.enums import AttnBackend
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=4, hidden_size=128, num_attention_heads=4, ffn_hidden_size=256,
            normalization="RMSNorm", add_bias_linear=False, gated_linear_unit=True,
            activation_func=torch.nn.functional.silu, attention_backend=AttnBackend.unfused,
            hidden_dropout=0.0, attention_dropout=0.0, is_hybrid_model=True, bf16=True,
            params_dtype=torch.bfloat16,
            experimental_attention_variant="triadic", linear_triadic_e=2,
            linear_triadic_base=base, linear_key_head_dim=64, linear_num_key_heads=2,
            linear_value_head_dim=64, linear_num_value_heads=2,
        )
        model = HybridModel(
            config=config, hybrid_stack_spec=hybrid_stack_spec, vocab_size=128,
            max_sequence_length=64, hybrid_layer_pattern="L-*-",
            position_embedding_type="none", share_embeddings_and_output_weights=False,
        ).cuda().to(torch.bfloat16)
        ids = torch.randint(0, 128, (2, 64), device="cuda")
        logits = model(ids, position_ids=None, attention_mask=None)
        assert torch.isfinite(logits).all()
        logits.float().logsumexp(-1).mean().backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, missing
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("e", [1, 2, 4, 8])
def test_cute_kernel_matches_reference(e):
    """cute-triadic-gdn (Triadic GDN, D = 128, sm90+) against the third-order recurrence,
    forward and gradients; and against the KDA expansion where both apply."""
    gdn_joint_call = triadic._cute_gdn()
    if gdn_joint_call is None:
        pytest.skip("needs cute-triadic-gdn on an sm90+ GPU")
    gen = torch.Generator(device="cuda").manual_seed(0)
    b, t, h, d = 1, 128, 2, 128
    q, k, v, q2, k2, g2, _, _, bl = _inputs(b, t, h, d, e, False, gen)
    beta = torch.sigmoid(bl)
    ref = triadic.triadic_reference(q, k, v, q2, k2, g2, beta)
    out = gdn_joint_call(q.bfloat16(), k.bfloat16(), v.bfloat16(), k2, q2, g2, beta,
                         scale=d**-0.5).float()
    assert (out - ref).norm() / ref.norm() < 2e-2
    if e * d <= 256:
        via_kda = triadic.triadic_chunk(q, k, v.bfloat16(), q2, k2, g2, bl).float()
        assert (out - via_kda).norm() / via_kda.norm() < 2e-2
    # gradients: the CuTe backward against autograd through the KDA expansion is
    # covered by cute-triadic-gdn's own tests; here, that they exist and are finite
    leaves = [x.clone().requires_grad_() for x in (q2, k2, g2, beta)]
    qb, kb, vb = (x.bfloat16().requires_grad_() for x in (q, k, v))
    gdn_joint_call(qb, kb, vb, *leaves[:2], leaves[2], leaves[3], scale=d**-0.5).float().sum().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in [qb, kb, vb] + leaves)


@pytest.mark.parametrize("e", [1, 2, 4, 8])
@pytest.mark.parametrize("base", ["kda", "ckda"])
def test_cute_kda_matches_reference(e, base):
    """cute-triadic-gdn's Triadic KDA / Complex KDA (sm90) against the third-order recurrence:
    forward and the gradient of every input."""
    chunk_kda_joint = triadic._cute_kda()
    if chunk_kda_joint is None:
        pytest.skip("needs cute-triadic-gdn with the KDA kernels on an sm90 GPU")
    gen = torch.Generator(device="cuda").manual_seed(5)
    b, t, h, d = 1, 128, 2, 128
    q, k, v, q2, k2, g2, g, sign, bl = _inputs(b, t, h, d, e, base == "ckda", gen)
    beta = torch.sigmoid(bl) * (2.0 if base == "ckda" else 1.0)
    leaves = {"q": q, "k": k, "v": v, "q2": q2, "k2": k2, "g2": g2, "g": g, "beta": beta}
    dy = torch.randn(b, t, h, d, device="cuda", generator=gen)
    results = []
    for path in ("cute", "reference"):
        xs = {n: x.detach().clone().requires_grad_() for n, x in leaves.items()}
        if path == "cute":
            q_, k_, v_ = (xs[n].bfloat16() for n in "qkv")
            out = chunk_kda_joint(q_, k_, v_, xs["k2"], xs["q2"], xs["g2"], xs["g"], xs["beta"],
                                  sign=sign, scale=d**-0.5).float()
        else:
            out = triadic.triadic_reference(xs["q"], xs["k"], xs["v"], xs["q2"], xs["k2"],
                                            xs["g2"], xs["beta"], g=xs["g"], sign=sign)
        out.backward(dy)
        results.append((out.detach(), {n: x.grad for n, x in xs.items()}))
    (o_c, g_c), (o_r, g_r) = results
    assert (o_c - o_r).norm() / o_r.norm() < 2e-2
    for name in leaves:
        a, r = g_c[name].float(), g_r[name].float()
        err = (a - r).norm() / r.norm().clamp(min=1e-6)
        assert err < 5e-2, (name, float(err))


@pytest.mark.parametrize("base", ["kda", "ckda"])
def test_layer_takes_the_cute_kda_kernels(base):
    if triadic._cute_kda() is None:
        pytest.skip("needs cute-triadic-gdn with the KDA kernels on an sm90 GPU")
    torch.manual_seed(0)
    layer = triadic.TriadicDeltaAttention(hidden_size=256, head_dim=128, num_heads=2, e=4,
                                          base=base, kernel="cute").cuda().to(torch.bfloat16)
    assert layer.cute is not None
    x = torch.randn(2, 128, 256, device="cuda", dtype=torch.bfloat16)
    o_cute = layer(x)[0].float()
    layer.cute = None                                          # the KDA-expansion path
    o_kda = layer(x)[0].float()
    assert (o_cute - o_kda).norm() / o_kda.norm() < 3e-2


@pytest.mark.parametrize("e,d", [(2, 64), (4, 128), (8, 64)])
@pytest.mark.parametrize("base", ["gdn", "kda", "ckda"])
def test_gradients_match_the_reference(e, d, base):
    """Every input's gradient through the KDA pipeline (FLA's kernels below K = 256, the
    wide-key state kernels above) against autograd through the third-order recurrence."""
    gen = torch.Generator(device="cuda").manual_seed(4)
    b, t, h = 1, 64, 2
    q, k, v, q2, k2, g2, g, sign, bl = _inputs(b, t, h, d, e, base == "ckda", gen)
    if base == "gdn":
        g = None
    neg = base == "ckda"
    leaves = {"q": q, "k": k, "v": v, "q2": q2, "k2": k2, "g2": g2, "bl": bl}
    if g is not None:
        leaves["g"] = g
    dy = torch.randn(b, t, h, d, device="cuda", generator=gen)
    grads = []
    for path in ("kernel", "reference"):
        xs = {n: x.detach().clone().requires_grad_() for n, x in leaves.items()}
        if path == "kernel":
            out = triadic.triadic_chunk(xs["q"], xs["k"], xs["v"].bfloat16(), xs["q2"], xs["k2"],
                                        xs["g2"], xs["bl"], g=xs.get("g"), sign=sign,
                                        allow_neg_eigval=neg).float()
        else:
            beta = torch.sigmoid(xs["bl"]) * (2.0 if neg else 1.0)
            out = triadic.triadic_reference(xs["q"], xs["k"], xs["v"], xs["q2"], xs["k2"],
                                            xs["g2"], beta, g=xs.get("g"), sign=sign)
        out.backward(dy)
        grads.append({n: x.grad for n, x in xs.items()})
    for name in leaves:
        a, r = grads[0][name].float(), grads[1][name].float()
        err = (a - r).norm() / r.norm().clamp(min=1e-6)
        assert err < 5e-2, (name, float(err))
