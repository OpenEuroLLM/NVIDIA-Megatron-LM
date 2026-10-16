# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Power retention (the ``power`` linear variant) against its quadratic definition."""

import pytest
import torch
import torch.nn.functional as F

power = pytest.importorskip("megatron.core.ssm.power_retention")
if not power.HAVE_RETENTION or not torch.cuda.is_available():
    pytest.skip("needs the retention package and a GPU", allow_module_level=True)


def naive_power_attention(q, k, v, log_g, deg, scale):
    """o_t = sum_{s<=t} exp(L_t - L_s) (scale q_t.k_s)^deg v_s / sum(...), L = cumsum(log g)."""
    b, t, h, d = q.shape
    s = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) * scale
    w = s.pow(deg)
    if log_g is not None:
        L = log_g.float().cumsum(1).transpose(1, 2)  # [b, h, t]
        w = w * (L[..., :, None] - L[..., None, :]).clamp(max=0).exp()
    causal = torch.ones(t, t, device=q.device, dtype=torch.bool).tril()
    w = w.masked_fill(~causal, 0.0)
    o = torch.einsum("bhts,bshd->bthd", w, v.float())
    return o / (w.sum(-1).transpose(1, 2)[..., None] + 1e-6)


@pytest.mark.parametrize("t", [64, 384])  # within one chunk, and chunked
@pytest.mark.parametrize("deg", [2])  # the Triton kernels' chunked path is deg = 2 only
@pytest.mark.parametrize("gated", [False, True])
def test_kernel_matches_the_quadratic_definition(t, deg, gated):
    torch.manual_seed(0)
    b, h, d = 2, 2, 32
    q = torch.randn(b, t, h, d, device="cuda") * 0.5
    k = torch.randn(b, t, h, d, device="cuda") * 0.5
    v = torch.randn(b, t, h, d, device="cuda")
    log_g = F.logsigmoid(3.0 + torch.randn(b, t, h, device="cuda")) if gated else None
    ref = naive_power_attention(q, k, v, log_g, deg, d**-0.5)
    out = power.power_retention(
        q.bfloat16(), k.bfloat16(), v.bfloat16(), log_g, deg=deg, scale=d**-0.5,
        chunk_size=128 if t > 128 else None,
    ).float()
    assert (out - ref).norm() / ref.norm() < 3e-2


@pytest.mark.parametrize("gated", [False, True])
def test_flex_attention_form_matches_the_quadratic_definition(gated):
    torch.manual_seed(0)
    b, t, h, d = 2, 384, 2, 32
    q = torch.randn(b, t, h, d, device="cuda") * 0.5
    k = torch.randn(b, t, h, d, device="cuda") * 0.5
    v = torch.randn(b, t, h, d, device="cuda")
    log_g = F.logsigmoid(3.0 + torch.randn(b, t, h, device="cuda")) if gated else None
    leaves = [x.clone().requires_grad_() for x in (q, k, v)]
    ref = naive_power_attention(*leaves, log_g, 2, d**-0.5)
    out = power.power_attention_flex(*leaves, log_g, deg=2, scale=d**-0.5)
    assert (out - ref).norm() / ref.norm() < 3e-3  # TF32 / bf16 inside the kernel on H100
    do = torch.randn_like(ref)
    g_ref = torch.autograd.grad(ref, leaves, do)
    g = torch.autograd.grad(out, leaves, do)
    for a, r in zip(g, g_ref):
        assert (a - r).norm() / r.norm() < 1e-2


def test_attention_form_matches_chunked_form():
    """switch_over_seq_len: the quadratic form below it, the chunked one above -- same layer."""
    torch.manual_seed(0)
    layer = power.PowerRetentionAttention(hidden_size=128, head_dim=32, num_heads=4,
                                          chunk_size=128).cuda().to(torch.bfloat16)
    x = torch.randn(2, 512, 128, device="cuda", dtype=torch.bfloat16)
    outs, grads = [], []
    for switch in (0, 4096):
        layer.switch_over_seq_len = switch
        xi = x.clone().requires_grad_(True)
        o, _, _ = layer(xi)
        o.float().pow(2).mean().backward()
        outs.append(o.float())
        grads.append(xi.grad.float())
    assert (outs[0] - outs[1]).norm() / outs[1].norm() < 3e-2
    assert (grads[0] - grads[1]).norm() / grads[1].norm() < 5e-2


@pytest.mark.parametrize("rope,xsa", [(False, False), (True, False), (False, True)])
def test_layer_trains(rope, xsa):
    torch.manual_seed(0)
    layer = power.PowerRetentionAttention(hidden_size=128, head_dim=32, num_heads=4, rope=rope,
                                          xsa=xsa).cuda().to(torch.bfloat16)
    x = torch.randn(2, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    o, _, _ = layer(x)
    assert o.shape == x.shape and torch.isfinite(o).all()
    o.float().pow(2).mean().backward()
    assert all(p.grad is not None for p in layer.parameters())


def test_hybrid_model_with_power_layers_trains():
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
            params_dtype=torch.bfloat16, experimental_attention_variant="power",
            linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=4,
            linear_num_value_heads=4, linear_power_deg=2,
        )
        model = HybridModel(
            config=config, hybrid_stack_spec=hybrid_stack_spec, vocab_size=128,
            max_sequence_length=256, hybrid_layer_pattern="L-*-",
            position_embedding_type="none", share_embeddings_and_output_weights=False,
        ).cuda().to(torch.bfloat16)
        ids = torch.randint(0, 128, (2, 256), device="cuda")
        logits = model(ids, position_ids=None, attention_mask=None)
        assert torch.isfinite(logits).all()
        logits.float().logsumexp(-1).mean().backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, missing
    finally:
        Utils.destroy_model_parallel()


def test_layer_rejects_unsupported_degrees():
    with pytest.raises(ValueError, match="deg=2"):
        power.PowerRetentionAttention(hidden_size=64, head_dim=16, num_heads=4, deg=4)


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("gated", [False, True])
def test_triton_power_attention_matches_the_quadratic_definition(d, gated):
    from megatron.core.fusions.fused_power_attention import power_attention

    torch.manual_seed(0)
    b, t, h = 2, 320, 2
    q = torch.randn(b, t, h, d, device="cuda") * 0.5
    k = torch.randn(b, t, h, d, device="cuda") * 0.5
    v = torch.randn(b, t, h, d, device="cuda")
    log_g = F.logsigmoid(3.0 + torch.randn(b, t, h, device="cuda")) if gated else None
    ref_leaves = [x.double().requires_grad_() for x in (q, k, v)]
    ref_g = log_g.double().requires_grad_() if gated else None
    ref = naive_power_attention(*ref_leaves, ref_g, 2, d**-0.5)
    leaves = [x.bfloat16().requires_grad_() for x in (q, k, v)]
    g_leaf = log_g.clone().requires_grad_() if gated else None
    out = power_attention(*leaves, g_leaf, scale=d**-0.5)
    assert (out.double() - ref).norm() / ref.norm() < 1e-2
    do = torch.randn_like(ref)
    got = torch.autograd.grad(out.float(), leaves + ([g_leaf] if gated else []), do.float())
    want = torch.autograd.grad(ref, ref_leaves + ([ref_g] if gated else []), do)
    for a, r in zip(got, want):
        assert (a.double() - r).norm() / r.norm() < 5e-2
