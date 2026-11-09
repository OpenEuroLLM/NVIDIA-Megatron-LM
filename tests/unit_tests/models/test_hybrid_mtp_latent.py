# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""MTP in latent space with CKDA depths (mtp_latent: depth k reads the normalised hidden state of depth k - 1,
no next-token embedding) and the strided fused MTP loss (mtp_loss_stride with a fused_loss_fn: the vocabulary
head on every k-th position only). The fused path at stride 1 equals the output-layer path; at stride k only the
sampled positions get gradients; a hybrid model with "/L/L/L" CKDA depths trains end to end. One GPU:

    python -m torch.distributed.run --nproc-per-node 1 -m pytest -q tests/unit_tests/models/test_hybrid_mtp_latent.py
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.multi_token_prediction import process_mtp_loss
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
S, B, H, V, DEPTHS = 24, 2, 16, 50, 3


@pytest.fixture
def mp():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def _cfg(**kw):
    return TransformerConfig(num_layers=1, hidden_size=H, num_attention_heads=2, mtp_num_layers=DEPTHS,
                             mtp_loss_scaling_factor=1.0, **kw)


def _grads(cfg, fused, seed=0):
    """The gradient of the MTP loss w.r.t. every depth's hidden states (the main chunk gets none)."""
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn((1 + DEPTHS) * S, B, H, generator=g, dtype=torch.float64).cuda().requires_grad_()
    weight = torch.randn(V, H, generator=g, dtype=torch.float64).cuda()
    labels = torch.randint(0, V, (B, S), generator=g).cuda()
    mask = torch.ones(B, S, dtype=torch.float64).cuda()

    def output_layer(h, weight=None, runtime_gather_output=None):
        return h @ weight.t(), None

    def lm_loss(labels, logits):  # [b, s] per-token CE from [s, b, v] logits
        return F.cross_entropy(logits.transpose(0, 1).reshape(-1, V), labels.reshape(-1),
                               reduction="none").view(labels.shape)

    def fused_fn(h, labels, loss_mask):
        return lm_loss(labels, h @ weight.t())

    out = process_mtp_loss(hidden, labels, mask, output_layer, weight, None, True, lm_loss, cfg,
                           fused_loss_fn=fused_fn if fused else None)
    out.sum().backward()  # the main chunk's own loss is zero here: only the MTP losses' gradients remain
    return hidden.grad.view(1 + DEPTHS, S, B, H)


def test_fused_stride_one_equals_output_layer_path(mp):
    """At stride 1 the fused path (one call over all depths) is the output-layer path up to the normalisation:
    the depths share the mean of their token counts (B * (S - 1 - d), the rolled-out tail) instead of their own."""
    ref = _grads(_cfg(), fused=False)
    got = _grads(_cfg(), fused=True)
    counts = torch.tensor([B * (S - 1 - d) for d in range(DEPTHS)], dtype=torch.float64)
    ratio = counts / counts.mean()
    torch.testing.assert_close(got[0], ref[0], atol=1e-12, rtol=1e-10)
    for d in range(DEPTHS):
        torch.testing.assert_close(got[1 + d], ref[1 + d] * ratio[d], atol=1e-12, rtol=1e-10)
    assert ref[1:].abs().sum() > 0 and not torch.allclose(got, ref)


def test_stride_samples_positions(mp):
    k = 4
    g = _grads(_cfg(mtp_loss_stride=k), fused=True)
    for d in range(DEPTHS):  # depth d samples positions d % k, d % k + k, ...
        hit = g[1 + d].abs().sum((1, 2)) > 0
        pos = torch.arange(S, device=hit.device)
        expected = (pos % k) == (d % k)
        expected[-(d + 1):] = False  # the rolled-out tail has no label
        assert bool((hit == expected).all()), (d, hit, expected)


def _model(**kw):
    config = TransformerConfig(**{**dict(
        num_layers=4, hidden_size=64, num_attention_heads=4, num_query_groups=2, kv_channels=16,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=4, linear_num_value_heads=4,
        experimental_attention_variant="complex_kda", normalization="RMSNorm", add_bias_linear=False,
        gated_linear_unit=True, activation_func=F.silu, hidden_dropout=0.0, attention_dropout=0.0,
        num_moe_experts=4, moe_router_topk=2, moe_ffn_hidden_size=32, moe_grouped_gemm=False,
        moe_router_load_balancing_type="none", bf16=True, params_dtype=torch.bfloat16, is_hybrid_model=True,
        page_sparse_selector="dense", page_sparse_backend="gathered", mtp_num_layers=DEPTHS,
        mtp_latent=True, mtp_loss_scaling_factor=0.3,
    ), **kw})
    return HybridModel(config=config, hybrid_stack_spec=hybrid_stack_spec, vocab_size=128,
                       max_sequence_length=256, hybrid_layer_pattern="*ELE/L/L/L",
                       position_embedding_type="none", share_embeddings_and_output_weights=False).cuda()


def test_hybrid_latent_ckda_depths_train(mp):
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(0)
    model = _model()
    names = [n for n, _ in model.named_parameters()]
    assert not any("eh_proj" in n or "enorm" in n for n in names)  # latent: no next-token embedding path
    assert sum(".mtp." in n or n.startswith("mtp.") for n in names) > 0
    ids = torch.randint(0, 128, (2, 64), device="cuda")
    pos = torch.arange(64, device="cuda").expand(2, 64)
    with torch.autocast("cuda", dtype=torch.bfloat16):  # no Float16Module wrapper here
        loss = model(ids, position_ids=pos, attention_mask=None, labels=torch.roll(ids, -1, 1))
    loss.float().mean().backward()
    mtp_grads = [p.grad for n, p in model.named_parameters() if n.startswith("mtp.")]
    assert all(g is not None for g in mtp_grads) and sum(float(g.abs().sum()) for g in mtp_grads) > 0


def test_mtp_depths_plain_residual_under_depth_memory(mp):
    """With the decoder on Linear Depth Memory the MTP depths keep a plain residual: no depth-memory modules of
    their own (each would hold a full workspace outside autograd), and they still train."""
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(0)
    model = _model(depth_memory=True)
    mtp = [n for n, _ in model.named_parameters() if n.startswith("mtp.")]
    assert mtp and not any("ldm" in n or "depth_memory" in n for n in mtp), mtp
    assert any("ldm" in n or "depth_memory" in n for n, _ in model.named_parameters() if n.startswith("decoder."))
    ids = torch.randint(0, 128, (2, 64), device="cuda")
    pos = torch.arange(64, device="cuda").expand(2, 64)
    with torch.autocast("cuda", dtype=torch.bfloat16):  # no Float16Module wrapper here
        loss = model(ids, position_ids=pos, attention_mask=None, labels=torch.roll(ids, -1, 1))
    loss.float().mean().backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if n.startswith("mtp."))
