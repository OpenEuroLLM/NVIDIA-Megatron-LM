# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Exclusive self attention: the fused Triton kernel and its use in attention."""

import pytest
import torch

from megatron.core.fusions.fused_xsa import HAVE_TRITON, exclusive_self_attention, xsa_reference
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not (HAVE_TRITON and torch.cuda.is_available()), reason="needs Triton and a GPU"
)


@pytest.mark.parametrize("hq,hkv,dim", [(4, 4, 64), (8, 2, 64), (6, 1, 96), (2, 2, 128)])
@pytest.mark.parametrize("with_gate", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kernel_matches_reference(hq, hkv, dim, with_gate, dtype):
    gen = torch.Generator(device="cuda").manual_seed(0)
    T = 37
    o = torch.randn(T, hq, dim, device="cuda", generator=gen).to(dtype).requires_grad_()
    v = torch.randn(T, hkv, dim, device="cuda", generator=gen).to(dtype).requires_grad_()
    g = (
        torch.randn(T, hq, dim, device="cuda", generator=gen).to(dtype).requires_grad_()
        if with_gate
        else None
    )
    dy = torch.randn(T, hq, dim, device="cuda", generator=gen).to(dtype)
    outs = []
    for fn in (exclusive_self_attention, xsa_reference):
        for t in (o, v, g):
            if t is not None:
                t.grad = None
        y = fn(o, v, g)
        y.backward(dy)
        outs.append([y.detach()] + [t.grad.clone() for t in (o, v, g) if t is not None])
    tol = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=2e-2, atol=5e-2)
    for a, b in zip(*outs):
        torch.testing.assert_close(a.float(), b.float(), **tol)


def test_output_is_orthogonal_to_own_value():
    o = torch.randn(10, 4, 32, device="cuda")
    v = torch.randn(10, 2, 32, device="cuda")
    y = exclusive_self_attention(o, v)
    dots = (y * v.repeat_interleave(2, dim=1)).sum(-1)
    assert dots.abs().max() < 1e-4


@pytest.mark.parametrize("gated", [False, True])
def test_model_with_xsa_trains(gated):
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(1)
        config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            num_query_groups=2,
            ffn_hidden_size=128,
            normalization="RMSNorm",
            add_bias_linear=False,
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            attention_backend=AttnBackend.unfused,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            is_hybrid_model=True,
            exclusive_self_attention=True,
            attention_output_gate=gated,
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=64,
            hybrid_layer_pattern="*-*-",
            position_embedding_type="rope",
            share_embeddings_and_output_weights=False,
        ).cuda()
        ids = torch.randint(0, 128, (2, 32), device="cuda")
        loss = model(ids, position_ids=None, attention_mask=None).float().logsumexp(-1).mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    finally:
        Utils.destroy_model_parallel()
