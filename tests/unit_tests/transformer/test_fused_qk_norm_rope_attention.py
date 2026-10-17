# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""config.fused_qk_norm_rope: the attention layer computes the same as the unfused path."""

import pytest
import torch

from megatron.core.fusions.fused_qk_norm_rope import HAVE_TRITON
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not HAVE_TRITON, reason="needs Triton")


@pytest.fixture(autouse=True)
def setup():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def _model(
    fused, groups=2, rotary_percent=1.0, zero_centered=False, gate=False, pos_emb="rope",
    interleaved=False,
):
    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    config = TransformerConfig(
        num_layers=2, hidden_size=256, num_attention_heads=4, num_query_groups=groups,
        kv_channels=64, ffn_hidden_size=512, normalization="RMSNorm", add_bias_linear=False,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu, qk_layernorm=True,
        layernorm_zero_centered_gamma=zero_centered, attention_backend=AttnBackend.unfused,
        hidden_dropout=0.0, attention_dropout=0.0, is_hybrid_model=True, bf16=True,
        params_dtype=torch.bfloat16, fused_qk_norm_rope=fused, attention_output_gate=gate,
        rotary_interleaved=interleaved,
    )
    return HybridModel(
        config=config, hybrid_stack_spec=hybrid_stack_spec, vocab_size=128,
        max_sequence_length=64, hybrid_layer_pattern="*-*-", position_embedding_type=pos_emb,
        rotary_percent=rotary_percent, share_embeddings_and_output_weights=False,
    ).cuda().to(torch.bfloat16)


@pytest.mark.parametrize(
    "groups,rotary_percent,zero_centered,gate,pos_emb,interleaved",
    [
        (2, 1.0, False, False, "rope", False),
        (4, 0.5, True, False, "rope", False),
        (1, 1.0, False, False, "rope", False),
        (2, 1.0, False, True, "rope", False),
        # NoPE: q / k norm only; RoPE-only requirements (layout, mscale) do not apply
        (2, 1.0, False, False, "none", False),
        (4, 1.0, True, True, "none", True),
    ],
)
def test_fused_matches_unfused(groups, rotary_percent, zero_centered, gate, pos_emb, interleaved):
    tokens = torch.randint(0, 128, (2, 64), device="cuda")
    pos = torch.arange(64, device="cuda").unsqueeze(0).expand(2, -1)
    results = []
    for fused in (False, True):
        model = _model(fused, groups, rotary_percent, zero_centered, gate, pos_emb, interleaved)
        with torch.no_grad():  # a non-trivial norm weight
            for name, p in model.named_parameters():
                if "layernorm" in name and p.dim() == 1:
                    p.add_(0.05 * torch.randn_like(p))
        logits = model(tokens, pos, attention_mask=None)
        logits.float().pow(2).mean().backward()
        results.append((logits.detach().float(),
                        {n: p.grad.float() for n, p in model.named_parameters() if p.grad is not None}))
    (ref, ref_grads), (got, got_grads) = results
    torch.testing.assert_close(got, ref, rtol=3e-2, atol=3e-2)
    assert ref_grads.keys() == got_grads.keys()
    for name in ref_grads:
        a, b = got_grads[name], ref_grads[name]
        assert (a - b).norm() <= 5e-2 * b.norm() + 1e-3, name
