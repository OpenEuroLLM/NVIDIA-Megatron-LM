# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""gdn_fused_out_norm: the GDN layer computes the same with the fused gated RMSNorm."""

import pytest
import torch

pytest.importorskip("fla")

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec  # noqa: E402
from megatron.core.models.hybrid.hybrid_model import HybridModel  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer import TransformerConfig  # noqa: E402
from megatron.core.transformer.enums import AttnBackend  # noqa: E402
from tests.unit_tests.test_utilities import Utils  # noqa: E402


@pytest.fixture(autouse=True)
def setup():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def _model(fused, zero_centered):
    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    config = TransformerConfig(
        num_layers=2, hidden_size=256, num_attention_heads=4, ffn_hidden_size=512,
        normalization="RMSNorm", add_bias_linear=False, gated_linear_unit=True,
        activation_func=torch.nn.functional.silu, attention_backend=AttnBackend.unfused,
        hidden_dropout=0.0, attention_dropout=0.0, is_hybrid_model=True, bf16=True,
        params_dtype=torch.bfloat16, linear_key_head_dim=64, linear_value_head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layernorm_zero_centered_gamma=zero_centered, gdn_fused_out_norm=fused,
    )
    return HybridModel(
        config=config, hybrid_stack_spec=hybrid_stack_spec, vocab_size=128,
        max_sequence_length=64, hybrid_layer_pattern="G-*-", position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    ).cuda().to(torch.bfloat16)


@pytest.mark.parametrize("zero_centered", [False, True])
def test_fused_out_norm_matches(zero_centered):
    tokens = torch.randint(0, 128, (2, 64), device="cuda")
    pos = torch.arange(64, device="cuda").unsqueeze(0).expand(2, -1)
    results = []
    for fused in (False, True):
        model = _model(fused, zero_centered)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "out_norm" in name:
                    p.add_(0.05 * torch.randn_like(p))
        logits = model(tokens, pos, attention_mask=None)
        logits.float().pow(2).mean().backward()
        results.append((logits.detach().float(),
                        {n: p.grad.float() for n, p in model.named_parameters() if p.grad is not None}))
    (ref, rg), (got, gg) = results
    torch.testing.assert_close(got, ref, rtol=3e-2, atol=3e-2)
    for n in rg:
        assert (gg[n] - rg[n]).norm() <= 5e-2 * rg[n].norm() + 1e-3, n
