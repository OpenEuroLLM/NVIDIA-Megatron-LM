# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Exclusive self attention for the linear mixers (linear_xsa)."""

import pytest
import torch

pytest.importorskip("fla")

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec  # noqa: E402
from megatron.core.models.hybrid.hybrid_model import HybridModel  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer import TransformerConfig  # noqa: E402
from megatron.core.transformer.enums import AttnBackend  # noqa: E402
from tests.unit_tests.test_utilities import Utils  # noqa: E402

VARIANTS = {
    "triadic": dict(linear_triadic_base="kda", linear_triadic_e=2),
    "gated_delta_net": {},
    "kda": {},
    "signed_kda": {},
    "complex_kda": {},
    "mlstm": {},
}


@pytest.fixture(autouse=True)
def setup():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def _model(variant, xsa):
    import inspect

    from fla.layers import KimiDeltaAttention

    if variant == "mlstm":
        pytest.importorskip("mlstm_kernels")
    if variant not in ("triadic", "gated_delta_net", "mlstm") and \
            "xsa" not in inspect.signature(KimiDeltaAttention.__init__).parameters:
        pytest.skip("the fla on the path has no xsa option (ComplexKDA feat/hybrid-kernels)")
    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    config = TransformerConfig(
        num_layers=4, hidden_size=128, num_attention_heads=4, ffn_hidden_size=256,
        normalization="RMSNorm", add_bias_linear=False, gated_linear_unit=True,
        activation_func=torch.nn.functional.silu, attention_backend=AttnBackend.unfused,
        hidden_dropout=0.0, attention_dropout=0.0, is_hybrid_model=True, bf16=True,
        params_dtype=torch.bfloat16, experimental_attention_variant=variant,
        linear_key_head_dim=64, linear_value_head_dim=64, linear_num_key_heads=2,
        linear_num_value_heads=2, linear_xsa=xsa, **VARIANTS[variant],
    )
    return HybridModel(
        config=config, hybrid_stack_spec=hybrid_stack_spec, vocab_size=128,
        max_sequence_length=64, hybrid_layer_pattern="L-*-", position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    ).cuda().to(torch.bfloat16)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_linear_xsa_removes_the_self_component(variant, monkeypatch):
    import megatron.core.fusions.fused_xsa as fx

    calls = []
    orig = fx.exclusive_self_attention

    def spy(o, v, gate=None, eps=fx.EPS):
        out = orig(o, v, gate, eps)
        group = o.shape[1] // v.shape[1]
        vv = v.float().repeat_interleave(group, dim=1)
        calls.append(((out.float() * vv).sum(-1) / vv.pow(2).sum(-1).clamp(min=1e-6)).abs().max())
        return out

    monkeypatch.setattr(fx, "exclusive_self_attention", spy)
    try:
        import fla.modules.xsa as fla_xsa

        monkeypatch.setattr(fla_xsa, "exclusive_self_attention", spy)
    except ImportError:
        pass
    model = _model(variant, xsa=True)
    ids = torch.randint(0, 128, (2, 64), device="cuda")
    logits = model(ids, position_ids=None, attention_mask=None)
    logits.float().logsumexp(-1).mean().backward()
    assert calls, "XSA did not run"
    assert max(calls) < 2e-2
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing
