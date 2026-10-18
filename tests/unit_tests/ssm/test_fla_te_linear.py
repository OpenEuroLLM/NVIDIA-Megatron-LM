# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""linear_te_projections: the complex_kda mixer with Transformer Engine projections must be the
same model -- same parameter names, shapes and init -- and compute the same outputs and
gradients as with fla's nn.Linear projections."""

import pytest
import torch
from torch import nn

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN, HEAD_DIM, HEADS = 128, 32, 4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _config(**overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        kv_channels=HEAD_DIM,
        linear_key_head_dim=HEAD_DIM,
        linear_num_key_heads=HEADS,
        experimental_attention_variant="complex_kda",
        linear_attention_freq=3,
        linear_output_gate="lowrank",
        linear_gate_init_style="shipped",
        linear_drop_qkv_silu=True,
        linear_beta_max=2.0,
        gradient_accumulation_fusion=False,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@pytest.fixture(autouse=True)
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def _build(te: bool):
    from megatron.core.ssm.complex_kda import ComplexKDA

    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    return ComplexKDA(_config(linear_te_projections=te), layer_number=1).cuda()


def test_same_parameters_and_all_projections_swapped():
    from megatron.core.extensions.transformer_engine import TELinear

    plain, te = _build(False), _build(True)
    a, b = dict(plain.named_parameters()), dict(te.named_parameters())
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(b[name], a[name], rtol=0, atol=0, msg=name)
    assert not any(type(m) is nn.Linear for m in te.modules())
    n_linear = sum(type(m) is nn.Linear for m in plain.modules())
    assert n_linear > 0 and sum(isinstance(m, TELinear) for m in te.modules()) == n_linear


def test_same_outputs_and_gradients(monkeypatch):
    # TE's fp32 GEMMs run in TF32; the plain reference does the same (training is bf16)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    plain, te = _build(False), _build(True)
    x = torch.randn(64, 2, HIDDEN, device="cuda")
    outs, grads = [], []
    for model in (plain, te):
        xi = x.clone().requires_grad_(True)
        out, _ = model(xi)
        out.float().square().mean().backward()
        outs.append(out.detach())
        grads.append({n: p.grad.detach().clone() for n, p in model.named_parameters()
                      if p.grad is not None} | {"input": xi.grad.detach()})
    torch.testing.assert_close(outs[1], outs[0], rtol=1e-4, atol=1e-5)
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        torch.testing.assert_close(grads[1][name], grads[0][name], rtol=1e-3, atol=1e-5, msg=name)
