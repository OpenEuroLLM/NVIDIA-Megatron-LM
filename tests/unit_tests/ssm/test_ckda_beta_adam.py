# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""linear_beta_adam: complex_kda's beta projection (one scalar gate per value head) goes to the
scalar optimizer under a Muon-family optimizer; the alpha / output-gate projections (a vector
per head) and q / k / v / o stay on the matrix optimizer."""

import pytest
import torch

from megatron.core.optimizer.emerging_optimizers import _is_nonlinear_or_embedding
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


def _scalar_optimizer_params(**overrides):
    """Names of the 2-D parameters the Muon-family optimizers hand to Adam."""
    from megatron.core.ssm.complex_kda import ComplexKDA

    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    layer = ComplexKDA(_config(**overrides), layer_number=1).cuda()
    return {
        name
        for name, p in layer.named_parameters()
        if len(p.shape) == 2 and _is_nonlinear_or_embedding(p)
    }, {name for name, p in layer.named_parameters() if len(p.shape) == 2}


@pytest.mark.parametrize("te", [False, True])
def test_beta_projection_on_adam_with_the_flag(te):
    adam, matrices = _scalar_optimizer_params(linear_beta_adam=True, linear_te_projections=te)
    assert {n for n in adam if "b_proj" in n} == {n for n in matrices if "b_proj" in n} != set()
    # alpha (f_proj), the output gate (g_proj) and q / k / v / o stay on the matrix optimizer
    assert not any(k in n for n in adam for k in ("f_proj", "g_proj", "q_proj", "k_proj",
                                                  "v_proj", "o_proj"))


def test_beta_projection_on_the_matrix_optimizer_by_default():
    adam, matrices = _scalar_optimizer_params()
    assert not any("b_proj" in n for n in adam)
    assert any("b_proj" in n for n in matrices)
