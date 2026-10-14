# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Grouped-query mode of the mLSTM layer.

A GQA layer (num_key_heads < num_value_heads) must compute exactly what an
MHA layer computes when each q/k head block is duplicated across its group of
value heads. The cell is swapped for a plain-torch causal function so the test
runs without mlstm_kernels: the property under test is the head layout around
the cell, not the cell itself.
"""

import pytest
import torch
import torch.nn.functional as F

import megatron.core.ssm.mlstm as mlstm_mod
from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_mlstm_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN = 128
DQK = 32
DV = 32
VALUE_HEADS = 4


def _toy_cell(q, k, v, i, f, return_last_states=False, chunk_size=64):
    """Causal gated linear attention: depends on every head of q, k, v, i, f."""
    s = q.shape[2]
    scores = torch.einsum("bhsd,bhtd->bhst", q.float(), k.float())
    gate = torch.sigmoid(i)[:, :, None, :] * torch.sigmoid(f)[:, :, :, None]
    mask = torch.ones(s, s, device=q.device, dtype=torch.bool).tril()
    scores = (scores * gate).masked_fill(~mask, 0.0)
    return torch.einsum("bhst,bhtd->bhsd", scores, v.float()).to(v.dtype)


def _config(num_key_heads):
    return TransformerConfig(
        hidden_size=HIDDEN,
        num_layers=1,
        num_attention_heads=VALUE_HEADS,
        linear_key_head_dim=DQK,
        linear_value_head_dim=DV,
        linear_num_key_heads=num_key_heads,
        linear_num_value_heads=VALUE_HEADS,
        linear_attention_freq=[1],
        experimental_attention_variant="mlstm",
        normalization="RMSNorm",
        activation_func=F.silu,
        use_cpu_initialization=True,
        bf16=False,
        params_dtype=torch.float32,
        transformer_impl="transformer_engine",
    )


def _build(num_key_heads, pg):
    cfg = _config(num_key_heads)
    spec = get_mlstm_module_spec(cfg)
    return mlstm_mod.MLSTM(cfg, submodules=spec.submodules, layer_number=1, pg_collection=pg).cuda()


@pytest.fixture
def pg(monkeypatch):
    monkeypatch.setattr(mlstm_mod, "HAVE_MLSTM_KERNELS", True)
    monkeypatch.setattr(mlstm_mod, "get_mlstm_kernel", lambda name: _toy_cell, raising=False)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    yield ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("num_key_heads", [1, 2])
def test_gqa_matches_duplicated_mha(pg, num_key_heads):
    gqa = _build(num_key_heads, pg)
    mha = _build(VALUE_HEADS, pg)
    rep = VALUE_HEADS // num_key_heads

    # Fused in_proj rows: [q | k | v | o | i | f]; q/k heads repeat_interleave'd.
    w = gqa.in_proj.weight.data
    qk = DQK * num_key_heads
    q, k, rest = w[:qk], w[qk : 2 * qk], w[2 * qk :]
    expand = lambda t: t.view(num_key_heads, DQK, HIDDEN).repeat_interleave(rep, 0).reshape(-1, HIDDEN)
    with torch.no_grad():
        mha.in_proj.weight.copy_(torch.cat([expand(q), expand(k), rest]))
        for name, p in gqa.named_parameters():
            if name != "in_proj.weight":
                dict(mha.named_parameters())[name].copy_(p)

    # Gate rows are zero-initialised; randomise them so i/f actually vary.
    with torch.no_grad():
        g = torch.randn(2 * VALUE_HEADS, HIDDEN, device=w.device) * 0.1
        gqa.in_proj.weight[-2 * VALUE_HEADS :] = g
        mha.in_proj.weight[-2 * VALUE_HEADS :] = g

    x = torch.randn(16, 2, HIDDEN, device="cuda")
    out_gqa, _ = gqa(x, None)
    out_mha, _ = mha(x, None)
    torch.testing.assert_close(out_gqa, out_mha, rtol=1e-5, atol=1e-5)


def test_gqa_param_count(pg):
    """q/k projections shrink by the group factor; everything else is unchanged."""
    gqa = _build(2, pg)
    mha = _build(VALUE_HEADS, pg)
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(mha) - n(gqa) == 2 * HIDDEN * DQK * (VALUE_HEADS - 2)


def test_rejects_non_divisible_heads():
    with pytest.raises(AssertionError, match="multiple of"):
        _config(3)
