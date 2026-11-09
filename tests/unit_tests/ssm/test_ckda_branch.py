# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""ckda_branch: a second mixer next to a complex_kda mixer, ``alpha(t) ckda + m(t) gamma branch``.

The checkpoint surgery (autoexp scripts/korbi/production/add_ckda_branch.py) writes the branch's
tensors under the names and shapes checked here; at gamma = 0 the layer computes exactly what it
computed without the branch."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.ssm import ckda_branch as cb

HIDDEN, HEAD_DIM, HEADS = 128, 32, 4
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _sched(**kw):
    base = dict(ckda_branch_start_iteration=100, ckda_branch_alpha_hold=25, ckda_branch_alpha_ramp=175,
                ckda_branch_alpha_end=0.0, ckda_branch_m_start=4.0, ckda_branch_m_ramp=200,
                ckda_branch_gamma_scale=1.0, ckda_branch_log_interval=0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_schedule():
    c = _sched()
    assert cb.schedule(c, None) == (1.0, 4.0)  # no training loop: the insertion values
    assert cb.schedule(c, 100) == (1.0, 4.0)
    assert cb.schedule(c, 125) == (1.0, pytest.approx(4.0 - 3.0 * 25 / 200))
    alpha, m = cb.schedule(c, 125 + 175 // 2 + 1)
    assert 0.0 < alpha < 1.0 and 1.0 < m < 4.0
    assert cb.schedule(c, 300) == (0.0, 1.0)
    assert cb.schedule(c, 10_000) == (0.0, 1.0)
    assert cb.schedule(c, 50) == (1.0, 4.0)  # before the start: as at the start
    assert cb.schedule(_sched(ckda_branch_alpha_end=0.25), 10_000)[0] == 0.25


def test_branch_mix_gamma_zero_is_exact_and_gamma_scale_scales():
    torch.manual_seed(0)
    mix = cb.BranchMix(_sched(), layer_number=10)
    y, b = torch.randn(5, 2, 8), torch.randn(5, 2, 8)
    assert torch.equal(mix(y, b), y)  # alpha 1, gamma 0
    out = mix(y, b)
    out.sum().backward()
    assert mix.gamma.grad is not None and mix.gamma.grad.abs().item() > 0  # gamma learns from step 1
    with torch.no_grad():
        mix.gamma.fill_(0.1)
    scaled = cb.BranchMix(_sched(ckda_branch_gamma_scale=10.0), layer_number=10)
    with torch.no_grad():
        scaled.gamma.fill_(0.01)
    assert torch.allclose(mix(y, b), scaled(y, b))  # 0.1 * 1 = 0.01 * 10, m = 4 in both


def test_has_branch_uses_zero_based_layer_indices():
    c = SimpleNamespace(ckda_branch="attention", ckda_branch_layers=[9, 25])
    assert cb.has_branch(c, 10) and cb.has_branch(c, 26) and not cb.has_branch(c, 9)
    assert not cb.has_branch(SimpleNamespace(ckda_branch=None, ckda_branch_layers=[9]), 10)


# --------------------------------------------------------------------------------------------
# the layer (Transformer Engine, one GPU)
# --------------------------------------------------------------------------------------------


@pytest.fixture
def model_parallel():
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def _config(**overrides):
    from megatron.core.transformer.transformer_config import TransformerConfig

    kwargs = dict(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_query_groups=2,
        kv_channels=2 * HEAD_DIM,  # the attention's head dim: twice complex_kda's, as in the main run
        qk_layernorm=True,
        attention_output_gate=True,
        exclusive_self_attention=True,
        linear_key_head_dim=HEAD_DIM,
        linear_num_key_heads=HEADS,
        experimental_attention_variant="complex_kda",
        linear_attention_freq=3,
        linear_output_gate="lowrank",
        linear_gate_init_style="shipped",
        linear_drop_qkv_silu=True,
        linear_beta_max=2.0,
        linear_beta_adam=True,
        linear_fla_gate="signed_sigmoid2",
        linear_fla_allow_neg_eigval=True,
        linear_kronlstm_output_gate="lowrank",
        bf16=True,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
        ckda_branch_layers=[0],
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _layer(**overrides):
    from megatron.core.ssm.complex_kda import ComplexKDA
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    return ComplexKDA(_config(**overrides), layer_number=1).cuda()


def _x(seq=64, batch=2):
    torch.manual_seed(1)
    return torch.randn(seq, batch, HIDDEN, device="cuda", dtype=torch.bfloat16)


@gpu
@pytest.mark.parametrize("groups", [2, 4])
def test_attention_branch_names_shapes_and_exact_start(model_parallel, groups):
    layer = _layer(ckda_branch="attention", ckda_branch_num_query_groups=groups)
    shapes = {n: tuple(p.shape) for n, p in layer.named_parameters() if n.startswith(("branch.", "branch_mix."))}
    d = 2 * HEAD_DIM
    # what add_ckda_branch.py writes: per KV group [q heads | gate heads | k | v] x d rows
    assert shapes == {
        "branch.linear_qkv.weight": (groups * (2 * HEADS // groups + 2) * d, HIDDEN),
        "branch.linear_proj.weight": (HIDDEN, HEADS * d),
        "branch.q_layernorm.weight": (d,),
        "branch.k_layernorm.weight": (d,),
        "branch_mix.gamma": (1,),
    }, shapes
    assert layer.branch.linear_qkv.weight.headwise_dim == d
    x = _x()
    with torch.no_grad():
        with_branch = layer(x)[0]
        branch, layer.branch = layer.branch, None
        without = layer(x)[0]
        layer.branch = branch
    assert torch.equal(with_branch, without)  # gamma = 0: the model is unchanged
    with torch.no_grad():
        layer.branch_mix.gamma.fill_(0.5)
    out = layer(x)[0]
    assert not torch.equal(out.detach(), without)
    out.float().pow(2).mean().backward()
    assert layer.branch.linear_qkv.weight.grad.abs().sum() > 0
    assert layer.branch_mix.gamma.grad.abs().sum() > 0


@gpu
def test_kronlstm_branch_names_tags_and_exact_start(model_parallel):
    layer = _layer(ckda_branch="kronlstm")
    kron = layer.branch.layer
    names = {n for n, _ in layer.named_parameters() if n.startswith("branch.")}
    assert {"branch.layer.qkv_proj.weight", "branch.layer.qkv_conv1d.weight", "branch.layer.b_proj.weight",
            "branch.layer.c_proj.weight", "branch.layer.f_proj.0.weight", "branch.layer.f_proj.1.weight",
            "branch.layer.g_proj.0.weight", "branch.layer.g_proj.1.weight", "branch.layer.g_proj.1.bias",
            "branch.layer.A_log", "branch.layer.dt_bias", "branch.layer.o_norm.weight",
            "branch.layer.o_proj.weight"} == names, names
    assert tuple(kron.qkv_proj.weight.shape) == (3 * HEADS * HEAD_DIM, HIDDEN)
    # complex_kda's convolution without SiLU, its per-head q / k / v blocks, the gates on Adam
    assert kron.qkv_conv1d.activation is None
    assert kron.qkv_proj.weight.headwise_dim == HEAD_DIM
    assert kron.b_proj.weight.use_adam_with_emerging_optimizers
    assert kron.c_proj.weight.use_adam_with_emerging_optimizers
    assert kron.b_proj.bias is None  # neg_eigval_init_bias off: complex_kda's bias-free beta
    x = _x()
    with torch.no_grad():
        with_branch = layer(x)[0]
        branch, layer.branch = layer.branch, None
        without = layer(x)[0]
        layer.branch = branch
    assert torch.equal(with_branch, without)


@gpu
def test_stand_alone_kronlstm_keeps_its_silu(model_parallel):
    from megatron.core.ssm.fla_hybrid import KronLSTM

    kron = KronLSTM(_config(), layer_number=1)
    assert kron.layer.qkv_conv1d.activation == "silu"
