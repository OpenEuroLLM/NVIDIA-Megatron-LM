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
                ckda_branch_gamma_scale=1.0, ckda_branch_crossfade=False, ckda_branch_calibrate=False,
                ckda_branch_log_interval=0)
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


def test_crossfade_weights_add_up_to_one():
    c = _sched(ckda_branch_crossfade=True, ckda_branch_alpha_hold=0, ckda_branch_alpha_ramp=150)
    mix = cb.BranchMix(c, layer_number=10)
    y, b = torch.randn(5, 2, 8), torch.randn(5, 2, 8)
    assert torch.equal(mix(y, b), y)  # alpha 1 without a training loop: exactly the old layer
    import megatron.core.ssm.ckda_branch as mod

    for it, alpha in ((100, 1.0), (175, 0.5), (250, 0.0), (400, 0.0)):
        orig = mod.current_iteration
        mod.current_iteration = lambda it=it: it
        try:
            assert torch.allclose(mix(y, b), alpha * y + (1 - alpha) * b)  # gamma 0: weight 1 - alpha
            with torch.no_grad():
                mix.gamma.fill_(0.2)
            assert torch.allclose(mix(y, b), alpha * y + (1 - alpha) * 1.2 * b)
            with torch.no_grad():
                mix.gamma.zero_()
        finally:
            mod.current_iteration = orig


def test_calibration_matches_the_branch_to_the_mixers_rms_once():
    import megatron.core.ssm.ckda_branch as mod

    c = _sched(ckda_branch_crossfade=True, ckda_branch_alpha_hold=0, ckda_branch_alpha_ramp=150,
               ckda_branch_calibrate=True, pipeline_model_parallel_size=1)
    mix = cb.BranchMix(c, layer_number=10)
    assert "calib" in mix.state_dict() and float(mix.calib) == 0.0  # persistent: the surgery writes 0
    y, b = 3.0 * torch.randn(64, 2, 8), torch.randn(64, 2, 8)
    mix.eval()
    mix(y, b)
    assert float(mix.calib) == 0.0  # not in evaluation
    mix.train()
    assert torch.equal(mix(y, b), y)  # alpha 1 (no training loop): the output does not depend on calib
    want = (y.pow(2).sum() / b.pow(2).sum()).sqrt()
    assert float(mix.calib) == pytest.approx(float(want), rel=1e-6)
    mix(2 * y, b)  # measured once
    assert float(mix.calib) == pytest.approx(float(want), rel=1e-6)
    orig = mod.current_iteration
    mod.current_iteration = lambda: 250  # after the ramp: the branch alone, at the mixer's scale
    try:
        assert torch.allclose(mix(y, b), float(want) * b)
    finally:
        mod.current_iteration = orig
    # a loaded (nonzero) value is kept
    other = cb.BranchMix(c, layer_number=10)
    other.load_state_dict({"gamma": torch.zeros(1), "calib": torch.tensor([1.5])})
    other.train()
    other(y, b)
    assert float(other.calib) == 1.5


def test_current_iteration_takes_the_loaded_checkpoints_step_in_an_eval_only_run(monkeypatch):
    import megatron.training as mt

    args = SimpleNamespace(curr_iteration=0, iteration=23400)  # eval-only: the loop never ran
    monkeypatch.setattr(mt, "get_args", lambda: args, raising=False)
    assert cb.current_iteration() == 23400
    args.curr_iteration, args.iteration = 23450, 23300  # training: the loop's step
    assert cb.current_iteration() == 23450
    args.curr_iteration = None
    assert cb.current_iteration() == 23300


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
        add_bias_linear=False,
        normalization="RMSNorm",
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
    # bf16 throughout, as Float16Module holds the model
    return ComplexKDA(_config(**overrides), layer_number=1).cuda().bfloat16()


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


def test_kvshare_source_mapping():
    c = SimpleNamespace(ckda_branch="kvshare", ckda_branch_layers=[9, 25], ckda_branch_kv_source_layers=[0, 17])
    assert cb.kv_source(c, 10) == 0 and cb.kv_source(c, 26) == 17
    assert cb.publishes_kv(c, 1) and cb.publishes_kv(c, 18) and not cb.publishes_kv(c, 10)
    assert not cb.publishes_kv(SimpleNamespace(ckda_branch="attention", ckda_branch_kv_source_layers=[0]), 1)
    with pytest.raises(ValueError):  # one source per branch layer
        cb.kv_source(SimpleNamespace(ckda_branch_layers=[9, 25], ckda_branch_kv_source_layers=[0]), 10)
    with pytest.raises(ValueError):  # the source must come first
        cb.kv_source(SimpleNamespace(ckda_branch_layers=[9], ckda_branch_kv_source_layers=[12]), 10)


def _source_attention(config):
    """The hybrid's ``*`` attention as a kvshare source (layer index 0)."""
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TERowParallelLinear,
    )
    from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
    from megatron.core.transformer.enums import AttnMaskType

    torch.manual_seed(2)
    return SelfAttention(
        config,
        SelfAttentionSubmodules(linear_qkv=TEColumnParallelLinear, core_attention=TEDotProductAttention,
                                linear_proj=TERowParallelLinear),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    ).cuda().bfloat16()


@gpu
def test_kvshare_branch_is_attention_over_the_sources_keys_and_values(model_parallel):
    kw = dict(ckda_branch="kvshare", ckda_branch_layers=[1], ckda_branch_kv_source_layers=[0])
    from megatron.core.ssm.complex_kda import ComplexKDA
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    config = _config(**kw)
    layer = ComplexKDA(config, layer_number=2).cuda().bfloat16()
    src = _source_attention(config)
    d, g = 2 * HEAD_DIM, 2
    shapes = {n: tuple(p.shape) for n, p in layer.named_parameters() if n.startswith(("branch.", "branch_mix."))}
    # what add_ckda_branch.py writes: per KV group [q heads | gate heads] x d rows; no k / v, no k norm
    assert shapes == {
        "branch.linear_qkv.weight": (2 * HEADS * d, HIDDEN),
        "branch.linear_proj.weight": (HIDDEN, HEADS * d),
        "branch.q_layernorm.weight": (d,),
        "branch_mix.gamma": (1,),
    }, shapes
    assert layer.branch.kv_src == 0 and layer.branch.linear_qkv.weight.headwise_dim == d

    # reference: one ordinary attention layer with the branch's q / gate rows, the source's k / v rows
    # and k norm, the branch's q norm and output projection -- on the same input it is the same function
    ref = _source_attention(config)
    with torch.no_grad():
        bq = layer.branch.linear_qkv.weight.view(g, 2 * HEADS // g * d, HIDDEN)
        sk = src.linear_qkv.weight.view(g, (2 * HEADS // g + 2) * d, HIDDEN)[:, 2 * HEADS // g * d:]
        ref.linear_qkv.weight.copy_(torch.cat([bq, sk], 1).reshape(-1, HIDDEN))
        ref.q_layernorm.weight.copy_(layer.branch.q_layernorm.weight.add(0.25))
        layer.branch.q_layernorm.weight.add_(0.25)  # a non-trivial q norm on both
        ref.k_layernorm.weight.copy_(src.k_layernorm.weight)
        ref.linear_proj.weight.copy_(layer.branch.linear_proj.weight)
    cb.KV_STORE.clear()
    x = _x()
    with torch.no_grad():
        src(x, attention_mask=None)  # publishes layer 0's keys / values
        assert set(cb.KV_STORE) == {0}
        got = layer.branch(x, attention_mask=None)[0]
        want = ref(x, attention_mask=None)[0]
    torch.testing.assert_close(got, want, atol=2e-2, rtol=0)  # bf16, two kernels' rounding
    assert set(cb.KV_STORE) == {0}  # the branch does not publish

    # gamma = 0: the model is unchanged; then gradients reach the branch, never the source (detached)
    with torch.no_grad():
        with_branch = layer(x)[0]
        branch, layer.branch = layer.branch, None
        without = layer(x)[0]
        layer.branch = branch
    assert torch.equal(with_branch, without)
    with torch.no_grad():
        layer.branch_mix.gamma.fill_(0.5)
    src(x, attention_mask=None)[0].float().sum().mul(0).backward()  # graph-carrying source keys
    for p in src.parameters():
        p.grad = None
    layer(x)[0].float().pow(2).mean().backward()
    assert layer.branch.linear_qkv.weight.grad.abs().sum() > 0
    assert layer.branch.linear_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in src.parameters())
    cb.KV_STORE.clear()


@gpu
def test_kvshare_under_full_recompute_matches_no_recompute(model_parallel):
    """Source and branch layer in separate recompute segments (as in the main run, 2 layers each): the
    backward recomputes the branch's segment first, reading the forward's published keys / values."""
    from megatron.core.ssm.complex_kda import ComplexKDA
    from megatron.core.tensor_parallel.random import checkpoint, model_parallel_cuda_manual_seed

    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    # no dropout (as the main run): the two runs must draw the same function
    config = _config(ckda_branch="kvshare", ckda_branch_layers=[1], ckda_branch_kv_source_layers=[0],
                     attention_dropout=0.0, hidden_dropout=0.0)
    layer = ComplexKDA(config, layer_number=2).cuda().bfloat16()
    src = _source_attention(config)
    with torch.no_grad():
        layer.branch_mix.gamma.fill_(0.5)
    x = _x()

    def run(recompute):
        cb.KV_STORE.clear()
        for p in list(layer.parameters()) + list(src.parameters()):
            p.grad = None
        xs = x.clone().requires_grad_()
        if recompute:
            h = checkpoint(lambda t: src(t, attention_mask=None)[0], False, xs)
            out = checkpoint(lambda t: layer(t)[0], False, xs + h)
        else:
            h = src(xs, attention_mask=None)[0]
            out = layer(xs + h)[0]
        out.float().pow(2).mean().backward()
        grads = {n: p.grad.float().clone() for n, p in layer.named_parameters()
                 if n.startswith("branch") and p.grad is not None}
        grads.update({f"src.{n}": p.grad.float().clone() for n, p in src.named_parameters()})
        return out.detach().float(), xs.grad.float(), grads

    def rel(a, b):
        return float((a - b).norm() / b.norm().clamp_min(1e-30))

    # bf16 kernels may round differently between the no-autograd forward (checkpoint) and the autograd one;
    # stale or missing keys / values would be O(1)
    out0, gx0, g0 = run(False)
    out1, gx1, g1 = run(True)
    branch, layer.branch = layer.branch, None
    base0, basegx0, _ = run(False)
    base1, basegx1, _ = run(True)
    layer.branch = branch
    print(f"recompute vs not: out {rel(out1, out0):.2e} (no branch {rel(base1, base0):.2e}), "
          f"input grad {rel(gx1, gx0):.2e} (no branch {rel(basegx1, basegx0):.2e}), "
          + ", ".join(f"{n} {rel(g1[n], g0[n]):.2e}" for n in g0))
    assert rel(out1, out0) < 1e-2 and rel(gx1, gx0) < 3e-2
    assert g0.keys() == g1.keys()
    for n in g0:
        assert g0[n].abs().sum() > 0, n
        assert rel(g1[n], g0[n]) < 3e-2, (n, rel(g1[n], g0[n]))
    cb.KV_STORE.clear()
