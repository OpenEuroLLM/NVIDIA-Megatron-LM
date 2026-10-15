# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""MLA with grouped-query KV heads, NoPE, exclusive self attention and the output gate,
each optional, on a HybridModel (`+` layers)."""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MLATransformerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from tests.unit_tests.test_utilities import Utils

HEADS, QK, V = 4, 32, 32


@pytest.fixture(autouse=True)
def setup():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def _config(**kw):
    base = dict(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=HEADS,
        ffn_hidden_size=256,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_head_dim=QK,
        v_head_dim=V,
        qk_pos_emb_head_dim=16,
        rope_type="rope",
        normalization="RMSNorm",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        attention_backend=AttnBackend.unfused,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        is_hybrid_model=True,
    )
    base.update(kw)
    return MLATransformerConfig(**base)


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    return HybridModel(
        config=_config(**kw),
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=32,
        hybrid_layer_pattern="+-+-",
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    ).cuda()


def _mla_layers(model):
    return [l.self_attention for l in model.decoder.layers if hasattr(l, "self_attention")
            and isinstance(l.self_attention, MLASelfAttention)]


IDS = None


def _ids():
    global IDS
    if IDS is None:
        IDS = torch.randint(0, 128, (2, 16), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
    return IDS


@pytest.mark.parametrize("groups", [None, 2, 1])
@pytest.mark.parametrize("pos", [16, 0])
@pytest.mark.parametrize("xsa,gate", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("fused_down", [False, True])
def test_combinations_train(groups, pos, xsa, gate, fused_down):
    model = _model(
        mla_num_query_groups=groups,
        qk_pos_emb_head_dim=pos,
        exclusive_self_attention=xsa,
        attention_output_gate=gate,
        mla_down_proj_fusion=fused_down,
    )
    attn = _mla_layers(model)
    assert len(attn) == 2
    kv_heads = groups or HEADS
    assert attn[0].linear_kv_up_proj.weight.shape[0] == kv_heads * (QK + V)
    assert (attn[0].linear_out_gate is not None) == gate
    assert (attn[0].rotary_pos_emb is None) == (pos == 0)
    logits = model(_ids(), position_ids=None, attention_mask=None)
    assert torch.isfinite(logits).all()
    logits.float().logsumexp(-1).mean().backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


@pytest.mark.parametrize("pos", [16, 0])
def test_grouped_kv_heads_equal_repeated_full_heads(pos):
    """G key/value heads behave exactly as H heads whose up-projection repeats each
    of the G heads H/G times (the GQA head mapping)."""
    groups = 2
    gqa = _model(mla_num_query_groups=groups, qk_pos_emb_head_dim=pos)
    full = _model(qk_pos_emb_head_dim=pos)
    state = {}
    for name, value in gqa.state_dict().items():
        if name.endswith("linear_kv_up_proj.weight"):
            w = value.view(groups, QK + V, -1).repeat_interleave(HEADS // groups, dim=0)
            value = w.reshape(HEADS * (QK + V), -1)
        state[name] = value
    full.load_state_dict(state)
    with torch.no_grad():
        a = gqa(_ids(), position_ids=None, attention_mask=None)
        b = full(_ids(), position_ids=None, attention_mask=None)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("groups", [None, 2])
def test_xsa_output_is_orthogonal_to_own_values(groups):
    model = _model(mla_num_query_groups=groups, exclusive_self_attention=True)
    attn = _mla_layers(model)[0]
    seen = {}
    core = attn.core_attention.forward

    def capture_core(query, key, value, *args, **kwargs):
        seen["v"] = value.detach()
        return core(query, key, value, *args, **kwargs)

    attn.core_attention.forward = capture_core
    attn.linear_proj.register_forward_pre_hook(lambda m, a: seen.__setitem__("o", a[0].detach()))
    with torch.no_grad():
        model(_ids(), position_ids=None, attention_mask=None)
    v = seen["v"].float()  # [s, b, kv_heads, V]
    o = seen["o"].float().view(*v.shape[:2], HEADS, V)
    v = v.repeat_interleave(HEADS // v.shape[2], dim=2)
    # The coefficient of the own value left in the output; cosine would be 1 at the
    # first token, which attends only to itself and keeps an eps-sized multiple of v.
    coef = (o * v).sum(-1) / v.pow(2).sum(-1)
    assert coef.abs().max() < 1e-3


def test_zero_gate_halves_the_attention_output():
    gated = _model(attention_output_gate=True)
    plain = _model()
    plain.load_state_dict({k: v for k, v in gated.state_dict().items() if "out_gate" not in k})
    outs = {}
    for name, model in (("gated", gated), ("plain", plain)):
        attn = _mla_layers(model)[0]
        if name == "gated":
            with torch.no_grad():
                attn.linear_out_gate.weight.zero_()
        attn.linear_proj.register_forward_pre_hook(
            lambda m, a, name=name: outs.__setitem__(name, a[0].detach())
        )
        with torch.no_grad():
            model(_ids(), position_ids=None, attention_mask=None)
    torch.testing.assert_close(outs["gated"], 0.5 * outs["plain"], rtol=1e-4, atol=1e-5)


def test_nope_uses_plain_softmax_scale():
    attn = _mla_layers(_model(qk_pos_emb_head_dim=0, rope_type="yarn", rotary_scaling_factor=40.0,
                              mscale_all_dim=1.0))[0]
    assert attn.softmax_scale == pytest.approx(QK**-0.5)


def test_config_guards():
    with pytest.raises(NotImplementedError, match="cache_mla_latents"):
        _config(attention_output_gate=True, cache_mla_latents=True)
    with pytest.raises(ValueError, match="multiple of"):
        _config(mla_num_query_groups=3)
