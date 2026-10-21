# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Internal attention over the mHC streams (mhc_internal_attention, internal_attention.py)
through the hybrid recipe: the module itself, compute mode as a strict extension of the plain
residual, packed documents = separate documents (both modes, with and without full recompute),
and gradients into every parameter."""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.internal_attention import InternalAttention, IATStreamMixer
from tests.unit_tests.test_utilities import Utils

VOCAB = 128
LENGTHS = (40, 64, 24)
# IAT_PATTERN=LELE locally (TE attention needs the cluster)
import os

PATTERN = os.environ.get("IAT_PATTERN", "LELE*E")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _config(pattern=PATTERN, **kw):
    return TransformerConfig(**{**dict(
        num_layers=len(pattern),
        hidden_size=128,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        experimental_attention_variant="complex_kda",
        linear_beta_max=2.0,
        linear_output_gate="lowrank",
        linear_gate_init_style="shipped",
        linear_drop_qkv_silu=True,
        attention_output_gate=True,
        exclusive_self_attention=True,
        qk_layernorm=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=64,
        moe_shared_expert_intermediate_size=64,
        moe_grouped_gemm=True,
        moe_permute_fusion=True,
        moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="none",
        moe_router_score_function="sigmoid",
        mhc=True,
        mhc_expansion_rate=4,
        mhc_internal_attention="compute",
        is_hybrid_model=True,
    ), **kw})


def _model(pattern=PATTERN, **kw):
    return HybridModel(
        config=_config(pattern, **kw),
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=VOCAB,
        max_sequence_length=128,
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    ).cuda()


@pytest.fixture(autouse=True)
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def test_internal_attention_is_per_token_and_mixes_streams():
    torch.manual_seed(0)
    ia = InternalAttention(_config(), n=4).cuda()
    x = torch.randn(5, 2, 4, 128, device="cuda")
    y = ia(x)
    assert y.shape == x.shape
    # per token: changing one token leaves every other token unchanged
    x2 = x.clone()
    x2[3, 1] += 1.0
    d = (ia(x2) - y).abs().amax(dim=(-1, -2))
    assert d[3, 1] > 0 and d.sum() == d[3, 1]
    # a memory stream reaches stream 0
    x3 = x.clone()
    x3[..., 2, :] += 1.0
    assert (ia(x3)[..., 0, :] - y[..., 0, :]).abs().max() > 1e-3
    # zero output map: the identity
    with torch.no_grad():
        ia.w_o.zero_()
    torch.testing.assert_close(ia(x), x)


def test_mixer_and_mlp_sublayers():
    cfg = _config()
    mixer, mlp = IATStreamMixer(cfg, "mixer"), IATStreamMixer(cfg, "mlp")
    assert mixer.iattn is not None and mlp.iattn is None and mixer.mhc is None
    s = torch.randn(3, 1, 4, 128)
    torch.testing.assert_close(mlp.read(s), s[..., 0, :])
    out = torch.randn(3, 1, 128)
    w = mlp.write(s, out)
    torch.testing.assert_close(w[..., 0, :], s[..., 0, :] + out)
    torch.testing.assert_close(w[..., 1:, :], s[..., 1:, :])
    assert IATStreamMixer(_config(mhc_internal_attention="mhc"), "mixer").mhc is not None
    # full: the mixer reads a latent of all n h dims and writes back to all of them
    full = IATStreamMixer(_config(mhc_internal_attention="full"), "mixer")
    assert full.down.shape == (128, 512) and full.up.shape == (512, 128)
    assert IATStreamMixer(_config(mhc_internal_attention="full"), "mlp").down is None
    x = torch.zeros_like(s)
    x[..., 3, :] = 1.0
    assert full.read(x).abs().max() > 0  # a memory stream reaches the mixer directly
    with torch.no_grad():
        full.iattn.w_o.zero_()
    w = full.write(s, out)
    assert (w - s)[..., 1:, :].abs().max() > 0  # and its output reaches the memory


def test_compute_mode_extends_the_plain_residual():
    """With the internal attention's output map at zero, compute mode is the plain residual
    model exactly (memory streams stay zero, the head reads stream 0)."""
    torch.manual_seed(1)
    plain = _model(mhc=False, mhc_internal_attention="none").float().eval()
    iat = _model().float().eval()
    missing, unexpected = iat.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all("iattn" in k for k in missing), (missing, unexpected)
    with torch.no_grad():
        for name, p in iat.named_parameters():
            if name.endswith("iattn.w_o"):
                p.zero_()
    ids = torch.randint(0, VOCAB, (1, 48), device="cuda")
    pos = torch.arange(48, device="cuda").unsqueeze(0)
    with torch.no_grad():
        a = plain(ids, position_ids=pos, attention_mask=None)
        b = iat(ids, position_ids=pos, attention_mask=None)
    torch.testing.assert_close(b, a, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("mode", ["compute", "full", "mhc"])
@pytest.mark.parametrize(
    "recompute",
    [{}, dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)],
    ids=["no_recompute", "full_recompute"],
)
def test_packed_documents_match_separate(mode, recompute):
    model = _model(mhc_internal_attention=mode, **recompute).float()
    model.train()
    docs = [torch.randint(0, VOCAB, (1, n), device="cuda") for n in LENGTHS]

    def run(ids, packed=None):
        pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
        return model(ids, position_ids=pos, attention_mask=None, packed_seq_params=packed)

    alone = [run(d) for d in docs]
    cu = torch.tensor([0, *torch.tensor(LENGTHS).cumsum(0).tolist()], dtype=torch.int32,
                      device="cuda")
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                             max_seqlen_q=max(LENGTHS), max_seqlen_kv=max(LENGTHS))
    together = run(torch.cat(docs, dim=1), packed)
    ref = torch.cat(alone, dim=1)
    scale = ref.std()
    err = (together - ref).abs().max() / scale
    assert err < 1e-2, float(err)
    leaked = (run(torch.cat(docs, dim=1)) - ref)[:, LENGTHS[0]:].abs().max() / scale
    assert leaked > 4 * err, (float(err), float(leaked))

    together.logsumexp(-1).mean().backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and "expert_bias" not in n
               and p.grad is None and getattr(p, "main_grad", None) is None]
    assert not missing, missing
    # the memory streams learn: the internal attention's maps get gradient
    for n, p in model.named_parameters():
        if "iattn.w_" in n or "iattn.pos" in n:
            assert p.grad.abs().sum() > 0, n


def test_config_validation():
    with pytest.raises(ValueError):
        _config(mhc_internal_attention="bogus")
    with pytest.raises(ValueError):
        _config(mhc=False)
    with pytest.raises(ValueError):
        _config(mhc_routed_topk=2)
