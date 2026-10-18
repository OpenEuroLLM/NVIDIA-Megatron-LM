# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""mHC (dense and routed) on a HybridModel with full activation recompute.

Dense mHC: the widened [s, b, n, h] residual goes through the checkpointed layer chunks like
the plain residual. Routed streams: each sublayer checkpoints its body between the routed read
and the in-place write, so no n-stream state is saved per layer. Either way full recompute
must give the same loss and gradients as no recompute.
"""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from tests.unit_tests.test_utilities import Utils

PATTERN = "*-C-*-C-"
VOCAB = 128

SCHEMES = {
    "mhc": dict(mhc=True, mhc_expansion_rate=4, mhc_sinkhorn_iters=4),
    "mhc_routed": dict(mhc=True, mhc_expansion_rate=4, mhc_routed_topk=2),
    "mhc_routed8_keys": dict(
        mhc=True, mhc_expansion_rate=8, mhc_routed_topk=2, mhc_routed_key_dim=8
    ),
    "mhc_routed8_reversible": dict(
        mhc=True, mhc_expansion_rate=8, mhc_routed_topk=2, mhc_routed_reversible=True
    ),
    "mhc_routed8_mix": dict(
        mhc=True, mhc_expansion_rate=8, mhc_routed_topk=2, mhc_routed_mix=True
    ),
}
ROUTED = [k for k in SCHEMES if "routed" in k]
FULL = dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)


def _model(**kw):
    config = TransformerConfig(
        num_layers=len(PATTERN),
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        normalization="RMSNorm",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        attention_backend=AttnBackend.unfused,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        is_hybrid_model=True,
        **kw,
    )
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=VOCAB,
        max_sequence_length=32,
        hybrid_layer_pattern=PATTERN,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    ).cuda()


@pytest.fixture(autouse=True)
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def _loss_and_grads(model, ids):
    model.train()
    model.zero_grad(set_to_none=True)
    loss = model(ids, position_ids=None, attention_mask=None).float().logsumexp(-1).mean()
    loss.backward()
    return loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters()
                           if p.grad is not None}


@pytest.mark.parametrize("name", list(SCHEMES))
def test_full_recompute_matches_no_recompute(name):
    plain = _model(**SCHEMES[name])
    recomputed = _model(**SCHEMES[name], **FULL)
    recomputed.load_state_dict(plain.state_dict())
    ids = torch.randint(0, VOCAB, (2, 24), device="cuda")
    loss_a, grads_a = _loss_and_grads(plain, ids)
    loss_b, grads_b = _loss_and_grads(recomputed, ids)
    torch.testing.assert_close(loss_b, loss_a)
    assert grads_a.keys() == grads_b.keys()
    for n in grads_a:
        torch.testing.assert_close(grads_b[n], grads_a[n], msg=n)
    # every mixer parameter took part
    assert any("mhc" in n for n in grads_b)


@pytest.mark.parametrize("name", ROUTED)
def test_routed_full_recompute_saves_no_stream_state(name):
    """Under full recompute the routed model saves nothing n streams wide in its forward
    (sublayer recompute; the reversible mode's one stream state per stage is held by its
    write, not saved per layer)."""
    n = SCHEMES[name]["mhc_expansion_rate"]
    ids = torch.randint(0, VOCAB, (2, 24), device="cuda")

    def wide_saved(model):
        model.train()
        wide = []

        def pack(t):
            if t.dim() == 4 and t.shape[2] == n:
                wide.append(tuple(t.shape))
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            loss = model(ids, position_ids=None, attention_mask=None).float().logsumexp(-1).mean()
        loss.backward()
        return wide

    recomputed = _model(**SCHEMES[name], **FULL)
    assert wide_saved(recomputed) == []
    # the check has teeth: dense mHC under (layer-chunk) full recompute saves its n-stream
    # chunk inputs
    if name == ROUTED[0]:
        dense = _model(**SCHEMES["mhc"], **FULL)
        n = SCHEMES["mhc"]["mhc_expansion_rate"]
        assert wide_saved(dense)


def test_attn_res_still_rejects_full_recompute():
    model = _model(attn_res=True, attn_res_num_blocks=2, **FULL)
    model.train()
    ids = torch.randint(0, VOCAB, (2, 24), device="cuda")
    with pytest.raises(NotImplementedError, match="Block AttnRes does not support full"):
        model(ids, position_ids=None, attention_mask=None)
