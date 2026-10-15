# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Per-layer embeddings (PLE, Gemma 3n/4) and value embeddings in a HybridModel."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.token_embeddings import PerLayerEmbedding, ValueEmbedding
from tests.unit_tests.test_utilities import Utils

VOCAB = 128


@pytest.fixture(autouse=True)
def setup():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def _model(pattern="*-*-", seed=0, **kw):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    config = TransformerConfig(
        num_layers=len(pattern),
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
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
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    ).cuda()


def _ids():
    return torch.randint(0, VOCAB, (2, 16), device="cuda", generator=torch.Generator("cuda").manual_seed(5))


def _logits(model):
    with torch.no_grad():
        return model(_ids(), position_ids=None, attention_mask=None)


@pytest.mark.parametrize("which,count", [("mlp", 2), ("all", 4)])
def test_ple_placement(which, count):
    model = _model(per_layer_embedding_dim=16, per_layer_embedding_layers=which)
    ples = [m for m in model.decoder.per_layer_embeddings if m is not None]
    assert len(ples) == count and all(isinstance(m, PerLayerEmbedding) for m in ples)


def test_ple_with_zero_output_projection_is_the_plain_model():
    ple = _model(per_layer_embedding_dim=16)
    plain = _model()
    plain.load_state_dict({k: v for k, v in ple.state_dict().items() if "per_layer" not in k})
    with torch.no_grad():
        for m in ple.decoder.per_layer_embeddings:
            if m is not None:
                m.up.weight.zero_()
    torch.testing.assert_close(_logits(ple), _logits(plain))


def test_value_embedding_with_zero_lambda_is_the_plain_model():
    ve = _model(value_embedding=True)
    plain = _model()
    plain.load_state_dict({k: v for k, v in ve.state_dict().items() if "value_embedding" not in k})
    tables = [m for m in ve.modules() if isinstance(m, ValueEmbedding)]
    assert len(tables) == 2
    with torch.no_grad():
        for t in tables:
            t.lambdas.copy_(torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(_logits(ve), _logits(plain))
    with torch.no_grad():
        tables[0].lambdas.copy_(torch.tensor([1.0, 1.0]))
    assert (_logits(ve) - _logits(plain)).abs().max() > 1e-4


@pytest.mark.parametrize("kw", [
    dict(per_layer_embedding_dim=16),
    dict(value_embedding=True),
    dict(per_layer_embedding_dim=16, per_layer_embedding_layers="all", value_embedding=True,
         engram_ngram_orders=[2, 3], engram_num_heads=2, engram_table_size=500, engram_embed_dim=8),
])
def test_trains_and_reaches_the_tables(kw):
    pattern = "N*-*-" if "engram_ngram_orders" in kw else "*-*-"
    model = _model(pattern, **kw)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    ids = torch.randint(0, VOCAB, (4, 16), device="cuda")
    losses = []
    for _ in range(5):
        logits = model(ids[:, :-1], position_ids=None, attention_mask=None)
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, VOCAB), ids[:, 1:].reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    tables = [p for n, p in model.named_parameters() if n.endswith(("table.weight", "value_embedding.weight"))]
    assert tables and all(p.grad is not None and p.grad.abs().sum() > 0 for p in tables)


def test_value_table_checkpoint_declares_the_head_axis():
    model = _model(value_embedding=True)
    table = next(m for m in model.modules() if isinstance(m, ValueEmbedding))
    sd = table.sharded_state_dict(prefix="ve.")
    tp = parallel_state.get_tensor_model_parallel_world_size()
    assert sd["ve.weight"].global_shape == (VOCAB, table.weight.shape[1] * tp)


def test_ple_rejects_depth_schemes():
    with pytest.raises(NotImplementedError, match="per-layer embeddings"):
        _model(per_layer_embedding_dim=16, mhc=True, mhc_expansion_rate=2)
