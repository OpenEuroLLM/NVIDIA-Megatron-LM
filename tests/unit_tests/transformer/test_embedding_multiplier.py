# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""embedding_multiplier: a fixed scale on the word-embedding output without muP; embedding_norm:
an RMSNorm on it with a learnable gain."""

import pytest
import torch

from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


@pytest.fixture(autouse=True)
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def _module(mult=1.0, **overrides):
    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    cfg = TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=4,
                            embedding_multiplier=mult, hidden_dropout=0.0, **overrides)
    return LanguageModelEmbedding(cfg, vocab_size=64, max_sequence_length=16,
                                  position_embedding_type="none").cuda()


def _ids():
    return torch.arange(16, device="cuda").reshape(2, 8) % 64


def _embed(mult, **overrides):
    return _module(mult, **overrides)(_ids(), position_ids=None)


def test_embedding_multiplier_scales_the_output():
    base, scaled = _embed(1.0), _embed(50.0)
    torch.testing.assert_close(scaled, 50.0 * base)


def test_embedding_norm_fixes_the_token_rms():
    emb = _module(embedding_norm=True)
    with torch.no_grad():
        emb.word_embeddings.weight.mul_(10 + 100 * torch.rand(64, 1, device="cuda"))  # any row scale
        emb.embedding_norm_weight.fill_(0.5)
    out = emb(_ids(), position_ids=None)
    torch.testing.assert_close(out.pow(2).mean(-1).sqrt(), torch.full(out.shape[:2], 0.5,
                                                                     device="cuda"),
                               rtol=1e-4, atol=1e-4)
    # invariant to the embedding scale, so to embedding_multiplier as well (up to eps: rows at
    # init std 0.02 have rms^2 4e-4, so the default eps 1e-5 would cost ~1 % here)
    kw = dict(embedding_norm=True, layernorm_epsilon=1e-12)
    torch.testing.assert_close(_embed(50.0, **kw), _embed(1.0, **kw))


def test_embedding_norm_gain_is_a_trained_1d_parameter():
    emb = _module(embedding_norm=True)
    names = dict(emb.named_parameters())
    assert names["embedding_norm_weight"].shape == (32,)  # 1-D: Adam, no weight decay
    emb(_ids(), position_ids=None).sum().backward()
    assert emb.embedding_norm_weight.grad is not None
    assert emb.word_embeddings.weight.grad.abs().sum() > 0


def test_no_embedding_norm_by_default():
    assert "embedding_norm_weight" not in dict(_module().named_parameters())
