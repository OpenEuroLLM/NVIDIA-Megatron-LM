# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""embedding_multiplier: a fixed scale on the word-embedding output without muP."""

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


def _embed(mult):
    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    cfg = TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=4,
                            embedding_multiplier=mult, hidden_dropout=0.0)
    emb = LanguageModelEmbedding(cfg, vocab_size=64, max_sequence_length=16,
                                 position_embedding_type="none").cuda()
    ids = torch.arange(16, device="cuda").reshape(2, 8) % 64
    return emb(ids, position_ids=None)


def test_embedding_multiplier_scales_the_output():
    base, scaled = _embed(1.0), _embed(50.0)
    torch.testing.assert_close(scaled, 50.0 * base)
