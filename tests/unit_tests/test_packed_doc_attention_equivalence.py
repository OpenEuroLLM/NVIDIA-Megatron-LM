# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""OELLM PATCH: does the cu_seqlens kernel mask EXACTLY what the dense mask means?

The runtime A/B arms only ever showed that turning --packed-doc-attention on
CHANGES the loss. That is a presence test: it proves masking happens somewhere,
not that it happens at the right document boundaries. test_packed_doc_attention.py
covers the boundary arithmetic, but purely as CPU tensor math -- it never reaches
an attention kernel.

This closes the gap. Same q/k/v, three ways:

    causal   TE, attn_mask_type=causal          -- no document masking at all
    packed   TE, qkv_format='thd' + cu_seqlens  -- what the patch actually runs
    dense    mcore DotProductAttention with the block-diagonal [b,1,s,s] mask,
             i.e. the semantics --reset-attention-mask was always *supposed* to
             have, computed by an independent unfused implementation

`packed` must match `dense` and must NOT match `causal`. The second assertion is
the positive control: without it a test that trivially passes proves nothing,
because a build where cu_seqlens was silently ignored would still satisfy the
first comparison against a wrongly-computed reference.
"""

import os
import sys

import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

# Two documents inside one 16-token pack. Deliberately unequal and not a power of
# two, so an off-by-one in the boundary lands somewhere visible.
DOC_LENGTHS = [7, 9]
SEQ_LENGTH = sum(DOC_LENGTHS)
NUM_HEADS = 4
HEAD_DIM = 32


def _config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=NUM_HEADS * HEAD_DIM,
        num_attention_heads=NUM_HEADS,
        kv_channels=HEAD_DIM,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )


def _document_mask():
    """Block-diagonal causal mask, True == masked out (Megatron's convention)."""
    segment = torch.zeros(SEQ_LENGTH, dtype=torch.long)
    offset = 0
    for doc_id, length in enumerate(DOC_LENGTHS):
        segment[offset : offset + length] = doc_id
        offset += length
    causal = torch.tril(torch.ones(SEQ_LENGTH, SEQ_LENGTH, dtype=torch.bool))
    same_document = segment[:, None] == segment[None, :]
    return (~(causal & same_document)).view(1, 1, SEQ_LENGTH, SEQ_LENGTH).cuda()


def _cu_seqlens():
    bounds = torch.tensor([0, *torch.tensor(DOC_LENGTHS).cumsum(0).tolist()])
    return bounds.to(dtype=torch.int32, device="cuda")


@pytest.mark.skipif(not HAVE_TE, reason="needs Transformer Engine")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cu_seqlens_matches_dense_mask_through_the_kernel():
    Utils.initialize_model_parallel(1, 1)
    # mcore's DotProductAttention forks the model-parallel RNG tracker for dropout,
    # which raises "cuda rng state model-parallel-rng is not added" unless seeded --
    # even at attention_dropout=0.0, because the fork happens unconditionally.
    tensor_parallel.model_parallel_cuda_manual_seed(1234)
    try:
        config = _config()
        torch.manual_seed(1234)
        # [s, b, h, d] with b == 1, the shape mcore folds the micro-batch into.
        query, key, value = (
            torch.randn(
                SEQ_LENGTH, 1, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
            )
            for _ in range(3)
        )

        te_attention = TEDotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        local_attention = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

        # 1. No document masking: plain causal over the whole pack.
        causal_out = te_attention(
            query, key, value, None, AttnMaskType.causal
        ).float()

        # 2. What the patch runs. thd wants [t, h, d]; mcore gets there via
        #    query.squeeze(1) in transformer/attention.py.
        cu_seqlens = _cu_seqlens()
        packed_out = te_attention(
            query.squeeze(1),
            key.squeeze(1),
            value.squeeze(1),
            None,
            AttnMaskType.causal,
            packed_seq_params=PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=max(DOC_LENGTHS),
                max_seqlen_kv=max(DOC_LENGTHS),
            ),
        ).float()
        packed_out = packed_out.reshape(SEQ_LENGTH, -1)

        # 3. The reference: an independent unfused implementation given the dense
        #    block-diagonal mask.
        dense_out = local_attention(
            query, key, value, _document_mask(), AttnMaskType.arbitrary
        ).float()
        dense_out = dense_out.reshape(SEQ_LENGTH, -1)
        causal_out = causal_out.reshape(SEQ_LENGTH, -1)

        # Positive control: the two references must actually disagree, otherwise
        # this test could pass while masking nothing. Only the second document can
        # differ -- the first has no earlier document to leak from.
        leak = (causal_out - dense_out).abs().max().item()
        assert leak > 1e-2, (
            f"causal and document-masked attention differ by only {leak:.2E}; the "
            "fixture is not sensitive enough to detect missing masking"
        )

        # The real assertion: cu_seqlens reproduces the dense document mask.
        # bf16 through two different kernels (cuDNN/flash vs unfused softmax), so
        # the tolerance is loose -- but it is 100x below the leak above, which is
        # the effect size a masking bug would produce.
        torch.testing.assert_close(packed_out, dense_out, atol=2e-2, rtol=2e-2)
    finally:
        Utils.destroy_model_parallel()
