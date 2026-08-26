# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""OELLM PATCH: tests for --packed-doc-attention (pretrain_gpt.py).

The cu_seqlens handed to Transformer Engine must describe exactly the same
block-diagonal attention pattern that --reset-attention-mask would have built as
a dense mask -- the difference being that cu_seqlens actually reaches the kernel,
whereas the dense mask is discarded because the GPT layer specs pin
attn_mask_type to `causal`.
"""

import os
import sys
import types

import numpy
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pretrain_gpt
from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids
from megatron.training import packed_doc_attention as pda
from megatron.training.training import get_pipeline_tensor_shapes
from pretrain_gpt import _build_packed_seq_params, _shared_packed_seq_params

EOD = 0

# Token patterns that exercise every boundary case the derivation has to get right. Shared
# by the dense-mask test and the CPU/GPU equivalence test so the two cannot drift.
TOKEN_PATTERNS = [
    pytest.param([[1, 2, EOD, 3, 4, 5, EOD, 8]], id="three_docs_no_trailing_eod"),
    pytest.param([[1, 2, EOD, 3, 4, 5, 6, EOD]], id="sample_ends_on_eod"),
    pytest.param([[1, 2, 3, 4, 5, 6, 7, 8]], id="no_eod_at_all"),
    pytest.param([[1, EOD, EOD, 4, 5, EOD, 7, 8]], id="consecutive_eods"),
    pytest.param([[EOD, 2, 3, 4, 5, 6, 7, 8]], id="eod_in_first_position"),
    pytest.param([[1, 2, EOD, 4], [5, EOD, 7, 8]], id="mbs2"),
    pytest.param([[1, 2, 3, EOD], [5, 6, 7, EOD]], id="mbs2_both_end_on_eod"),
    pytest.param([[EOD, EOD, EOD, EOD]], id="all_eod"),
    pytest.param([[1, 2, 3, 4]], id="single_sample_single_doc"),
]


def _stub_lone_tp_rank(monkeypatch):
    """Present as a tensor-parallel group of one: no gloo group, so no collective is issued.

    The world size has to be stubbed too, not just the rank -- share_cu_seqlens_over_tp
    asserts that a TP group wider than 1 has a gloo sibling, precisely so a missing group
    cannot let non-zero ranks fall through with a stale buffer.
    """
    monkeypatch.setattr(pda.parallel_state, "get_tensor_model_parallel_group_gloo", lambda: None)
    monkeypatch.setattr(pda.parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(pda.parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)


def _mask_from_cu_seqlens(cu_seqlens, total):
    """Expand cu_seqlens into the causal block-diagonal mask it implies.

    True means "masked out", matching Megatron's convention.
    """
    segment = torch.zeros(total, dtype=torch.long)
    for i in range(len(cu_seqlens) - 1):
        segment[cu_seqlens[i] : cu_seqlens[i + 1]] = i
    causal = torch.tril(torch.ones(total, total, dtype=torch.bool))
    same_document = segment[:, None] == segment[None, :]
    return ~(causal & same_document)


def _dense_reference_mask(tokens):
    """Megatron's own per-sample mask, laid out block-diagonally over the flat pack.

    Folding the micro-batch into one sequence must not let sample i attend to
    sample j, which is what the block-diagonal layout asserts.
    """
    batch_size, seq_length = tokens.shape
    reference = torch.ones(batch_size * seq_length, batch_size * seq_length, dtype=torch.bool)
    for i, row in enumerate(tokens):
        mask, _, _ = _get_ltor_masks_and_position_ids(
            row,
            EOD,
            reset_position_ids=False,
            reset_attention_mask=True,
            eod_mask_loss=False,
            create_attention_mask=True,
        )
        reference[i * seq_length : (i + 1) * seq_length, i * seq_length : (i + 1) * seq_length] = (
            mask[0]
        )
    return reference


@pytest.mark.parametrize("rows", TOKEN_PATTERNS)
def test_cpu_derivation_matches_gpu(rows):
    """The host derivation must be bit-identical to the GPU one it replaced.

    cu_seqlens moved off the device to kill two readbacks per microbatch (max_seqlen for
    TE's kernel-launch int, and the entry count to size the slice). That is only a free win
    if the numbers are the same -- a single off-by-one in the boundary set changes which
    tokens attend to which, silently and without ever raising.
    """
    cpu_tokens = torch.tensor(rows, dtype=torch.long)
    gpu = _build_packed_seq_params(cpu_tokens.cuda(), EOD)
    cu_seqlens, max_seqlen = pda.derive_cu_seqlens_cpu(cpu_tokens, EOD)

    assert cu_seqlens.dtype == numpy.int32
    assert max_seqlen == gpu.max_seqlen_q
    torch.testing.assert_close(torch.from_numpy(cu_seqlens), gpu.cu_seqlens_q.cpu())


def test_cpu_derivation_rejects_device_tensors():
    """Passing CUDA tokens would silently reintroduce the readback this replaced."""
    with pytest.raises(AssertionError, match="host tokens"):
        pda.derive_cu_seqlens_cpu(torch.tensor([[1, 2, EOD, 4]]).cuda(), EOD)


# ---------------------------------------------------------------------------
# ROPE VARIANTS
# ---------------------------------------------------------------------------
# cu_seqlens does two jobs: it masks cross-document attention in the kernel, and it restarts
# RoPE at every document boundary. The second one is delivered by a different code path
# depending on apply_rope_fusion, so both have to be checked -- and the two must agree, or
# turning fusion off silently changes what the model learns.


class _CpGroupStub:
    """Stand-in for the CP group; --packed-doc-attention asserts context_parallel_size == 1."""

    def size(self):
        return 1

    def rank(self):
        return 0


def _rope_freqs(length, dim, device):
    """[s, 1, 1, d] rotary frequencies, the layout rope_utils expects."""
    return torch.randn(length, 1, 1, dim, device=device, dtype=torch.float32)


@pytest.mark.parametrize("rows", TOKEN_PATTERNS)
def test_true_max_seqlen_keeps_rope_off_the_offset_branch(rows):
    """The invariant that makes the true max_seqlen load-bearing, not just tidy.

    rope_utils._apply_rotary_pos_emb_thd picks offset mapping (absolute positions across the
    whole pack) over per-document mapping when ``freqs.size(0) == cu_seqlens[-1]``, and
    freqs is sized by max_seqlen. Since the segments partition the pack and each has length
    >= 1, the true maximum can only equal the total when there is exactly ONE segment -- and
    with one segment the two mappings coincide. So per-document RoPE can never be lost.

    Pinning max_seqlen to a constant (args.seq_length, say) breaks precisely this: it can
    equal cu_seqlens[-1] while several documents are present, which silently stops the
    restart. That is why the derivation computes the real maximum instead.
    """
    cu_seqlens, max_seqlen = pda.derive_cu_seqlens_cpu(torch.tensor(rows, dtype=torch.long), EOD)
    if max_seqlen == int(cu_seqlens[-1]):
        assert len(cu_seqlens) - 1 == 1, (
            "offset mapping would be selected with more than one document present"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="thd rope kernels need CUDA")
@pytest.mark.parametrize("rows", TOKEN_PATTERNS)
def test_thd_rope_restarts_at_every_document_boundary(rows):
    """Every document's first token must get the SAME rotation as the pack's first token.

    Feed identical vectors for every token: with per-document RoPE each document restarts at
    frequency index 0, so all the boundary tokens come out equal. Without the restart they
    would each get their absolute position's frequency and differ. This is the property
    --packed-doc-attention exists to provide, asserted directly rather than via loss deltas.
    """
    from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_thd

    cu_np, max_seqlen = pda.derive_cu_seqlens_cpu(torch.tensor(rows, dtype=torch.long), EOD)
    cu_seqlens = torch.from_numpy(cu_np).cuda()
    total, heads, dim = int(cu_np[-1]), 2, 8

    tokens = torch.ones(total, heads, dim, device="cuda", dtype=torch.float32)
    out = _apply_rotary_pos_emb_thd(
        tokens, cu_seqlens, _rope_freqs(max_seqlen, dim, "cuda"), cp_group=_CpGroupStub()
    )

    first = out[0]
    for start in cu_np[:-1]:
        torch.testing.assert_close(out[int(start)], first)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="thd rope kernels need CUDA")
@pytest.mark.parametrize("rows", TOKEN_PATTERNS)
def test_fused_and_unfused_thd_rope_agree(rows):
    """apply_rope_fusion must be a performance switch, never a semantic one.

    yarn silently forces fusion off (arguments.py disables it whenever
    position_embedding_type != 'rope'), so this pair really is reachable from config.
    """
    fused = pytest.importorskip(
        "megatron.core.extensions.transformer_engine"
    ).fused_apply_rotary_pos_emb_thd
    if fused is None:
        pytest.skip("Transformer Engine fused thd rope unavailable")
    from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_thd

    cu_np, max_seqlen = pda.derive_cu_seqlens_cpu(torch.tensor(rows, dtype=torch.long), EOD)
    cu_seqlens = torch.from_numpy(cu_np).cuda()
    total, heads, dim = int(cu_np[-1]), 2, 8

    tokens = torch.randn(total, heads, dim, device="cuda", dtype=torch.float32)
    freqs = _rope_freqs(max_seqlen, dim, "cuda")

    torch.testing.assert_close(
        fused(tokens, cu_seqlens, freqs, cp_size=1, cp_rank=0),
        _apply_rotary_pos_emb_thd(tokens, cu_seqlens, freqs, cp_group=_CpGroupStub()),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("rows", TOKEN_PATTERNS)
def test_cu_seqlens_matches_dense_document_mask(rows):
    tokens = torch.tensor(rows, dtype=torch.long)
    batch_size, seq_length = tokens.shape
    total = batch_size * seq_length

    params = _build_packed_seq_params(tokens, EOD)
    cu_seqlens = params.cu_seqlens_q
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

    assert params.qkv_format == "thd"
    assert params.cu_seqlens_kv is cu_seqlens
    # TE requires int32 cu_seqlens that span the whole pack with no empty segment.
    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens[0] == 0 and cu_seqlens[-1] == total
    assert all(length > 0 for length in lengths)
    # The true maximum, not args.seq_length: _apply_rotary_pos_emb_thd switches to
    # absolute-position mapping when freqs.size(0) == cu_seqlens[-1], which would
    # silently stop RoPE from restarting per document on the unfused rope path.
    assert params.max_seqlen_q == max(lengths)
    assert params.max_seqlen_kv == max(lengths)

    torch.testing.assert_close(
        _mask_from_cu_seqlens(cu_seqlens, total), _dense_reference_mask(tokens)
    )


@pytest.mark.parametrize(
    "packed, seq_length, micro_batch_size, expected",
    [
        pytest.param(False, 4096, 2, (4096, 2), id="off_passes_through"),
        pytest.param(True, 4096, 2, (8192, 1), id="on_folds_batch_into_sequence"),
        pytest.param(True, 4096, 1, (4096, 1), id="on_mbs1_is_identity"),
    ],
)
def test_pipeline_tensor_shapes_fold(packed, seq_length, micro_batch_size, expected):
    """The p2p buffers must be folded exactly as get_batch folds the micro-batch.

    If these disagree the element count still matches and nothing raises -- the receiving
    stage just reinterprets a [t, 1, h] activation as [s, b, h], which is silently wrong.
    """
    args = types.SimpleNamespace(
        packed_doc_attention=packed,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
    )
    assert get_pipeline_tensor_shapes(args) == expected
    # Whatever the fold, the number of tokens in flight is unchanged.
    folded_seq, folded_mbs = get_pipeline_tensor_shapes(args)
    assert folded_seq * folded_mbs == seq_length * micro_batch_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU for the shared buffer")
@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([[1, 2, EOD, 3, 4, 5, EOD, 8]], id="three_docs"),
        pytest.param([[1, 2, EOD, 4], [5, EOD, 7, 8]], id="mbs2"),
        pytest.param([[1, 2, 3, 4, 5, 6, 7, 8]], id="single_document"),
    ],
)
def test_shared_packed_seq_params_roundtrip(rows, monkeypatch):
    """The fixed-size TP buffer must reproduce exactly what rank 0 computed.

    Encoding is [n_entries, max_seqlen, cu_seqlens...] padded to a fixed length so every
    rank agrees on the shape before it knows the document count; the slice arithmetic on
    the way out is the part that can silently drop or pad an entry.
    """
    # Pretend to be a lone tensor-parallel rank: no gloo group, so no collective is issued.
    _stub_lone_tp_rank(monkeypatch)

    cpu_tokens = torch.tensor(rows, dtype=torch.long)
    tokens = cpu_tokens.cuda()
    packed_length = tokens.numel()

    direct = _build_packed_seq_params(tokens, EOD)
    # Stand in for get_batch_on_this_tp_rank, which derives on the host before the H2D copy.
    pda.stash_cpu_cu_seqlens(*pda.derive_cu_seqlens_cpu(cpu_tokens, EOD))
    shared = _shared_packed_seq_params(packed_length)

    assert shared.qkv_format == "thd"
    assert shared.max_seqlen_q == direct.max_seqlen_q
    assert shared.max_seqlen_kv == direct.max_seqlen_kv
    assert shared.cu_seqlens_q.dtype == torch.int32
    torch.testing.assert_close(shared.cu_seqlens_q.cpu(), direct.cu_seqlens_q.cpu())
    # No trailing padding leaked into the slice, and it still spans the whole pack.
    assert shared.cu_seqlens_q[-1].item() == packed_length
    assert shared.cu_seqlens_q.numel() == direct.cu_seqlens_q.numel()


def test_cpu_header_handoff_is_pop_not_peek():
    """The hand-off slot must clear on read, and reset() must not leave one behind.

    This is the property the module-global hand-off rests on: a value can be consumed at
    most once, so a stash that nobody consumed (training.dummy_train_step does exactly
    that) is overwritten before the next take rather than being read for the wrong
    microbatch. Lose it and the failure is silently wrong document boundaries.
    """
    pda.reset()
    assert pda.take_cpu_cu_seqlens() is None, "starts empty"

    cu = numpy.array([0, 4, 8], dtype=numpy.int32)
    pda.stash_cpu_cu_seqlens(cu, 4)
    first = pda.take_cpu_cu_seqlens()
    assert first is not None and first[1] == 4
    assert pda.take_cpu_cu_seqlens() is None, "second read must not repeat the value"

    # An unconsumed stash is replaced, not queued -- the dummy_train_step case.
    pda.stash_cpu_cu_seqlens(numpy.array([0, 8], dtype=numpy.int32), 8)
    pda.stash_cpu_cu_seqlens(cu, 4)
    taken = pda.take_cpu_cu_seqlens()
    assert taken[1] == 4, "take must see the most recent stash"

    pda.stash_cpu_cu_seqlens(cu, 4)
    pda.reset()
    assert pda.take_cpu_cu_seqlens() is None, "reset must drop an unconsumed hand-off"


def test_scatter_shares_cu_seqlens_across_chunks():
    """Chunks 1..VPP-1 must get chunk 0's cu_seqlens, and must not share its list object.

    Scatter was broken at VPP>1: prefetch_iteration only fills the broadcast buffer inside
    `if reads_data:`, and reads_data is per-chunk, so the model-parallel source rank -- the
    first stage for chunk 0 only -- never wrote anything for the other chunks. Zeros were
    broadcast, n_entries came out 0, and TE died on an empty cu_seqlens 60 layers later
    (fused_rope.cu:477, 1865 ranks, job 1497767).

    Sharing is legitimate because cu_seqlens depends on the MICROBATCH, not the chunk: every
    virtual stage sees microbatch m on the same tokens. The copy matters -- pop() mutates the
    list, so a shared reference would let whichever chunk ran first consume the others' entries.
    """
    pda.reset()
    entries = [(torch.tensor([0, 4, 8], dtype=torch.int32), 4) for _ in range(3)]
    pda._CU_SEQLENS_STASH[pda._key(0)] = list(entries)

    for vp in (1, 2, 3):
        pda.share_cu_seqlens_from(vp, 0)

    for vp in (1, 2, 3):
        shared = pda._CU_SEQLENS_STASH[pda._key(vp)]
        assert len(shared) == 3
        assert shared is not pda._CU_SEQLENS_STASH[pda._key(0)], "must be a copy, not an alias"
        for (cu, m), (cu0, m0) in zip(shared, entries):
            torch.testing.assert_close(cu, cu0)
            assert m == m0

    # Draining one chunk must leave the others intact -- the aliasing failure mode.
    pda._CU_SEQLENS_STASH[pda._key(1)].pop(0)
    assert len(pda._CU_SEQLENS_STASH[pda._key(2)]) == 3
    assert len(pda._CU_SEQLENS_STASH[pda._key(0)]) == 3
    pda.reset()


def test_scatter_refuses_to_share_from_a_chunk_that_never_derived():
    """The failure has to surface here, not inside a TE kernel assert much later."""
    pda.reset()
    with pytest.raises(AssertionError, match="was not prefetched"):
        pda.share_cu_seqlens_from(1, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a device tensor")
def test_consecutive_microbatches_do_not_alias(monkeypatch):
    """Each microbatch must own its cu_seqlens, because TE reads it again in backward.

    Under 1F1B the backward for microbatch m runs several microbatches after its forward.
    If the device tensor were a view into a recycled buffer, microbatch m+1 would overwrite
    the boundaries m is still going to use -- and nothing would raise, because the dtype and
    (usually) the shape match. Attention would simply be masked against the wrong document
    layout in the backward pass.
    """
    _stub_lone_tp_rank(monkeypatch)
    packed_length = 8

    pda.stash_cpu_cu_seqlens(*pda.derive_cu_seqlens_cpu(torch.tensor([[1, 2, EOD, 4, 5, 6, 7, 8]]), EOD))
    first, _ = pda.share_cu_seqlens_over_tp(packed_length)
    kept = first.clone()

    pda.stash_cpu_cu_seqlens(*pda.derive_cu_seqlens_cpu(torch.tensor([[EOD, 2, EOD, 4, EOD, 6, 7, 8]]), EOD))
    second, _ = pda.share_cu_seqlens_over_tp(packed_length)

    assert first.data_ptr() != second.data_ptr()
    torch.cuda.synchronize()
    torch.testing.assert_close(first, kept)
