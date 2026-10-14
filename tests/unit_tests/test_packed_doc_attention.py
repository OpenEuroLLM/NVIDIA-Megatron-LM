# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""OELLM PATCH: tests for scattered packed document attention.

The feature has two halves and this file covers both.

DISTRIBUTION (megatron/training/packed_doc_attention.py). Every pipeline stage needs the
same cu_seqlens, but only the endpoint stages read the dataloader. The scatter publishes it
once per iteration over the model-parallel group. What has to hold: the wire format survives
a round trip for an arbitrary mix of document counts, the per-chunk queues hand each chunk
its microbatches in schedule order, virtual stages share one publication rather than
publishing VPP times, and a stage that does not read still gets its metadata.

SHAPE (megatron/training/training.py:get_pipeline_tensor_shapes). Packing folds [b, s] into
[1, b*s], and the pipeline p2p buffers have to be folded to match or a middle stage
reinterprets the activation it receives. That is not a hypothetical: it is what killed
job 1511160 with `RuntimeError: expected 3D tensor` inside TE's thd rope kernel.

CONTEXT PARALLELISM (snap_cu_seqlens_to_multiple / apply_cp_document_padding). TE's
context-parallel thd kernels require a non-None cu_seqlens_padded whose document lengths are
divisible by 2 * cp_size, and GPTDataset emits neither. The snapping has to produce a valid
partition of the SAME token stream -- covering, non-decreasing, no empty segments -- while
satisfying that divisibility.

These run on CPU. The collectives are exercised through a model-parallel group of one, which
is the branch where the encode and decode arithmetic still runs but no NCCL is involved.
"""

import argparse
import os
import sys

import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from megatron.training import packed_doc_attention as pda
from megatron.training.training import get_pipeline_tensor_shapes

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

# Document layouts that between them hit every boundary case the encode/decode and the CP
# snapping have to survive. Each entry is the cumulative-length array for one microbatch.
CU_SEQLENS_PATTERNS = [
    pytest.param([0, 3, 7, 16], id="three_uneven_docs"),
    pytest.param([0, 16], id="single_document_fills_the_pack"),
    pytest.param([0, 8, 16], id="two_equal_docs"),
    pytest.param([0, 1, 2, 3, 16], id="run_of_one_token_docs"),
    pytest.param([0, 15, 16], id="one_token_tail"),
    pytest.param(list(range(0, 17)), id="every_token_its_own_doc"),
]


@pytest.fixture(autouse=True)
def _clean_stash():
    """No test may see another's prefetch. The stashes are module globals by design."""
    pda.reset()
    yield
    pda.reset()


@pytest.fixture
def lone_rank(monkeypatch):
    """Present as a model-parallel group of one, so no collective is issued.

    This is the honest single-process stand-in for the real thing: prefetch_iteration still
    builds the header, still computes the payload offsets and still decodes them back into
    per-microbatch slices. Only torch.distributed.broadcast is skipped, and a broadcast into
    an identically-sized buffer from the rank that filled it is the identity.
    """
    monkeypatch.setattr(pda.parallel_state, "get_model_parallel_group", lambda: None)
    monkeypatch.setattr(pda.parallel_state, "get_model_parallel_src_rank", lambda: 0)
    monkeypatch.setattr(pda.torch.distributed, "get_rank", lambda: 0)
    # prefetch_iteration allocates its header on the current device. There is no GPU here and
    # the arithmetic does not need one.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")


def _batch(cu_seqlens, cu_seqlens_padded=None, tokens=True):
    """A batch dict shaped the way _read_microbatch returns one.

    cu_seqlens carries a leading batch dimension -- that is the convention
    flatten_batch_for_packed_sequences leaves it in and forward_step squeezes.
    """
    cu = torch.tensor(cu_seqlens, dtype=torch.int32).unsqueeze(0)
    padded = (
        None
        if cu_seqlens_padded is None
        else torch.tensor(cu_seqlens_padded, dtype=torch.int32).unsqueeze(0)
    )
    lengths = cu[0, 1:] - cu[0, :-1]
    return {
        "cu_seqlens": cu,
        "cu_seqlens_padded": padded,
        "max_seqlen": lengths.max().reshape(1),
        "tokens": torch.arange(int(cu[0, -1])).reshape(1, -1) if tokens else None,
    }


def _prefetch(batches, vp_stage=None, reads_data=True, derive=True):
    """Drive prefetch_iteration over a list of pre-made batch dicts."""
    iterator = iter(batches)
    pda.prefetch_iteration(
        data_iterator=None,
        vp_stage=vp_stage,
        num_microbatches=len(batches),
        fetch_batch=lambda _: next(iterator),
        reads_data=reads_data,
        derive=derive,
    )


# ---------------------------------------------------------------------------
# Wire format: header + payload round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cu_seqlens", CU_SEQLENS_PATTERNS)
def test_single_microbatch_round_trips(lone_rank, cu_seqlens):
    """What the source published is what every rank pops."""
    _prefetch([_batch(cu_seqlens)])
    batch, metadata = pda.pop(None, reads_data=True)

    assert batch is not None
    torch.testing.assert_close(
        metadata.cu_seqlens.reshape(-1), torch.tensor(cu_seqlens, dtype=torch.int32)
    )
    assert metadata.cu_seqlens.shape[0] == 1, "the leading batch dim forward_step squeezes"
    assert metadata.cu_seqlens_padded is None, "CP=1 must not invent a padded array"
    expected_max = max(b - a for a, b in zip(cu_seqlens, cu_seqlens[1:]))
    assert int(metadata.max_seqlen.item()) == expected_max


def test_microbatches_come_back_in_schedule_order(lone_rank):
    """Different document counts per microbatch, decoded from one flat payload.

    This is the part a single fixed-width buffer would have made trivial and a packed one
    makes worth testing: microbatch i's slice starts where microbatch i-1's ended, and the
    two arrays of a microbatch are adjacent. An off-by-one in that offset walk would hand a
    later microbatch a shifted view -- right dtype, right-ish shape, wrong boundaries, and
    nothing anywhere would raise.
    """
    layouts = [[0, 4, 16], [0, 16], [0, 1, 2, 3, 16], [0, 8, 12, 16]]
    _prefetch([_batch(layout) for layout in layouts])

    for expected in layouts:
        _, metadata = pda.pop(None, reads_data=True)
        torch.testing.assert_close(
            metadata.cu_seqlens.reshape(-1), torch.tensor(expected, dtype=torch.int32)
        )


def test_consecutive_microbatches_do_not_alias(lone_rank):
    """Two microbatches must not end up as two views of the same slice.

    The payload is one concatenated tensor and the entries are views into it, so a decode
    that reused an offset would silently give microbatch 1 microbatch 0's boundaries.
    """
    _prefetch([_batch([0, 4, 16]), _batch([0, 12, 16])])
    _, first = pda.pop(None, reads_data=True)
    _, second = pda.pop(None, reads_data=True)
    assert not torch.equal(first.cu_seqlens, second.cu_seqlens)


def test_padded_cu_seqlens_round_trips(lone_rank):
    """cu_seqlens_padded travels alongside cu_seqlens, not instead of it.

    At CP>1 TE indexes with the padded array and computes lengths from the real one, so both
    have to arrive and they have to arrive distinguishable.
    """
    _prefetch([_batch([0, 3, 7, 16], cu_seqlens_padded=[0, 4, 8, 16])])
    _, metadata = pda.pop(None, reads_data=True)

    torch.testing.assert_close(
        metadata.cu_seqlens.reshape(-1), torch.tensor([0, 3, 7, 16], dtype=torch.int32)
    )
    assert metadata.cu_seqlens_padded is not None
    torch.testing.assert_close(
        metadata.cu_seqlens_padded.reshape(-1), torch.tensor([0, 4, 8, 16], dtype=torch.int32)
    )


def test_max_seqlen_is_a_host_tensor(lone_rank):
    """forward_step calls .item() on this once per microbatch.

    On a device tensor that drains the stream -- the exact per-microbatch host sync the
    scatter exists to collapse into one per iteration. Keeping it on the CPU is not a detail,
    it is the reason the sync count is one.
    """
    _prefetch([_batch([0, 5, 16])])
    _, metadata = pda.pop(None, reads_data=True)
    assert metadata.max_seqlen.device.type == "cpu"
    assert int(metadata.max_seqlen.item()) == 11


# ---------------------------------------------------------------------------
# Pipeline stages that do not read
# ---------------------------------------------------------------------------


def test_middle_stage_gets_metadata_without_a_batch(lone_rank):
    """A stage that never touched the dataloader still gets cu_seqlens.

    This is the whole point of the scatter, and the alternative -- letting the middle stage
    read -- is what upstream does and what runs a node out of RAM at PP=4/VPP=4 (job 1497412).
    """
    _prefetch([_batch([0, 6, 16])])
    # Same publication, popped by a rank whose stage reads nothing.
    batch, metadata = pda.pop(None, reads_data=False)

    assert batch is None
    torch.testing.assert_close(
        metadata.cu_seqlens.reshape(-1), torch.tensor([0, 6, 16], dtype=torch.int32)
    )


def test_popping_without_a_prefetch_is_an_error():
    """get_batch must not silently serve a stale or empty stash.

    Every call site that drives forward_backward_func has to call
    training.maybe_prefetch_cu_seqlens first; forgetting one (evaluate() was nearly it) has
    to fail here rather than 60 layers down in a TE assert.
    """
    with pytest.raises(AssertionError, match="was not prefetched"):
        pda.pop(None, reads_data=False)


def test_reset_clears_the_stash(lone_rank):
    """Iterations must not leak into each other."""
    _prefetch([_batch([0, 16])])
    assert pda.is_primed(None)
    pda.reset()
    assert not pda.is_primed(None)


# ---------------------------------------------------------------------------
# Virtual pipeline stages
# ---------------------------------------------------------------------------


def test_virtual_stages_share_one_publication(lone_rank):
    """Chunk 1 gets chunk 0's boundaries, and consuming one does not consume the other.

    cu_seqlens depends on the microbatch, not the chunk: microbatch m flows through virtual
    stage 0, then 1, then 2, on the same tokens. Publishing per chunk instead is what
    broadcast an EMPTY cu_seqlens at PP=4/VPP=4 and killed 1865 ranks in
    `fused_rope.cu:477 Assertion failed: cu_seqlens != nullptr` (job 1497767) -- the source
    rank is the first stage for chunk 0 and for no other chunk, so for the others it never
    filled the buffer.
    """
    layouts = [[0, 4, 16], [0, 9, 16]]
    # Chunk 0 publishes; chunk 1 fetches its own batches (it needs the labels) but takes the
    # metadata from chunk 0. This mirrors training.maybe_prefetch_cu_seqlens exactly.
    _prefetch([_batch(layout) for layout in layouts], vp_stage=0, derive=True)
    _prefetch([_batch(layout) for layout in layouts], vp_stage=1, derive=False)
    pda.share_metadata_from(1, 0)

    for expected in layouts:
        _, from_chunk_1 = pda.pop(1, reads_data=True)
        torch.testing.assert_close(
            from_chunk_1.cu_seqlens.reshape(-1), torch.tensor(expected, dtype=torch.int32)
        )
    # Chunk 0's queue is untouched: share_metadata_from copies the list, it does not alias it.
    for expected in layouts:
        _, from_chunk_0 = pda.pop(0, reads_data=True)
        torch.testing.assert_close(
            from_chunk_0.cu_seqlens.reshape(-1), torch.tensor(expected, dtype=torch.int32)
        )


def test_refuses_to_share_from_a_chunk_that_never_published(lone_rank):
    """Wiring the chunks in the wrong order must fail loudly, not hand back nothing."""
    _prefetch([_batch([0, 16])], vp_stage=1, derive=False)
    with pytest.raises(AssertionError, match="it was not prefetched"):
        pda.share_metadata_from(1, 0)


def test_non_publishing_chunk_still_stashes_its_batches(lone_rank):
    """derive=False skips the collectives, not the dataloader reads.

    The last-stage chunk needs its labels whether or not it publishes cu_seqlens, so the
    batches must still be there after share_metadata_from fills in the metadata.
    """
    _prefetch([_batch([0, 16])], vp_stage=0, derive=True)
    _prefetch([_batch([0, 16])], vp_stage=1, derive=False)
    pda.share_metadata_from(1, 0)
    batch, _ = pda.pop(1, reads_data=True)
    assert batch is not None and batch["tokens"] is not None


# ---------------------------------------------------------------------------
# Pipeline tensor shapes -- the RuntimeError: expected 3D tensor regression
# ---------------------------------------------------------------------------


def _args(**overrides):
    defaults = dict(
        seq_length=8192,
        micro_batch_size=2,
        sft=False,
        dataloader_inter_document_masking=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        pytest.param({}, (8192, 2), id="unpacked_is_untouched"),
        pytest.param(
            {"dataloader_inter_document_masking": True}, (16384, 1), id="masking_folds"
        ),
        pytest.param({"sft": True}, (16384, 1), id="sft_folds_too"),
        pytest.param(
            {"dataloader_inter_document_masking": True, "micro_batch_size": 1},
            (8192, 1),
            id="mbs1_fold_is_the_identity",
        ),
    ],
)
def test_pipeline_tensor_shapes(overrides, expected):
    assert get_pipeline_tensor_shapes(_args(**overrides)) == expected


def test_pipeline_tensor_shapes_uses_the_eval_micro_batch_size():
    """evaluate() may run a different micro_batch_size from train_step.

    Folding with args.micro_batch_size there would size the validation p2p buffers for the
    training shape, which is the same silent reinterpretation this function exists to prevent.
    """
    args = _args(dataloader_inter_document_masking=True)
    assert get_pipeline_tensor_shapes(args, micro_batch_size=4) == (8192 * 4, 1)


@pytest.mark.parametrize("micro_batch_size", [1, 2, 8])
def test_folded_shape_makes_the_thd_query_three_dimensional(micro_batch_size):
    """The regression from job 1511160, expressed as the shape invariant behind it.

    attention.py does `query.squeeze(1)` to reach the [t, h, d] that TE's thd kernels
    require. squeeze(1) only removes the dimension if it is 1, so it is the p2p buffer shape
    -- not the squeeze -- that decides whether the tensor arrives 3-D. With upstream's
    unfolded (seq_length, micro_batch_size) the middle stage receives [s, b, h], its query is
    4-D, the squeeze is a no-op and TE raises `RuntimeError: expected 3D tensor`.
    """
    seq_length, hidden, heads, head_dim = 128, 64, 4, 16
    args = _args(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        dataloader_inter_document_masking=True,
    )
    pipeline_seq_length, pipeline_micro_batch_size = get_pipeline_tensor_shapes(args)

    # What a middle stage receives over p2p, and the query it projects out of it.
    query = torch.empty(pipeline_seq_length, pipeline_micro_batch_size, heads, head_dim)
    assert query.squeeze(1).dim() == 3

    # And the counter-example: the unfolded shape only works at micro_batch_size 1.
    unfolded = torch.empty(seq_length, micro_batch_size, heads, head_dim)
    assert (unfolded.squeeze(1).dim() == 3) == (micro_batch_size == 1)

    # The fold moves tokens between the two dimensions; it must not lose or add any.
    assert pipeline_seq_length * pipeline_micro_batch_size == seq_length * micro_batch_size
    assert hidden  # keeps the hidden size in the test's stated shape, unused by the assertions


class _GroupStub:
    """Minimal stand-in for a process group: get_tensor_shapes only calls .size()."""

    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


@pytest.mark.parametrize("cp", [1, 2, 4])
@pytest.mark.parametrize("tp", [1, 4])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("micro_batch_size", [1, 2])
def test_p2p_shape_matches_the_real_activation_across_tp_cp(
    cp, tp, sequence_parallel, micro_batch_size
):
    """Feed our folded numbers to the REAL get_tensor_shapes and compare to the real tensor.

    This is the check that would have caught the shape bug without a 16-node job. Two
    independent computations of the same thing:

      * what the pipeline will ALLOCATE -- schedules.get_tensor_shapes, unmodified upstream
        code, given the (seq_length, micro_batch_size) that get_pipeline_tensor_shapes hands
        it;
      * what will actually ARRIVE -- tokens folded to [1, mbs*seq] by
        flatten_batch_for_packed_sequences, split to [1, mbs*seq/cp] by the context-parallel
        zigzag, transposed to [t, b, h] and, under sequence parallelism, scattered along the
        sequence dimension to [t/tp, b, h].

    They must agree for every combination, because a mismatch does NOT raise -- the element
    count is preserved and the receiving stage silently reinterprets the activation. That is
    the failure mode that reached a 16-node job as `RuntimeError: expected 3D tensor` sixty
    layers downstream (job 1511160).
    """
    from megatron.core.pipeline_parallel.schedules import get_tensor_shapes

    seq_length, hidden = 128, 64
    args = _args(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        dataloader_inter_document_masking=True,
    )
    config = argparse.Namespace(
        variable_seq_lengths=False, sequence_parallel=sequence_parallel, hidden_size=hidden
    )

    pipeline_seq_length, pipeline_micro_batch_size = get_pipeline_tensor_shapes(args)
    allocated = get_tensor_shapes(
        seq_length=pipeline_seq_length,
        micro_batch_size=pipeline_micro_batch_size,
        decoder_seq_length=None,
        config=config,
        tp_group=_GroupStub(tp),
        cp_group=_GroupStub(cp),
    )

    # Independently: the packed sequence, split by CP, then scattered by sequence parallelism.
    arriving_tokens = seq_length * micro_batch_size // cp
    if sequence_parallel:
        arriving_tokens //= tp
    assert allocated == [(arriving_tokens, 1, hidden)]

    # And the batch dimension must be 1 whatever the parallelism, because that is what makes
    # attention.py's query.squeeze(1) produce the 3-D [t, h, d] TE's thd kernels require.
    assert allocated[0][1] == 1


@pytest.mark.parametrize("cp", [1, 2])
@pytest.mark.parametrize("micro_batch_size", [1, 2])
def test_unpacked_p2p_shape_is_untouched(cp, micro_batch_size):
    """The BSHD path must be bit-identical to upstream -- this changes nothing for it."""
    from megatron.core.pipeline_parallel.schedules import get_tensor_shapes

    seq_length, hidden = 128, 64
    args = _args(seq_length=seq_length, micro_batch_size=micro_batch_size)
    config = argparse.Namespace(
        variable_seq_lengths=False, sequence_parallel=False, hidden_size=hidden
    )
    pipeline_seq_length, pipeline_micro_batch_size = get_pipeline_tensor_shapes(args)
    assert (pipeline_seq_length, pipeline_micro_batch_size) == (seq_length, micro_batch_size)
    allocated = get_tensor_shapes(
        seq_length=pipeline_seq_length,
        micro_batch_size=pipeline_micro_batch_size,
        decoder_seq_length=None,
        config=config,
        tp_group=_GroupStub(1),
        cp_group=_GroupStub(cp),
    )
    assert allocated == [(seq_length // cp, micro_batch_size, hidden)]


# ---------------------------------------------------------------------------
# Context parallelism: dataset provenance fields
# ---------------------------------------------------------------------------


class _CpGroupStub:
    """A context-parallel group of the given size, seen from rank 0."""

    def __init__(self, size):
        self._size = size


@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_split_ignores_dataset_provenance_fields(monkeypatch, cp_size):
    """A 1-D field like BlendedDataset's dataset_id must not reach the sequence split.

    BlendedDataset returns ``{"dataset_id": ..., **sample}`` (blended_dataset.py:109) and
    default_collate makes that a 1-D [micro_batch_size] tensor. get_batch normalises the
    BATCH_KEYS entries but does not strip unknown ones, so it arrives here -- and the
    per-sequence split's ``val.shape[seq_dim]`` raises IndexError on it.

    Measured on rank 16, job 1516834 (TP4 x CP2 x PP4, 32 nodes): zero iterations. It hits
    ANY context-parallel run on a blended dataset, with or without masking -- the masking
    path escapes only because it routes to per-document balancing.
    """
    from megatron.core import utils as core_utils

    monkeypatch.setattr(core_utils.torch.distributed, "get_world_size", lambda group: cp_size)
    monkeypatch.setattr(core_utils.torch.distributed, "get_rank", lambda group: 0)

    seq_length, micro_batch_size = 32, 2
    batch = {
        "tokens": torch.arange(micro_batch_size * seq_length).reshape(
            micro_batch_size, seq_length
        ),
        "labels": torch.zeros(micro_batch_size, seq_length, dtype=torch.long),
        "loss_mask": torch.ones(micro_batch_size, seq_length),
        "position_ids": torch.zeros(micro_batch_size, seq_length, dtype=torch.long),
        "attention_mask": None,
        "cu_seqlens": None,
        "cu_seqlens_padded": None,
        "max_seqlen": None,
        "local_cp_size": None,
        "hybrid_cp_group": None,
        # the provenance field that crashed the job
        "dataset_id": torch.zeros(micro_batch_size, dtype=torch.long),
    }

    result = core_utils._get_batch_on_this_cp_rank_per_sequence_balancing(
        batch, cp_group=_CpGroupStub(cp_size)
    )

    # the real sequence tensors are split to this rank's share...
    for key in ("tokens", "labels", "loss_mask", "position_ids"):
        assert result[key].shape == (micro_batch_size, seq_length // cp_size)
    # ...and the provenance field is passed through untouched, not split and not dropped
    assert result["dataset_id"].shape == (micro_batch_size,)
    assert torch.equal(result["dataset_id"], torch.zeros(micro_batch_size, dtype=torch.long))


def test_cp_split_still_partitions_the_attention_mask(monkeypatch):
    """The rank-based skip must not accidentally exclude the 4-D attention mask.

    attention_mask uses seq_dim=2, so the guard has to compare against ITS seq_dim rather
    than a fixed 1 -- otherwise the fix for dataset_id would silently stop splitting the mask.
    """
    from megatron.core import utils as core_utils

    cp_size = 2
    monkeypatch.setattr(core_utils.torch.distributed, "get_world_size", lambda group: cp_size)
    monkeypatch.setattr(core_utils.torch.distributed, "get_rank", lambda group: 0)

    seq_length, micro_batch_size = 32, 2
    batch = {
        "tokens": torch.zeros(micro_batch_size, seq_length, dtype=torch.long),
        "attention_mask": torch.ones(
            micro_batch_size, 1, seq_length, seq_length, dtype=torch.bool
        ),
        "cu_seqlens": None,
    }
    result = core_utils._get_batch_on_this_cp_rank_per_sequence_balancing(
        batch, cp_group=_CpGroupStub(cp_size)
    )
    assert result["attention_mask"].shape == (
        micro_batch_size,
        1,
        seq_length // cp_size,
        seq_length,
    )


# ---------------------------------------------------------------------------
# Context parallelism: boundary snapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cu_seqlens", CU_SEQLENS_PATTERNS)
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_snapped_boundaries_still_partition_the_same_tokens(cu_seqlens, cp_size):
    """Snapping moves boundaries; it must not move tokens.

    The invariants TE and the rest of the batch depend on:
      * starts at 0 and ends at the packed length -- otherwise the segments stop covering
        the token stream and the kernel reads past the end or drops a tail;
      * strictly increasing -- TE requires every segment to have length >= 1;
      * every entry divisible by 2 * cp_size -- thd_get_partitioned_indices splits each
        document into 2 * cp_size chunks and cannot split a length it does not divide.
    """
    multiple = 2 * cp_size
    original = torch.tensor(cu_seqlens, dtype=torch.int32)
    snapped = pda.snap_cu_seqlens_to_multiple(original.clone(), multiple)

    assert snapped[0] == 0
    assert snapped[-1] == original[-1], "the packed length is fixed; only the interior moves"
    assert torch.all(snapped[1:] > snapped[:-1]), "no zero-length segments"
    if multiple > 1:
        assert torch.all(snapped % multiple == 0)
    assert snapped.dtype == torch.int32


def test_snapping_is_the_identity_at_cp_one():
    """CP=1 needs no padded array and no divisibility, so nothing may change."""
    original = torch.tensor([0, 3, 7, 16], dtype=torch.int32)
    assert torch.equal(pda.snap_cu_seqlens_to_multiple(original, 1), original)


def test_snapping_moves_each_boundary_less_than_the_multiple():
    """The cost of the approximation is bounded, and this is the bound.

    A boundary that moved by `multiple` or more would mean a whole chunk of one document
    being attributed to its neighbour rather than the few tokens the docstring promises.
    """
    multiple = 8
    original = torch.tensor([0, 5, 13, 22, 32], dtype=torch.int32)
    snapped = pda.snap_cu_seqlens_to_multiple(original.clone(), multiple)
    # Every surviving boundary is within `multiple` of some original boundary.
    for boundary in snapped.tolist():
        assert min(abs(boundary - o) for o in original.tolist()) < multiple


def test_snapping_absorbs_documents_shorter_than_the_chunk():
    """A document shorter than 2 * cp_size cannot be split, so it must be merged away.

    Leaving it in would hand TE a segment it cannot partition. The result is shorter than the
    input, which is why nothing downstream may assume the entry count is preserved.
    """
    original = torch.tensor([0, 1, 2, 3, 16], dtype=torch.int32)
    snapped = pda.snap_cu_seqlens_to_multiple(original, 8)
    assert snapped.numel() < original.numel()
    assert torch.equal(snapped, torch.tensor([0, 16], dtype=torch.int32))


@pytest.mark.parametrize("cp_size", [2, 4])
def test_apply_cp_document_padding_satisfies_te(cp_size):
    """What TE's context-parallel thd path checks for, checked here instead.

    context_parallel.py:419 asserts cu_seqlens_padded is not None, and
    thd_get_partitioned_indices requires divisible document lengths. GPTDataset gives neither,
    so this is the function that has to.
    """
    batch = _batch([0, 3, 7, 16])
    result = pda.apply_cp_document_padding(batch, cp_size)

    assert result["cu_seqlens_padded"] is not None
    assert result["cu_seqlens"].shape[0] == 1, "the leading batch dim survives"
    torch.testing.assert_close(result["cu_seqlens"], result["cu_seqlens_padded"])
    # Equal but not the same object: the CP split reads one and TE indexes with the other,
    # and a shared tensor would let an in-place change to either corrupt both.
    assert result["cu_seqlens"].data_ptr() != result["cu_seqlens_padded"].data_ptr()
    assert torch.all(result["cu_seqlens"][0] % (2 * cp_size) == 0)

    # max_seqlen must be recomputed: snapping can lengthen the longest document, and TE takes
    # this as the kernel's bound, so an understated value is out-of-bounds work rather than a
    # rounding difference.
    lengths = result["cu_seqlens"][0, 1:] - result["cu_seqlens"][0, :-1]
    assert int(result["max_seqlen"].item()) == int(lengths.max())


def test_apply_cp_document_padding_is_a_noop_at_cp_one():
    """At CP=1 TE wants cu_seqlens_padded to stay None, so it must not be invented."""
    batch = _batch([0, 3, 7, 16])
    before = batch["cu_seqlens"].clone()
    result = pda.apply_cp_document_padding(batch, 1)
    assert result["cu_seqlens_padded"] is None
    torch.testing.assert_close(result["cu_seqlens"], before)


def test_apply_cp_document_padding_ignores_unpacked_batches():
    """No cu_seqlens means no packing; the function must not manufacture one."""
    batch = {"cu_seqlens": None, "cu_seqlens_padded": None, "max_seqlen": None}
    assert pda.apply_cp_document_padding(batch, 4)["cu_seqlens"] is None


# ---------------------------------------------------------------------------
# Which runs the scatter applies to
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "masking, sft, expected",
    [
        pytest.param(True, False, True, id="pretraining_with_masking"),
        pytest.param(False, False, False, id="masking_off"),
        # SFT keeps upstream's per-stage read: its dataset is small, and it emits a genuine
        # padded layout rather than the boundary-snapped one, so the two must not share a path.
        pytest.param(True, True, False, id="sft_keeps_the_upstream_path"),
        pytest.param(False, True, False, id="sft_without_masking"),
    ],
)
def test_is_scattered(masking, sft, expected):
    args = argparse.Namespace(dataloader_inter_document_masking=masking, sft=sft)
    assert pda.is_scattered(args) is expected


def test_is_scattered_tolerates_missing_attributes():
    """Called from training.py against arg namespaces that predate both flags."""
    assert pda.is_scattered(argparse.Namespace()) is False


# ---------------------------------------------------------------------------
# Varlen FLOPs accounting
# ---------------------------------------------------------------------------


# (tp, cp, pp, vpp, dp) grids, chosen so every axis varies independently of the others and
# every pair varies together at least once. `vpp=None` is the no-virtual-pipelining case,
# which is what the accessor actually returns -- not 1.
PARALLELISM_GRID = [
    pytest.param(1, 1, 1, None, 1, id="single_rank"),
    pytest.param(4, 1, 1, None, 8, id="tp_only"),
    pytest.param(1, 1, 4, None, 8, id="pp_only"),
    pytest.param(1, 4, 1, None, 8, id="cp_only"),
    pytest.param(1, 1, 4, 4, 8, id="vpp_only"),
    pytest.param(4, 1, 4, 2, 4, id="production_gate_shape"),
    pytest.param(4, 1, 4, 4, 128, id="production_512n_shape"),
    pytest.param(2, 2, 4, 4, 2, id="all_four_axes"),
    pytest.param(8, 4, 2, 2, 1, id="cp_heavy_dp1"),
]


def _simulate_world_allreduce(tp, cp, pp, vpp, dp, microbatch_docs):
    """What the world all-reduce leaves in the accumulator, built rank by rank.

    Models the real accumulation rather than asserting a formula against itself:
      * every rank runs ``forward_step`` once per (model chunk, micro-batch), so it calls
        ``update_*`` ``vpp * M`` times;
      * each call adds the WHOLE micro-batch's cu_seqlens -- the context-parallel split
        leaves cu_seqlens untouched, so a CP rank reports the full pack, not its shard;
      * ranks sharing a data-parallel index see identical documents; different DP replicas
        see different ones.

    Args:
        microbatch_docs: per DP replica, a list of micro-batches, each a list of document
            lengths.

    Returns:
        (accumulated_sum_L, accumulated_sum_L2, truth_sum_L, truth_sum_L2)
    """
    chunks = vpp or 1
    replicated_ranks = tp * cp * pp
    truth_l = truth_l2 = 0.0
    accum_l = accum_l2 = 0.0
    for replica in range(dp):
        for docs in microbatch_docs[replica]:
            truth_l += sum(docs)
            truth_l2 += sum(length * length for length in docs)
            # every replicating rank, every model chunk, adds the same micro-batch
            accum_l += sum(docs) * replicated_ranks * chunks
            accum_l2 += sum(length * length for length in docs) * replicated_ranks * chunks
    return accum_l, accum_l2, truth_l, truth_l2


@pytest.mark.parametrize("tp, cp, pp, vpp, dp", PARALLELISM_GRID)
def test_seqlen_stats_recover_the_global_batch_at_every_parallelism(
    monkeypatch, tp, cp, pp, vpp, dp
):
    """consume() must return the TRUE global-batch totals, whatever the parallelism.

    The accumulator is fed redundantly along four axes and must be divided by exactly those
    four -- no more (that would under-report) and no fewer (the VPP bug). DP is the one axis
    that must survive: different replicas hold different documents and their sum IS the
    global batch.

    Rather than assert the formula against a rebuilt copy of itself, ``_simulate_world_allreduce``
    reconstructs the accumulator from the per-rank call pattern and this compares against an
    independently summed truth.

    MEASURED (jobs 1516064 / 1516065, TP4 x PP4, VPP=2): the packed arm reported 717.5
    TFLOP/s/GPU against the control's 366.7 -- a factor 1.96 -- while tokens/s/GPU, computed
    from the closed form rather than this accumulator, agreed to 0.3%. At production's VPP=4
    it would have been ~4x.
    """
    import megatron.training.training as tr

    # Different documents per replica, so a formula that wrongly divided by dp would fail.
    microbatch_docs = [
        [[100 + replica, 200, 300 + m] for m in range(3)] for replica in range(dp)
    ]
    accum_l, accum_l2, truth_l, truth_l2 = _simulate_world_allreduce(
        tp, cp, pp, vpp, dp, microbatch_docs
    )

    stats = torch.tensor([accum_l, accum_l2], dtype=torch.float64)
    monkeypatch.setattr(tr, "_seqlen_stats_in_iteration", stats)
    monkeypatch.setattr(tr, "_seqlen_stats_active", True)
    monkeypatch.setattr(tr.mpu, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(tr.mpu, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(tr.mpu, "get_context_parallel_world_size", lambda: cp)
    monkeypatch.setattr(tr.mpu, "get_pipeline_model_parallel_world_size", lambda: pp)
    # Returns None, not 1, when VPP is off -- hence the `or 1` in the implementation.
    monkeypatch.setattr(tr.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: vpp)
    monkeypatch.setattr(tr.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(tr.torch.distributed, "all_reduce", lambda t: None)

    got_l, got_l2 = tr.consume_seqlen_stats_in_iteration()
    assert got_l == pytest.approx(truth_l)
    assert got_l2 == pytest.approx(truth_l2)


@pytest.mark.parametrize("tp, cp, pp, vpp, dp", PARALLELISM_GRID)
def test_seqlen_stats_token_count_matches_the_closed_form(monkeypatch, tp, cp, pp, vpp, dp):
    """The recovered token count must equal ``global_batch_size * seq_length`` exactly.

    This is the invariant the runtime guard checks, and the reason the guard is worth having:
    it is an INDEPENDENT count of a quantity Megatron already knows, so any error in the dedup
    factor shows up as a ratio at iteration 1 of a single run -- no A/B needed. GPTDataset's
    inter-document masking folds any shortfall into the last document, so every sample's
    cu_seqlens ends exactly at seq_length regardless of CP or of boundary snapping.
    """
    import megatron.training.training as tr

    seq_length, micro_batch_size, microbatches = 128, 2, 3
    # Documents that exactly tile each micro-batch pack, as the dataset guarantees.
    pack = micro_batch_size * seq_length
    microbatch_docs = [
        [[pack // 4, pack // 4, pack // 2] for _ in range(microbatches)] for _ in range(dp)
    ]
    accum_l, accum_l2, truth_l, _ = _simulate_world_allreduce(
        tp, cp, pp, vpp, dp, microbatch_docs
    )
    global_batch_size = micro_batch_size * microbatches * dp

    stats = torch.tensor([accum_l, accum_l2], dtype=torch.float64)
    monkeypatch.setattr(tr, "_seqlen_stats_in_iteration", stats)
    monkeypatch.setattr(tr, "_seqlen_stats_active", True)
    monkeypatch.setattr(tr.mpu, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(tr.mpu, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(tr.mpu, "get_context_parallel_world_size", lambda: cp)
    monkeypatch.setattr(tr.mpu, "get_pipeline_model_parallel_world_size", lambda: pp)
    monkeypatch.setattr(tr.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: vpp)
    monkeypatch.setattr(tr.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(tr.torch.distributed, "all_reduce", lambda t: None)

    got_l, _ = tr.consume_seqlen_stats_in_iteration()
    assert got_l == pytest.approx(global_batch_size * seq_length)
    assert got_l == pytest.approx(truth_l)


def test_token_count_guard_fires_on_a_wrong_dedup(monkeypatch):
    """The guard must actually catch the bug it was written for.

    Feeds it the VPP=4 overcount and asserts it warns, naming the ratio -- otherwise it is
    decoration.
    """
    import megatron.training.training as tr

    warned = []
    monkeypatch.setattr(tr, "_SEQLEN_STATS_MISMATCH_WARNED", False)
    monkeypatch.setattr(tr, "warn_rank_0", lambda msg, rank=None: warned.append(msg))
    args = argparse.Namespace(seq_length=8192, sft=False, rank=0)

    tr.check_seqlen_stats_token_count(4 * 64 * 8192, batch_size=64, args=args)
    assert len(warned) == 1
    assert "4.000" in warned[0]
    assert "TFLOP/s" in warned[0]


def test_token_count_guard_is_silent_when_correct_and_for_sft(monkeypatch):
    """No false alarms: the correct count must not warn, and SFT padding must not either.

    SFT's dataset legitimately pads, so ``sum(L_i)`` counts real tokens only and is SUPPOSED
    to come in under ``batch_size * seq_length``. Warning there would train people to ignore
    the warning.
    """
    import megatron.training.training as tr

    warned = []
    monkeypatch.setattr(tr, "warn_rank_0", lambda msg, rank=None: warned.append(msg))

    monkeypatch.setattr(tr, "_SEQLEN_STATS_MISMATCH_WARNED", False)
    tr.check_seqlen_stats_token_count(
        64 * 8192, batch_size=64, args=argparse.Namespace(seq_length=8192, sft=False, rank=0)
    )
    monkeypatch.setattr(tr, "_SEQLEN_STATS_MISMATCH_WARNED", False)
    tr.check_seqlen_stats_token_count(
        64 * 5000, batch_size=64, args=argparse.Namespace(seq_length=8192, sft=True, rank=0)
    )
    # BSHD: nothing was accumulated, so there is nothing to cross-check.
    monkeypatch.setattr(tr, "_SEQLEN_STATS_MISMATCH_WARNED", False)
    tr.check_seqlen_stats_token_count(
        None, batch_size=64, args=argparse.Namespace(seq_length=8192, sft=False, rank=0)
    )
    assert warned == []


def test_evaluation_does_not_pollute_the_next_training_iteration(monkeypatch):
    """evaluate() runs the same forward_step, so it accumulates -- and never consumes.

    Without reset_seqlen_stats() the validation microbatches land in the next TRAINING
    iteration's FLOPs number and that one iteration reports inflated throughput. Rare and
    metrics-only, i.e. exactly the kind of thing nobody would ever track down from a log.
    """
    import megatron.training.training as tr

    stats = torch.zeros(2, dtype=torch.float64)
    monkeypatch.setattr(tr, "_seqlen_stats_in_iteration", stats)
    monkeypatch.setattr(tr, "_seqlen_stats_active", False)

    # a validation micro-batch accumulates
    tr.update_seqlen_stats_from_cu_seqlens(torch.tensor([0, 4, 10], dtype=torch.int32))
    assert tr._seqlen_stats_active and stats.sum() > 0

    tr.reset_seqlen_stats()

    assert not tr._seqlen_stats_active
    assert stats.sum() == 0
    # and with the flag down, consume takes the BSHD path: no collective, closed-form defaults
    monkeypatch.setattr(
        tr.torch.distributed,
        "all_reduce",
        lambda *a, **k: pytest.fail("nothing was accumulated; no collective may fire"),
    )
    assert tr.consume_seqlen_stats_in_iteration() == (None, None)


def test_seqlen_stats_are_skipped_entirely_when_unpacked(monkeypatch):
    """BSHD runs must pay no collective and fall back to the closed form.

    Returning (None, None) is what tells num_floating_point_operations to use
    ``batch_size * seq_length`` -- so an unpacked run's reported FLOPs cannot be perturbed by
    any of this.
    """
    import megatron.training.training as tr

    monkeypatch.setattr(tr, "_seqlen_stats_active", False)
    monkeypatch.setattr(
        tr.torch.distributed,
        "all_reduce",
        lambda *a, **k: pytest.fail("unpacked runs must not issue the collective"),
    )
    assert tr.consume_seqlen_stats_in_iteration() == (None, None)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
#
# Every one of these is a configuration that would otherwise fail inside a kernel, or run and
# be silently slower or wrong. The point of asserting them at parse time is that a 512-node
# allocation should not be spent discovering them.


def _packing_args(**overrides):
    """A configuration that passes every guard, so a test can break exactly one thing."""
    from megatron.core.transformer.enums import AttnBackend

    defaults = dict(
        seq_length=8192,
        micro_batch_size=2,
        position_embedding_type="rope",
        attention_backend=AttnBackend.flash,
        spec=None,
        cuda_graph_impl="none",
        apply_rope_fusion=True,
        sequence_parallel=True,
        tensor_model_parallel_size=4,
        context_parallel_size=1,
        sft=False,
        create_attention_mask_in_dataloader=False,
        rank=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _validate(args):
    from megatron.training.arguments import _validate_packed_doc_attention

    return _validate_packed_doc_attention(args)


def test_the_reference_configuration_validates():
    """Production's shape: rope + fusion, TE flash, TP4 with sequence parallelism, CP1."""
    _validate(_packing_args())


@pytest.mark.parametrize(
    "overrides, match",
    [
        pytest.param(
            {"position_embedding_type": "learned_absolute"},
            "rotary or absent position embeddings",
            id="learned_absolute",
        ),
        pytest.param({"cuda_graph_impl": "local"}, "cuda-graph-impl none", id="cuda_graphs"),
        pytest.param({"spec": ["local"]}, "transformer_engine spec", id="local_spec"),
        pytest.param(
            {"seq_length": 8193, "sequence_parallel": True},
            "tensor-parallel ranks",
            id="sequence_parallel_indivisible",
        ),
        # sequence_parallel off, or its own divisibility guard fires first and this test
        # would pass for the wrong reason.
        pytest.param(
            {"context_parallel_size": 4, "seq_length": 8193, "sequence_parallel": False},
            "divisible by 2 \\* context_parallel_size",
            id="cp_indivisible",
        ),
        pytest.param(
            {"create_attention_mask_in_dataloader": True},
            "must not also",
            id="dense_mask_still_on",
        ),
    ],
)
def test_rejected_configurations(overrides, match):
    with pytest.raises(AssertionError, match=match):
        _validate(_packing_args(**overrides))


@pytest.mark.parametrize("backend", ["local", "unfused"])
def test_rejects_non_varlen_attention_backends(backend):
    """cu_seqlens is only honoured by TE's varlen kernels.

    The local and unfused backends take a dense mask, which the packed path does not build --
    so they would train with full cross-document attention and report nothing.
    """
    from megatron.core.transformer.enums import AttnBackend

    with pytest.raises(AssertionError, match="TE varlen backend"):
        _validate(_packing_args(attention_backend=AttnBackend[backend]))


def test_rejects_unfused_rope_because_it_is_slower_than_what_it_replaces():
    """An error, not a warning, and the reason is performance rather than correctness.

    Without apply_rope_fusion the thd path is _apply_rotary_pos_emb_thd, which reads
    cu_seqlens back to the host once per ATTENTION CALL -- roughly 60 stream drains per
    microbatch at 32B, against the one per ITERATION the scatter costs. Numerically fine and
    slower than the dense mask it replaces, which is not a trade anyone would take knowingly.
    """
    with pytest.raises(AssertionError, match="UNFUSED thd rope path"):
        _validate(_packing_args(apply_rope_fusion=False))


def test_yarn_is_rejected_with_its_own_explanation():
    """yarn cannot satisfy the fusion requirement, so pointing at --apply-rope-fusion lies.

    validate_args force-disables apply_rope_fusion for every position_embedding_type other
    than exactly 'rope', so the user has no way to turn it back on.
    """
    with pytest.raises(AssertionError, match="yarn cannot be used"):
        _validate(_packing_args(position_embedding_type="yarn", apply_rope_fusion=False))


def test_position_embedding_none_is_allowed_without_fusion():
    """No rope at all means no thd rope path to be slow; cu_seqlens still masks documents."""
    _validate(_packing_args(position_embedding_type="none", apply_rope_fusion=False))


def test_context_parallelism_is_accepted_when_the_pack_divides():
    """CP>1 is supported -- via boundary snapping -- so it must not be rejected outright."""
    _validate(_packing_args(context_parallel_size=4))
