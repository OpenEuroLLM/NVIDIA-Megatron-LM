# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""OELLM PATCH: scatter cu_seqlens to every pipeline stage, once per iteration.

WHAT THIS REPLACES
------------------
Upstream 0.19 implements ``--dataloader-inter-document-masking`` correctly at the
dataloader: ``gpt_dataset.py`` builds ``cu_seqlens`` from the document index, resets
position ids per document, and ``core/utils.py`` merges per-sample ``cu_seqlens`` into
one packed stream and splits it across context-parallel ranks. None of that is touched
here and none of it needs to be.

What upstream does NOT get right is how those few dozen integers reach the pipeline
stages that never read data. Every transformer layer needs ``cu_seqlens`` -- it is what
makes the varlen kernel skip the off-diagonal blocks and what restarts RoPE per document
-- but only the FIRST and LAST pipeline stages read the dataloader. Upstream closes that
gap by making every stage read (``has_cu_seqlens`` widens the read in ``get_batch``,
backed by ``is_dataset_built_on_rank(..., is_packed_sequence=True)``). That is precisely
the "local" mode this project already built, measured and DELETED:

  * as shipped, upstream sets ``is_packed_sequence`` for SFT only, so with masking on and
    PP>1 a middle stage is asked to read a dataset it never built --
    ``TypeError: 'NoneType' object is not an iterator`` (jobs 1511085 / 1511146, ranks
    20/24 at PP=4). PP=1 hides it, because then every rank is an endpoint stage.
  * setting the flag correctly fixes that crash and reintroduces the reason local mode
    was deleted: one dataset index map per model chunk, ~412 GB against ~472 GB of node
    RAM at PP=4/VPP=4, rank 0 dying in the dataset build while the survivors hang waiting
    on it (job 1497412).

SCATTER instead: the stages that already read data derive nothing new -- they simply
publish the ``cu_seqlens`` the dataloader handed them -- and ONE broadcast over the
model-parallel group (which spans both TP and PP) serves every rank for the whole
iteration. Cost measured at 512 nodes: -2.8% against an unmasked control, inside a
+-3.2% noise floor (job 1498007). It scales with the MICROBATCH COUNT, not with the
address space, so large DP (small M) means it is free exactly where it matters.

WHY THE COLLECTIVE CANNOT LIVE IN get_batch
-------------------------------------------
Broadcasting per-microbatch from inside ``forward_step`` DEADLOCKS -- measured, job
1494386:

    rank 0 (first stage) enters forward_step -> get_batch -> broadcast, and blocks
           waiting for every rank in the group;
    rank N (last stage) is in recv_forward, blocked on rank N-1 <- ... <- rank 0;
    rank 0 cannot produce those activations because it is stuck in the broadcast.

Making the broadcast async does not help: rank 1 still has to WAIT for cu_seqlens before
it can run its own attention, so ranks 2..N never reach their matching post and the
circular wait re-forms one stage further down. The fix is placement, not asynchrony.
This module runs before the schedule starts, when no p2p is in flight, and issues one
broadcast covering all microbatches of the iteration.

WHY A PER-CHUNK QUEUE IS THE RIGHT INDEX
----------------------------------------
``get_batch`` is not told which microbatch it is handling. It does not need to be: within
a model chunk the schedule visits microbatches in strictly increasing order -- for the PP2
N3M5 VP2 table in ``schedules.py``, chunk 0 sees virtual ids 0,1,2,6,7 which are microbatch
ids 0,1,2,3,4 -- so popping a per-chunk queue in call order reproduces the microbatch index
exactly. And because ``get_schedule_table`` depends only on ``num_microbatches``, the chunk
count and ``microbatch_group_size_per_vp_stage`` -- never on pipeline rank -- every rank
pops the same entry on its k-th call.

WHAT TRAVELS
------------
Three things per microbatch, and all three are needed on middle stages:
``cu_seqlens`` (document boundaries), ``cu_seqlens_padded`` (what TE's context-parallel
thd kernels index with; ``None`` at CP=1) and ``max_seqlen`` (TE takes it as a Python int
at kernel launch). They are packed into two collectives per iteration -- a fixed-size
header giving the three lengths per microbatch, then one exactly-sized payload. A single
worst-case-sized buffer would have been one collective, but the worst case is
``seq_length * micro_batch_size + 1`` entries per array (one document per token), i.e. 131 KB
per microbatch at 8192x2 whether or not the documents are actually that short. The header
costs one extra broadcast of a few hundred bytes and makes the payload proportional to the
real document count.
"""

from typing import NamedTuple, Optional

import torch

from megatron.core import parallel_state


def is_scattered(args) -> bool:
    """Is inter-document masking being served by the per-iteration scatter on this run?

    True for plain pretraining with masking on -- the case this module exists for, where the
    dataset is 15 T tokens and building its index on every pipeline stage is what runs a node
    out of RAM (job 1497412).

    False for SFT, which keeps upstream's per-stage read untouched. Two reasons, and either
    alone would be enough: an SFT dataset is small enough that building it everywhere costs
    nothing, and SFTDataset emits a GENUINE padded layout (real padding tokens, a
    cu_seqlens_padded that differs from cu_seqlens) rather than the boundary-snapped one
    apply_cp_document_padding produces, so the two must not share a code path.
    """
    return bool(getattr(args, 'dataloader_inter_document_masking', False)) and not bool(
        getattr(args, 'sft', False)
    )


class PackedSeqMetadata(NamedTuple):
    """The per-microbatch quantities every pipeline stage needs for thd attention.

    ``cu_seqlens`` and ``cu_seqlens_padded`` keep the leading batch dimension that
    ``get_batch`` returns them with (shape ``(1, N)``); ``forward_step`` squeezes it before
    building ``PackedSeqParams``. ``max_seqlen`` is a 1-element CPU tensor rather than a
    device one on purpose: ``forward_step`` calls ``.item()`` on it once per microbatch, and
    on a device tensor that drains the stream -- the exact per-microbatch host sync this
    module exists to reduce to one per iteration.
    """

    cu_seqlens: torch.Tensor
    cu_seqlens_padded: Optional[torch.Tensor]
    max_seqlen: torch.Tensor


# ---------------------------------------------------------------------------
# CONTEXT PARALLELISM
# ---------------------------------------------------------------------------


def snap_cu_seqlens_to_multiple(cu_seqlens: torch.Tensor, multiple: int) -> torch.Tensor:
    """Move document boundaries to the nearest multiple of ``multiple``, keeping the total.

    WHY THIS EXISTS. Transformer Engine's context-parallel thd path asserts
    ``"cu_seqlens_padded is required for THD format!"`` (context_parallel.py:419) and
    partitions each document with ``thd_get_partitioned_indices``, which requires every
    document length to be divisible by ``2 * cp_size`` -- the zigzag load balance splits each
    document into ``2 * cp_size`` chunks and hands rank r chunks r and ``2*cp-1-r``.
    ``GPTDataset`` emits neither: its documents are whatever length the corpus made them, and
    it never produces ``cu_seqlens_padded`` at all (only the SFT path does).

    Two ways to satisfy TE. Insert real padding tokens after each document, which is exact
    but makes the packed length vary per microbatch and so forces ``variable_seq_lengths``
    on the pipeline; or MOVE the boundaries, which keeps the token stream and every tensor
    shape untouched. This does the latter.

    WHAT IT COSTS. A boundary moves by less than ``multiple`` tokens, so at most
    ``multiple - 1`` tokens per boundary end up attending into the neighbouring document.
    At CP=2 (multiple 4) with 8192-token samples and ~1000-token documents that is under
    0.3% of tokens -- against the 100% cross-document attention this feature exists to
    remove. It is an approximation and it is bounded; it is not silent (arguments.py warns
    once at rank 0 when CP>1).

    The first and last entries are pinned: ``cu_seqlens[0]`` must stay 0 and
    ``cu_seqlens[-1]`` must stay the packed length, or the segments would no longer cover
    the tokens. That is safe because the packed length is already required to be divisible
    by ``2 * cp_size`` (arguments.py asserts it).

    Args:
        cu_seqlens: 1-D int32 cumulative lengths, ``cu_seqlens[0] == 0``.
        multiple: ``2 * cp_size``.

    Returns:
        1-D int32 tensor, non-decreasing, every entry divisible by ``multiple``, with
        zero-length segments removed. May be SHORTER than the input: two boundaries that
        round to the same multiple describe a document shorter than ``multiple``, which
        cannot survive the split and is absorbed into its neighbour.
    """
    assert cu_seqlens.dim() == 1, f"expected 1-D cu_seqlens, got shape {tuple(cu_seqlens.shape)}"
    assert multiple >= 1, f"multiple must be positive, got {multiple}"
    if multiple == 1:
        return cu_seqlens

    total = cu_seqlens[-1]
    # Round to NEAREST rather than up: rounding up biases every boundary in one direction
    # and, on a pack whose documents are all shorter than `multiple`, would push the
    # cumulative sum past the total.
    snapped = ((cu_seqlens + multiple // 2) // multiple) * multiple
    # Pin the endpoints. `total` is divisible by `multiple` already, but computing it from
    # the input rather than trusting the rounding keeps this correct if that ever changes.
    snapped[0] = 0
    snapped[-1] = total
    # `unique` sorts (a no-op here, the input is non-decreasing and rounding preserves that)
    # and drops duplicates, which is what removes the collapsed zero-length segments. TE
    # requires every segment to have length >= 1.
    return torch.unique(snapped).to(torch.int32)


def apply_cp_document_padding(batch: dict, cp_size: int) -> dict:
    """Give ``batch`` the ``cu_seqlens_padded`` that TE's context-parallel thd path needs.

    Called on the reading ranks between ``flatten_batch_for_packed_sequences`` (which merges
    the per-sample ``cu_seqlens`` into one packed stream) and ``get_batch_on_this_cp_rank``
    (which does the zigzag split and reads ``cu_seqlens_padded`` to do it). A no-op at CP=1,
    where TE needs neither the padded array nor divisible document lengths.

    ``max_seqlen`` is recomputed from the snapped boundaries rather than carried over: it is
    what TE passes to the kernel as the longest segment, and snapping can lengthen a segment
    by up to ``multiple - 1``. An understated ``max_seqlen`` is not a rounding difference,
    it is out-of-bounds work.
    """
    if cp_size <= 1 or batch.get('cu_seqlens') is None:
        return batch

    # (1, N) -> (N,) for the arithmetic, then back: the batch dict's convention is a leading
    # batch dim, which forward_step squeezes.
    cu_seqlens = snap_cu_seqlens_to_multiple(batch['cu_seqlens'][0], 2 * cp_size)
    batch['cu_seqlens'] = cu_seqlens.unsqueeze(0)
    # Equal, not merely both present: no real padding tokens were inserted, the boundaries
    # moved instead, so the "padded" layout IS the layout.
    batch['cu_seqlens_padded'] = cu_seqlens.unsqueeze(0).clone()
    batch['max_seqlen'] = (cu_seqlens[1:] - cu_seqlens[:-1]).max().reshape(1)
    return batch


# ---------------------------------------------------------------------------
# PREFETCH / BROADCAST
# ---------------------------------------------------------------------------

# Set by the training script (pretrain_gpt.py) so train_step can drive the prefetch without
# training.py importing the model script.
_PREFETCH_HOOK = None


def register_prefetch_hook(hook):
    """Register callable(data_iterator, vp_stage, num_microbatches, derive) -> None."""
    global _PREFETCH_HOOK
    _PREFETCH_HOOK = hook


def prefetch_hook():
    """The hook registered by the training script, or None."""
    return _PREFETCH_HOOK


# {vp_stage: [batch or None, ...]} -- prefetched batches, only on ranks that read data.
_BATCH_STASH = {}
# {vp_stage: [PackedSeqMetadata, ...]} -- one entry per microbatch, on EVERY rank.
_METADATA_STASH = {}


def reset():
    """Drop any prefetched state. Call between iterations."""
    _BATCH_STASH.clear()
    _METADATA_STASH.clear()


def is_primed(vp_stage):
    """Has this chunk been prefetched for the current iteration?"""
    return _key(vp_stage) in _METADATA_STASH


def _key(vp_stage):
    return -1 if vp_stage is None else vp_stage


def share_metadata_from(vp_stage, source_vp_stage):
    """Give chunk ``vp_stage`` the metadata already published for ``source_vp_stage``.

    WHY THIS IS CORRECT, AND WHY IT IS NEEDED.

    cu_seqlens depends on the MICROBATCH, not on the model chunk. Every virtual stage
    processes microbatch m on the same underlying tokens -- m flows through virtual stage 0,
    then 1, then 2 -- so the document boundaries a chunk sees for its j-th call are the ones
    the first stage saw for ITS j-th call. Within a chunk the schedule visits microbatches in
    strictly increasing order, and ``get_schedule_table`` is rank-independent, so "j-th call"
    means the same microbatch everywhere.

    Without this, the broadcast is BROKEN at VPP>1 and fails loudly at the first step:
    ``prefetch_iteration`` only fills the buffer inside ``if reads_data:``, and reads_data is
    ``is_first_or_last_pipeline_stage(vp_stage)`` -- evaluated PER CHUNK. The model-parallel
    source rank hosts chunks 0..VPP-1 but is the first stage only for chunk 0, so for every
    other chunk it never wrote anything, zeros were broadcast, the entry count came out 0 and
    the cu_seqlens slice was EMPTY:

        fused_rope.cu:477 Assertion failed: cu_seqlens != nullptr, required for THD format

    measured at 512 nodes, PP=4/VPP=4, job 1497767, 1865 ranks. It never showed earlier
    because every test of it so far ran VPP=1, where the source IS always the first stage.

    Publishing once and sharing also drops the collective count from VPP per iteration to
    one, and costs no extra dataset reads -- which matters, because reading on more chunks is
    exactly what exhausted node memory in the deleted per-chunk mode at this layout.
    """
    source = _key(source_vp_stage)
    assert source in _METADATA_STASH, (
        f"cannot share cu_seqlens from chunk {source_vp_stage}: it was not prefetched. "
        "The publishing chunk must be prefetched first."
    )
    # The same list object per chunk would be popped by whichever chunk ran first, so copy
    # it. The tensors inside are read-only and safely shared.
    _METADATA_STASH[_key(vp_stage)] = list(_METADATA_STASH[source])


# Header columns, one row per microbatch: how many entries each array has, and the longest
# document. int64 because torch.distributed.broadcast wants a single dtype per collective and
# these are counts, not the int32 payload.
_HEADER_WIDTH = 3


def prefetch_iteration(
    data_iterator,
    vp_stage,
    num_microbatches,
    fetch_batch,
    reads_data,
    derive=True,
):
    """Publish cu_seqlens for a whole iteration across the model-parallel group.

    Args:
        data_iterator: this chunk's iterator (None on ranks that do not read data)
        vp_stage: model chunk index, or None without VPP
        num_microbatches: microbatches in this iteration
        fetch_batch: callable(data_iterator) -> batch dict, already on device and already
            through the TP broadcast and the CP split. Must carry 'cu_seqlens',
            'cu_seqlens_padded' and 'max_seqlen'.
        reads_data: whether this rank's pipeline stage pulls from the dataloader
        derive: whether this chunk publishes cu_seqlens. Only the chunk whose first stage IS
            the model-parallel source rank can (see share_metadata_from); the rest still
            fetch their batches but take the metadata from that chunk.

    MUST be called by every rank of the model-parallel group, the same number of times per
    iteration, and BEFORE the pipeline schedule runs. ``derive`` must agree across the group
    for a given chunk, or the collectives will not match up.

    ============ WHY BEFORE THE SCHEDULE, AND NOT INSIDE forward_step ============
    Broadcasting cu_seqlens from inside get_batch DEADLOCKS as soon as the group spans
    pipeline stages. Measured, not theorised (job 1494386, hung at the first training step,
    cancelled):

        rank 0 (first stage) enters forward_step -> get_batch -> broadcast, and blocks
               waiting for every rank in the group;
        rank N (last stage) is in recv_forward, blocked on rank N-1 <- ... <- rank 0;
        rank 0 cannot produce those activations because it is stuck in the broadcast.

    A circular wait between the collective and the p2p activation chain. It is NOT about
    microbatch ordering -- the interleaved schedule's tables are rank-independent, so the
    k-th forward is the same (chunk, microbatch) on every rank and the CONTENT would have
    been correct. A blocking collective simply cannot sit inside a pipeline whose stages are
    waiting on each other's activations. Hoisting it out here, where no p2p is in flight, is
    what makes one broadcast for all of TP and PP possible at all.
    =============================================================================
    """
    key = _key(vp_stage)
    device = torch.cuda.current_device()
    group = parallel_state.get_model_parallel_group()
    src_rank = parallel_state.get_model_parallel_src_rank()
    is_source = torch.distributed.get_rank() == src_rank

    batches = []
    header = torch.zeros(num_microbatches, _HEADER_WIDTH, dtype=torch.int64, device=device)
    payload_parts = []

    if reads_data:
        # Consume the whole iteration up front; get_batch then serves from this stash instead
        # of the iterator, so nothing is consumed twice.
        for index in range(num_microbatches):
            batch = fetch_batch(data_iterator)
            batches.append(batch)
            if not is_source:
                continue
            cu_seqlens = batch['cu_seqlens']
            assert cu_seqlens is not None, (
                "packed-doc-attention prefetch: the model-parallel source rank read a batch "
                "with no 'cu_seqlens'. Inter-document masking must be on in the dataset "
                "config, not only on the command line."
            )
            # (1, N) as the batch dict carries it -> (N,) for the wire.
            cu_seqlens = cu_seqlens.reshape(-1)
            padded = batch.get('cu_seqlens_padded')
            padded = None if padded is None else padded.reshape(-1)
            header[index, 0] = cu_seqlens.numel()
            header[index, 1] = 0 if padded is None else padded.numel()
            header[index, 2] = batch['max_seqlen'].reshape(-1)[0]
            payload_parts.append(cu_seqlens.to(torch.int32))
            if padded is not None:
                payload_parts.append(padded.to(torch.int32))

    _BATCH_STASH[key] = batches
    if not derive:
        # Batches still had to be fetched above (the last-stage chunk needs its labels), but
        # the metadata comes from the publishing chunk -- the caller wires that with
        # share_metadata_from. Returning before the collectives is what keeps the group in
        # step: every rank skips them for this chunk, not just the ones that read.
        return

    broadcasting = group is not None and group.size() > 1
    if broadcasting:
        torch.distributed.broadcast(header, src_rank, group=group)

    # THE ONE HOST SYNC PER ITERATION. Everything downstream -- how big the payload buffer is,
    # where each microbatch's slice starts, and the Python int TE wants for max_seqlen -- is
    # host-side information, so it has to come back once. The per-microbatch derivation this
    # replaced needed one sync EACH.
    header_rows = header.tolist()

    total_entries = sum(row[0] + row[1] for row in header_rows)
    assert total_entries > 0, (
        f"packed-doc-attention: chunk {vp_stage} published an EMPTY cu_seqlens for every "
        "microbatch. The model-parallel source rank did not read data for this chunk, so it "
        "never filled the buffer. Only the chunk whose first stage is the source may publish; "
        "the others must use share_metadata_from()."
    )

    if broadcasting:
        # Non-source ranks contributed nothing, so they receive into zeros of the size the
        # header just told them. `payload_parts` is non-empty exactly on the source.
        payload = (
            torch.cat(payload_parts)
            if payload_parts
            else torch.zeros(total_entries, dtype=torch.int32, device=device)
        )
        torch.distributed.broadcast(payload, src_rank, group=group)
    else:
        # A model-parallel group of one: the only member is the source, so there is nothing
        # to broadcast and the parts are guaranteed to be there.
        assert is_source and payload_parts, (
            "packed-doc-attention: the model-parallel group has one member but this rank is "
            "not its source. parallel_state is inconsistent."
        )
        payload = torch.cat(payload_parts)

    stash = []
    offset = 0
    for n_cu, n_padded, max_seqlen in header_rows:
        assert n_cu > 0, (
            f"packed-doc-attention: chunk {vp_stage} published an empty cu_seqlens for one "
            "microbatch. See share_metadata_from()."
        )
        cu_seqlens = payload[offset : offset + n_cu].unsqueeze(0)
        offset += n_cu
        padded = None
        if n_padded:
            padded = payload[offset : offset + n_padded].unsqueeze(0)
            offset += n_padded
        stash.append(
            PackedSeqMetadata(
                cu_seqlens=cu_seqlens,
                cu_seqlens_padded=padded,
                # CPU on purpose: see PackedSeqMetadata. The value is already host-side here,
                # so materialising it as a device tensor would only buy back the sync.
                max_seqlen=torch.tensor([max_seqlen], dtype=torch.int32),
            )
        )
    _METADATA_STASH[key] = stash


def pop(vp_stage, reads_data):
    """Return ``(batch_or_None, metadata)`` for this chunk's next microbatch."""
    key = _key(vp_stage)
    assert key in _METADATA_STASH, (
        f"packed-doc-attention: chunk {vp_stage} was not prefetched this iteration. "
        "training.maybe_prefetch_cu_seqlens() must run before the pipeline schedule."
    )
    metadata = _METADATA_STASH[key].pop(0)
    batch = _BATCH_STASH[key].pop(0) if reads_data else None
    return batch, metadata
