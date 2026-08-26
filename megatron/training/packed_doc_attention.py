# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""OELLM PATCH: per-iteration cu_seqlens prefetch for --packed-doc-attention.

Every pipeline stage needs cu_seqlens for its own transformer layers, but only the endpoint
stages read the dataloader (and within those, only TP rank 0). This closes that gap: the
reading stage derives cu_seqlens for the WHOLE iteration up front and broadcasts it in ONE
collective over the model-parallel group, which spans both TP and PP, before the pipeline
schedule starts.

Cost is the prefetch itself, which cannot hide behind compute, so it scales with the
MICROBATCH COUNT: +14.7-34.1% at M=1024 (DP=1, jobs 1497045/46), no measurable cost at M=16
(DP=128, job 1498007). Large DP means small M, so production pays nothing.

There used to be a second mode, selected by a --packed-doc-attention-scatter flag that no
longer exists: every stage read the dataloader and derived cu_seqlens itself, per model
chunk. Its cost scaled the other way -- with the ADDRESS SPACE those builds map, chunks x
(size of the dataset index cache), independent of M -- which made it cheaper at 2 nodes and
fatal at production shape: PP=4/VPP=4 against a 704 GB cache maps ~412 GB into ~472 GB of
CPU RAM, and rank 0 died in the dataset build while the surviving ranks hung waiting on it
(job 1497412). It was deleted. The lesson worth keeping is that the two costs moved in
OPPOSITE directions with scale, so the 2-node measurement said nothing about 512.

WHY THE COLLECTIVE CANNOT LIVE IN get_batch
-------------------------------------------
Broadcasting per-microbatch from inside forward_step DEADLOCKS -- measured, job 1494386:

    rank 0 (first stage) enters forward_step -> get_batch -> broadcast, and blocks
           waiting for every rank in the group;
    rank N (last stage) is in recv_forward, blocked on rank N-1 <- ... <- rank 0;
    rank 0 cannot produce those activations because it is stuck in the broadcast.

Making the broadcast async does not help: rank 1 still has to WAIT for cu_seqlens before
it can run its own attention, so ranks 2..N never reach their matching post and the
circular wait re-forms one stage further down.

The fix is placement, not asynchrony. This module runs before the schedule starts, when no
p2p is in flight, and issues ONE broadcast covering all microbatches of the iteration.

WHY A PER-CHUNK QUEUE IS THE RIGHT INDEX
----------------------------------------
get_batch is not told which microbatch it is handling. It does not need to be: within a
model chunk the schedule visits microbatches in strictly increasing order -- for the PP2
N3M5 VP2 table in schedules.py, chunk 0 sees virtual ids 0,1,2,6,7 which are microbatch ids
0,1,2,3,4 -- so popping a per-chunk queue in call order reproduces the microbatch index
exactly. And because get_schedule_table (schedules.py:1045) depends only on
num_microbatches, the chunk count and microbatch_group_size_per_vp_stage -- never on
pipeline rank -- every rank pops the same entry on its k-th call.
"""

import numpy as np
import torch

from megatron.core import parallel_state

# ---------------------------------------------------------------------------
# CPU DERIVATION
# ---------------------------------------------------------------------------
# cu_seqlens used to be derived on the GPU from the already-uploaded tokens, which cost TWO
# device->host readbacks per microbatch per rank:
#
#   1. max_seqlen -- TE takes it as a PYTHON INT at kernel launch, so
#      `int((cu_seqlens[1:] - cu_seqlens[:-1]).max())` had to drain the stream;
#   2. n_entries   -- the cu_seqlens slice length is host-side information, so unpacking the
#      broadcast buffer with `.tolist()` drained it again.
#
# A readback is not a cheap copy: the CPU cannot proceed until every kernel queued ahead of
# it has retired, so it destroys CPU run-ahead once per microbatch, right at the top of
# forward_step. At 512n production that is M=16 stalls per iteration; in a DP=1 debug config
# it is M=1024.
#
# Deriving from the CPU tokens BEFORE the H2D copy makes both quantities host-side by
# construction: max_seqlen and n_entries are plain Python ints, the exact-size cu_seqlens
# goes up in one async copy, and nothing is ever read back.
#
# THE PART THAT IS NOT OBVIOUS is how the ranks that never read the dataloader get the
# boundaries. Only TP rank 0 of an endpoint pipeline stage reads; everyone else -- the other
# TP ranks, and every middle pipeline stage -- has no host-side tokens at all, and a middle
# stage gets no token broadcast either (utils.get_batch_on_this_tp_rank only broadcasts on
# the first/last stage). So cu_seqlens for the WHOLE iteration is broadcast once over the
# model-parallel group (which spans both TP and PP) before the schedule starts, and every
# rank serves its microbatches out of that. It has to happen before the schedule: a
# collective inside forward_step deadlocks against the in-flight p2p activation chain
# (job 1494386).
#
# A per-microbatch alternative used to exist ("local" mode): every pipeline stage read the
# dataloader and derived its own cu_seqlens, with a host-side gloo TP group carrying the
# header to the other TP ranks. It was deleted -- at PP=4/VPP=4 it mapped the dataset
# indices once per model chunk, ~412 GB against ~472 GB of node RAM, and died in the
# dataset build (job 1497412). It also cost one collective and one host sync per
# microbatch, where this costs one of each per ITERATION.


# Resolved once. Lives here rather than in the training script because the derivation now
# happens in utils.get_batch_on_this_tp_rank, and utils must not import pretrain_gpt.
_EOD_TOKEN_ID = None


def get_eod_token_id():
    """End-of-document token id, building the tokenizer once on first use.

    Resolved exactly the way core_gpt_dataset_config_from_args() resolves it, so the
    boundaries handed to the attention kernel are the ones the dataset actually used.
    """
    global _EOD_TOKEN_ID
    if _EOD_TOKEN_ID is None:
        # Imported lazily: this module is pulled in from megatron.training.utils, which is
        # itself imported while megatron.training is still initialising.
        #
        # THE IMPORT PATH IS LOAD-BEARING. It must be the one
        # core_gpt_dataset_config_from_args() uses (pretrain_gpt.py), because that is the
        # tokenizer whose eod the DATASET used when it laid the documents out.
        # megatron.training.tokenizer also exports a build_tokenizer, and it is a DIFFERENT
        # function -- taking it would risk a different eod id, hence cu_seqlens cutting the
        # pack at the wrong offsets, with no error anywhere.
        from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
        from megatron.training import get_args, get_tokenizer

        args = get_args()
        tokenizer = get_tokenizer() if args.legacy_tokenizer else build_tokenizer(args)
        _EOD_TOKEN_ID = tokenizer.eod
    return _EOD_TOKEN_ID


def derive_cu_seqlens_cpu(tokens, eod_id):
    """Document boundaries of a [b, s] CPU token tensor, computed host-side.

    Returns ``(cu_seqlens, max_seqlen)`` with cu_seqlens an int32 numpy array and max_seqlen
    a Python int. Mirrors the GPU derivation exactly -- same EOD-plus-one convention, same
    injected sample boundaries, same unique() -- so the two are interchangeable and the
    equivalence test can compare them elementwise.

    `tokens` is consumed as ONE flattened pack of b*s tokens, so a sample boundary closes a
    document too; that is what makes the folded [1, b*s] micro-batch legitimate.
    """
    if isinstance(tokens, torch.Tensor):
        assert not tokens.is_cuda, (
            "derive_cu_seqlens_cpu wants host tokens -- passing a CUDA tensor would "
            "reintroduce the readback this function exists to remove."
        )
        flat = tokens.detach().numpy().reshape(-1)
        batch_size, seq_length = tokens.shape
    else:
        flat = np.asarray(tokens).reshape(-1)
        batch_size, seq_length = np.asarray(tokens).shape

    # A document ends *after* its EOD token, hence the +1.
    ends = np.flatnonzero(flat == eod_id) + 1
    sample_ends = np.arange(seq_length, batch_size * seq_length + 1, seq_length)
    # unique() sorts and drops the duplicate a sample ending on EOD would produce. Every
    # resulting segment has length >= 1, which is what TE requires of cu_seqlens.
    bounds = np.unique(np.concatenate((ends, sample_ends)))
    cu_seqlens = np.concatenate((np.zeros(1, dtype=bounds.dtype), bounds)).astype(np.int32)

    # The TRUE longest document, not args.seq_length. Free here (it is a numpy reduction on
    # a few dozen ints); on the GPU it was one of the two stream drains. Keeping it true also
    # removes a correctness footgun on the UNFUSED rope path: rope_utils._apply_rotary_pos_
    # emb_thd switches to offset mapping when freqs.size(0) == cu_seqlens[-1], and with the
    # true maximum that can only happen when the pack holds a SINGLE segment, where offset
    # mapping is equivalent. An inflated max_seqlen would trip it with many documents, which
    # silently stops RoPE restarting per document.
    max_seqlen = int(np.diff(cu_seqlens).max())
    return cu_seqlens, max_seqlen


# Key under which get_batch_on_this_tp_rank puts (cu_seqlens, max_seqlen) into the batch
# dict, on TP rank 0 only -- the one rank that reads the dataloader, and therefore the only
# one holding the host tokens the boundaries are derived from. Consumers pop it.
#
# THE VALUE TRAVELS WITH THE DATA IT DESCRIBES, deliberately. An earlier version handed it
# over through a module global, which worked only because every read happened to be
# immediately preceded by a write in the same call chain -- an invariant nothing enforced.
# Prefetching one microbatch ahead, batching reads, or a second thread would each have
# turned it into "consumed from the wrong microbatch": right shape, right dtype, wrong
# document boundaries, and no error anywhere. Carrying it in the batch removes the ordering
# question rather than documenting it, and matches how Megatron moves every other
# per-microbatch quantity (Megatron's module globals are process-lifetime config -- args,
# tokenizer, timers -- never per-step payload).
#
# SAFE TO ADD TO THE DICT because the only consumer that walks it, core.utils
# .get_batch_on_this_cp_rank, does so ONLY when cp_size > 1 -- and --packed-doc-attention
# asserts context_parallel_size == 1 (arguments.py). If CP support is ever added, that loop
# has to skip this key; it would fail loudly (`.view()` on a tuple), not silently.
# The positional `*batch.values()` unpackings (pretrain_gpt non-packed path,
# pretrain_mamba) never see it: it is added only when packing is on, and neither runs then.
CU_SEQLENS_KEY = "packed_doc_attention_cu_seqlens"


def pop_cu_seqlens(batch):
    """Take (cu_seqlens, max_seqlen) out of a batch, or None if this rank did not read.

    Popped rather than read so the entry never reaches the model or the reshape below it.
    """
    if batch is None:
        return None
    return batch.pop(CU_SEQLENS_KEY, None)


# Set by the training script (pretrain_gpt.py) so train_step can drive the prefetch without
# training.py importing the model script.
_PREFETCH_HOOK = None


def register_prefetch_hook(hook):
    """Register callable(data_iterator, vp_stage, num_microbatches) -> None."""
    global _PREFETCH_HOOK
    _PREFETCH_HOOK = hook


def prefetch_hook():
    return _PREFETCH_HOOK


# {vp_stage: [batch or None, ...]} -- prefetched batches, only on ranks that read data.
_BATCH_STASH = {}
# {vp_stage: [(cu_seqlens, max_seqlen), ...]} -- one entry per microbatch, on every rank.
_CU_SEQLENS_STASH = {}


def reset():
    """Drop any prefetched state. Call between iterations."""
    _BATCH_STASH.clear()
    _CU_SEQLENS_STASH.clear()


def is_primed(vp_stage):
    """Has this chunk been prefetched for the current iteration?"""
    return _key(vp_stage) in _CU_SEQLENS_STASH


def _key(vp_stage):
    return -1 if vp_stage is None else vp_stage


def share_cu_seqlens_from(vp_stage, source_vp_stage):
    """Give chunk `vp_stage` the cu_seqlens already derived for `source_vp_stage`.

    WHY THIS IS CORRECT, AND WHY IT IS NEEDED.

    cu_seqlens depends on the MICROBATCH, not on the model chunk. Every virtual stage
    processes microbatch m on the same underlying tokens -- m flows through virtual stage 0,
    then 1, then 2 -- so the document boundaries a chunk sees for its j-th call are the ones
    the first stage saw for ITS j-th call. Within a chunk the schedule visits microbatches in
    strictly increasing order, and get_schedule_table (schedules.py:1045) is rank-independent,
    so "j-th call" means the same microbatch everywhere.

    Without this, the broadcast is BROKEN at VPP>1 and fails loudly at the first step:
    prefetch_iteration only fills the broadcast buffer inside `if reads_data:`, and
    reads_data is `is_first_or_last_pipeline_stage(vp_stage)` -- evaluated PER CHUNK. The
    model-parallel source rank hosts chunks 0..VPP-1 but is the first stage only for chunk 0,
    so for every other chunk it never wrote anything, torch.zeros was broadcast, n_entries
    came out 0, and the cu_seqlens slice was EMPTY:

        fused_rope.cu:477 Assertion failed: cu_seqlens != nullptr, required for THD format

    measured at 512 nodes, PP=4/VPP=4, job 1497767, 1865 ranks. It never showed earlier
    because every test of it so far ran VPP=1, where the source IS always the first stage.

    Deriving once and sharing also drops the collective count from VPP per iteration to
    one, and costs no extra dataset reads -- which matters, because reading on more chunks
    is exactly what exhausted node memory in the deleted per-chunk mode at this layout.
    """
    source = _key(source_vp_stage)
    assert source in _CU_SEQLENS_STASH, (
        f"cannot share cu_seqlens from chunk {source_vp_stage}: it was not prefetched. "
        "The deriving chunk must be prefetched first."
    )
    # Same list object per chunk would be popped by whichever chunk ran first, so copy it.
    _CU_SEQLENS_STASH[_key(vp_stage)] = list(_CU_SEQLENS_STASH[source])


def prefetch_iteration(
    data_iterator,
    vp_stage,
    num_microbatches,
    packed_length,
    fetch_batch,
    reads_data,
    derive=True,
):
    """Derive cu_seqlens for a whole iteration and share it across the model-parallel group.

    Args:
        data_iterator: this chunk's iterator (None on ranks that do not read data)
        vp_stage: model chunk index, or None without VPP
        num_microbatches: microbatches in this iteration
        packed_length: seq_length * micro_batch_size -- the upper bound on document count
        fetch_batch: callable(data_iterator) -> batch dict, already on device. Must route
            through get_batch_on_this_tp_rank so the host-side cu_seqlens gets stashed.
        reads_data: whether this rank pulls from the dataloader
        derive: whether this chunk derives and broadcasts cu_seqlens. Only the chunk whose
            first stage IS the model-parallel source rank can (see share_cu_seqlens_from);
            the rest still fetch their batches but take cu_seqlens from that chunk.

    MUST be called by every rank of the model-parallel group, the same number of times per
    iteration, and BEFORE the pipeline schedule runs. `derive` must agree across the group
    for a given chunk, or the collectives will not match up.

    ============ WHY BEFORE THE SCHEDULE, AND NOT INSIDE forward_step ============
    Broadcasting cu_seqlens from inside get_batch DEADLOCKS as soon as the group spans
    pipeline stages. Measured, not theorised (job 1494386, hung at the first training
    step, cancelled):

        rank 0 (first stage) enters forward_step -> get_batch -> broadcast, and blocks
               waiting for every rank in the group;
        rank N (last stage) is in recv_forward, blocked on rank N-1 <- ... <- rank 0;
        rank 0 cannot produce those activations because it is stuck in the broadcast.

    A circular wait between the collective and the p2p activation chain. It is NOT about
    microbatch ordering -- the interleaved schedule's tables (get_schedule_table,
    schedules.py:1045) are rank-independent, so the k-th forward is the same
    (chunk, microbatch) on every rank and the CONTENT would have been correct. A blocking
    collective simply cannot sit inside a pipeline whose stages are waiting on each
    other's activations. Hoisting it out here, where no p2p is in flight, is what makes
    one broadcast for all of TP and PP possible at all.
    =============================================================================
    """
    key = _key(vp_stage)
    device = torch.cuda.current_device()
    group = parallel_state.get_model_parallel_group()
    src_rank = parallel_state.get_model_parallel_src_rank()
    is_source = torch.distributed.get_rank() == src_rank

    # [n_entries, max_seqlen, cu_seqlens...] per microbatch. A pack of `packed_length`
    # tokens holds at most that many documents, hence at most packed_length + 1 entries.
    width = packed_length + 3
    buffer = torch.zeros(num_microbatches, width, dtype=torch.int32, device=device)

    batches = []
    if reads_data:
        # Consume the whole iteration up front; get_batch then serves from this stash
        # instead of the iterator, so nothing is consumed twice.
        for index in range(num_microbatches):
            batch = fetch_batch(data_iterator)
            batches.append(batch)
            # fetch_batch went through get_batch_on_this_tp_rank, which derived cu_seqlens
            # on the HOST and put it in the batch. Popping it here rather than recomputing
            # from batch['tokens'] keeps the derivation single-sourced, and avoids the
            # per-microbatch device readback the GPU version needed for max_seqlen.
            # Present on TP rank 0 only; is_source implies TP rank 0.
            # Popped even when not is_source, so the entry never travels on to the model.
            header = pop_cu_seqlens(batch)
            if is_source:
                assert header is not None, (
                    "packed-doc-attention prefetch: the model-parallel source rank read a "
                    f"batch but it carried no {CU_SEQLENS_KEY!r} entry."
                )
                cu_seqlens, max_seqlen = header
                n_entries = int(cu_seqlens.shape[0])
                buffer[index, 0] = n_entries
                buffer[index, 1] = max_seqlen
                buffer[index, 2 : 2 + n_entries] = torch.from_numpy(cu_seqlens).to(device)

    _BATCH_STASH[key] = batches
    if not derive:
        # Batches still had to be fetched above (the last-stage chunk needs its labels), but
        # cu_seqlens comes from the deriving chunk -- caller wires that with
        # share_cu_seqlens_from. Returning before the collective is what keeps the group in
        # step: every rank skips it for this chunk, not just the ones that read.
        return

    # The one collective. No p2p is in flight here, so it cannot deadlock.
    if group is not None and group.size() > 1:
        torch.distributed.broadcast(buffer, src_rank, group=group)

    # ONE host sync for the whole iteration -- the per-microbatch derivation this
    # replaced needed one each: TE wants max_seqlen as a python int and the cu_seqlens
    # slice length is only known after the broadcast.
    header = buffer[:, :2].tolist()
    stash = [
        (buffer[index, 2 : 2 + int(n_entries)], int(max_seqlen))
        for index, (n_entries, max_seqlen) in enumerate(header)
    ]
    # An empty slice here means the source never wrote -- the VPP>1 bug this `derive` gate
    # exists to prevent. Catch it at the source instead of 60 layers later inside a TE
    # kernel assert, where the message says nothing about why.
    assert all(cu.numel() > 0 for cu, _ in stash), (
        f"packed-doc-attention: chunk {vp_stage} broadcast an EMPTY cu_seqlens. The "
        "model-parallel source rank did not read data for this chunk, so it never filled the "
        "buffer. Only the chunk whose first stage is the source may derive; the others must "
        "use share_cu_seqlens_from()."
    )
    _CU_SEQLENS_STASH[key] = stash


def pop(vp_stage, reads_data):
    """Return (batch_or_None, cu_seqlens, max_seqlen) for this chunk's next microbatch."""
    key = _key(vp_stage)
    assert key in _CU_SEQLENS_STASH, (
        f"packed-doc-attention: chunk {vp_stage} was not prefetched this iteration. "
        "prefetch_iteration() must run before the pipeline schedule."
    )
    cu_seqlens, max_seqlen = _CU_SEQLENS_STASH[key].pop(0)
    batch = _BATCH_STASH[key].pop(0) if reads_data else None
    return batch, cu_seqlens, max_seqlen
