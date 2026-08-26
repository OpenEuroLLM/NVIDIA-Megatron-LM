# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""OELLM PATCH: per-iteration cu_seqlens prefetch for --packed-doc-attention-scatter.

Every pipeline stage needs cu_seqlens for its own transformer layers, but only some stages
read the dataloader. Two ways to close that gap, with OPPOSITE scaling -- neither dominates,
and picking by scale rather than by taste is the whole point:

  local     every stage derives cu_seqlens from its own read, once PER MODEL CHUNK. Cost is
            the ADDRESS SPACE those builds map: chunks x (size of the dataset index cache).
            Independent of the microbatch count. Dies at PP=4/VPP=4 with a 704 GB cache --
            ~412 GB mapped against ~472 GB of CPU RAM (job 1497412, and it HANGS rather than
            exiting, because the other ranks wait on the dead one).

  scatter   the endpoint stages derive cu_seqlens for the WHOLE iteration up front and
            broadcast it in ONE collective. Cost is the prefetch, which cannot hide behind
            compute, so it scales with the MICROBATCH COUNT and not with chunks:
            +14.7-34.1% at M=1024 (DP=1, jobs 1497045/46), no measurable cost at M=16
            (DP=128, job 1498007). The only mode that runs the production layout.

Rule of thumb: large DP -> small M -> scatter. Small DP -> large M -> local, if it fits.
Beware of carrying a small-scale measurement across: the two costs move in opposite
directions with scale, so "local wins" at 2 nodes says nothing about 512.

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
# THE PART THAT IS NOT OBVIOUS is the tensor-parallel group. Only TP rank 0 reads the
# dataloader; the others have no host-side tokens, and on a MIDDLE pipeline stage they get no
# token broadcast at all (utils.get_batch_on_this_tp_rank only broadcasts on the first/last
# stage). Sharing the header over the NCCL TP group would put it back on the GPU and
# reintroduce exactly the readback we just removed. So the header travels over a GLOO TP
# group instead -- a few dozen bytes between ranks on one node, entirely host-side, no stream
# involvement. That group is only created when --packed-doc-attention is set
# (initialize_model_parallel(create_tensor_parallel_gloo_group=...)).


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


# Hand-off slot: written by utils.get_batch_on_this_tp_rank (TP rank 0 only, where the
# tokens still live on the host), read by share_cu_seqlens_over_tp and prefetch_iteration.
#
# A MODULE GLOBAL IS SAFE HERE, AND THE REASON IS NARROW -- read it before changing either
# side. It rests on three properties, ALL of which are load-bearing:
#
#   1. TAKE HAS POP SEMANTICS. take_cpu_cu_seqlens() clears the slot, so a value can be
#      consumed at most once and a second read gets None rather than a repeat.
#   2. EVERY TAKE IS IMMEDIATELY PRECEDED BY A STASH, in the same call chain, with no other
#      producer in between:
#        local    get_batch -> get_batch_on_this_tp_rank (stash) -> _shared_packed_seq_params
#                 -> share_cu_seqlens_over_tp (take)
#        scatter  prefetch_iteration -> fetch_batch -> get_batch_on_this_tp_rank (stash)
#                 -> take, once per microbatch
#      So a LEFTOVER from an unconsumed stash is always OVERWRITTEN before the next take.
#      training.dummy_train_step() is exactly that case -- it calls
#      get_batch_on_this_tp_rank in a loop and never consumes -- and it is harmless for
#      this reason, though it does pay the derivation for nothing.
#   3. A MISSING STASH FAILS LOUD, NOT SILENT. share_cu_seqlens_over_tp asserts when the
#      slot is empty on TP rank 0; ranks that never read (TP rank != 0) legitimately see
#      None and take the broadcast instead.
#
# What would BREAK it: separating the stash from the take -- prefetching a microbatch ahead,
# batching several reads before consuming, or calling get_batch from more than one thread.
# Any of those turns "overwritten before use" into "consumed from the wrong microbatch",
# which is silently wrong document boundaries with no error anywhere. If you need that,
# key the slot by microbatch instead of making it a single cell.
_CPU_HEADER = None


def stash_cpu_cu_seqlens(cu_seqlens, max_seqlen):
    """Record this microbatch's host-derived boundaries for the TP group to pick up.

    Overwriting an unconsumed value is deliberate, not sloppy -- see property 2 above.
    """
    global _CPU_HEADER
    _CPU_HEADER = (cu_seqlens, max_seqlen)


def take_cpu_cu_seqlens():
    """Pop what stash_cpu_cu_seqlens left, or None on a rank that did not read data.

    POP, not peek: clearing is what stops a value being reused for a later microbatch.
    """
    global _CPU_HEADER
    header, _CPU_HEADER = _CPU_HEADER, None
    return header


# Reused across microbatches. Safe ONLY because the gloo broadcast below is synchronous on
# the host: by the time this function returns, nothing else is reading it. It must never
# become the source of an async H2D -- see the lifetime note in share_cu_seqlens_over_tp.
_SHARED_BUFFER = None


def _shared_buffer(packed_length):
    """Fixed-size int32 host scratch, `packed_length + 2` wide, allocated once."""
    global _SHARED_BUFFER
    size = packed_length + 2
    if _SHARED_BUFFER is None or _SHARED_BUFFER.numel() < size:
        _SHARED_BUFFER = torch.empty(size, dtype=torch.int32)
    return _SHARED_BUFFER


def share_cu_seqlens_over_tp(packed_length):
    """Give every TP rank this microbatch's (device cu_seqlens, max_seqlen), sync-free.

    LOCAL MODE ONLY. Scatter broadcasts over the model-parallel group, which already spans
    TP, so it never comes through here -- which is why initialize.py only asks for the gloo
    tensor-parallel group when scatter is off.

    TP rank 0 supplies the host-derived values via stash_cpu_cu_seqlens; the rest receive
    them over the GLOO tensor-parallel group, so the header is host-side on arrival and no
    rank ever reads back from the device. The exact-size cu_seqlens is then uploaded with a
    single non-blocking copy out of pinned memory.

    Layout of the shared buffer: ``[n_entries, max_seqlen, cu_seqlens...]``. Fixed size,
    because a pack of `packed_length` tokens holds at most that many documents and therefore
    at most packed_length + 1 cu_seqlens entries -- so no size negotiation is needed and the
    broadcast is a single call.
    """
    shared = _shared_buffer(packed_length)
    header = take_cpu_cu_seqlens()

    if header is not None:
        cu_seqlens, max_seqlen = header
        n_entries = int(cu_seqlens.shape[0])
        shared[0] = n_entries
        shared[1] = max_seqlen
        shared[2 : 2 + n_entries] = torch.from_numpy(cu_seqlens)
    else:
        assert parallel_state.get_tensor_model_parallel_rank() != 0, (
            "TP rank 0 must derive cu_seqlens on the host before the TP share; nothing was "
            "stashed. Did get_batch_on_this_tp_rank run for this microbatch?"
        )

    gloo_group = parallel_state.get_tensor_model_parallel_group_gloo()
    # Without the gloo group, ranks other than TP rank 0 would silently fall through with
    # whatever the shared buffer held from the previous microbatch -- right shape, right
    # dtype, wrong document boundaries, no error. Fail loudly instead.
    assert gloo_group is not None or parallel_state.get_tensor_model_parallel_world_size() == 1, (
        "--packed-doc-attention needs the gloo tensor-parallel group to share cu_seqlens "
        "across a TP group of size "
        f"{parallel_state.get_tensor_model_parallel_world_size()}. It is created by "
        "initialize_model_parallel(create_tensor_parallel_gloo_group=True), which "
        "megatron/training/initialize.py requests whenever the flag is on."
    )
    if gloo_group is not None and torch.distributed.get_world_size(gloo_group) > 1:
        # Host-side collective, and SYNCHRONOUS: no CUDA stream is touched, so it cannot
        # drain the queue, and `shared` is free to reuse the moment this returns.
        torch.distributed.broadcast(
            shared[: packed_length + 2],
            parallel_state.get_tensor_model_parallel_src_rank(),
            group=gloo_group,
        )

    # .item() on a HOST tensor is a plain memory read -- no synchronisation. Reading these
    # off the device is exactly what used to cost a stream drain per microbatch.
    n_entries = int(shared[0].item())
    max_seqlen = int(shared[1].item())

    # ==== LIFETIME: BOTH OF THESE MUST BE FRESH, NOT SLICES OF A REUSED BUFFER ====
    # TE keeps cu_seqlens for the BACKWARD pass, which under 1F1B runs several microbatches
    # after the forward that produced it. Handing back a view into a recycled buffer would
    # let a later microbatch overwrite the boundaries an earlier one is still going to use
    # -- silently, since the shape never changes. So the device tensor is per-microbatch.
    #
    # The staging tensor is pinned and freshly requested for the same reason from the other
    # side: a non_blocking H2D reads it asynchronously, so it has to stay valid until the
    # copy retires. torch's caching host allocator tracks exactly that and will not hand the
    # block back until the recorded stream work completes, which is what makes a per-call
    # `pin_memory=True` request both cheap and correct. Pageable staging would be neither:
    # CUDA synchronises before a pageable H2D, reintroducing the stall.
    staging = torch.empty(n_entries, dtype=torch.int32, pin_memory=True)
    staging.copy_(shared[2 : 2 + n_entries])
    cu_seqlens_device = torch.empty(
        n_entries, dtype=torch.int32, device=torch.cuda.current_device()
    )
    cu_seqlens_device.copy_(staging, non_blocking=True)
    return cu_seqlens_device, max_seqlen


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
    global _CPU_HEADER
    _BATCH_STASH.clear()
    _CU_SEQLENS_STASH.clear()
    # Also drop any unconsumed hand-off. Harmless in the steady state -- a leftover is
    # always overwritten before the next take -- but leaving stale state across an
    # iteration boundary is the kind of thing that only bites once the call order changes.
    _CPU_HEADER = None


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

    Without this, scatter is BROKEN at VPP>1 and fails loudly at the first step:
    prefetch_iteration only fills the broadcast buffer inside `if reads_data:`, and
    reads_data is `is_first_or_last_pipeline_stage(vp_stage)` -- evaluated PER CHUNK. The
    model-parallel source rank hosts chunks 0..VPP-1 but is the first stage only for chunk 0,
    so for every other chunk it never wrote anything, torch.zeros was broadcast, n_entries
    came out 0, and the cu_seqlens slice was EMPTY:

        fused_rope.cu:477 Assertion failed: cu_seqlens != nullptr, required for THD format

    measured at 512 nodes, PP=4/VPP=4, job 1497767, 1865 ranks. It never showed earlier
    because every scatter test so far ran VPP=1, where the source IS always the first stage.

    Deriving once and sharing also drops the collective count from VPP per iteration to one,
    and costs no extra dataset reads -- which matters, because reading on more chunks is
    exactly what makes local mode run out of memory at this layout.
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
            # fetch_batch went through get_batch_on_this_tp_rank, which derives cu_seqlens
            # on the HOST and stashes it. Taking it here rather than recomputing from
            # batch['tokens'] keeps scatter on the same single derivation as local mode --
            # and avoids the per-microbatch device readback the GPU version needed for
            # max_seqlen. Populated on TP rank 0 only; is_source implies TP rank 0.
            header = take_cpu_cu_seqlens()
            if is_source:
                assert header is not None, (
                    "scatter prefetch: the model-parallel source rank read a batch but no "
                    "host-derived cu_seqlens was stashed."
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

    # A single host sync for the whole iteration, rather than one per microbatch as the
    # local mode needs: TE wants max_seqlen as a python int and the cu_seqlens slice
    # length is only known after the broadcast.
    header = buffer[:, :2].tolist()
    stash = [
        (buffer[index, 2 : 2 + int(n_entries)], int(max_seqlen))
        for index, (n_entries, max_seqlen) in enumerate(header)
    ]
    # An empty slice here means the source never wrote -- the VPP>1 bug this `derive` gate
    # exists to prevent. Catch it at the source instead of 60 layers later inside a TE
    # kernel assert, where the message says nothing about why.
    assert all(cu.numel() > 0 for cu, _ in stash), (
        f"packed-doc-attention scatter: chunk {vp_stage} broadcast an EMPTY cu_seqlens. The "
        "model-parallel source rank did not read data for this chunk, so it never filled the "
        "buffer. Only the chunk whose first stage is the source may derive; the others must "
        "use share_cu_seqlens_from()."
    )
    _CU_SEQLENS_STASH[key] = stash


def pop(vp_stage, reads_data):
    """Return (batch_or_None, cu_seqlens, max_seqlen) for this chunk's next microbatch."""
    key = _key(vp_stage)
    assert key in _CU_SEQLENS_STASH, (
        f"packed-doc-attention scatter: chunk {vp_stage} was not prefetched this iteration. "
        "prefetch_iteration() must run before the pipeline schedule."
    )
    cu_seqlens, max_seqlen = _CU_SEQLENS_STASH[key].pop(0)
    batch = _BATCH_STASH[key].pop(0) if reads_data else None
    return batch, cu_seqlens, max_seqlen
