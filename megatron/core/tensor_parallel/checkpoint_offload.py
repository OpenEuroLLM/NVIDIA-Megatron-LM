# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Activation-checkpoint inputs in pinned host memory (RECOMPUTE_OFFLOAD=1).

Under full recompute every recompute chunk keeps only its input -- but with mHC that input is the
n-stream state, n times the hidden state: on the first pipeline stage of the mHC-4 hybrid (8k
tokens, PP4 x VPP2) 67 of them are live at the peak, 8.4 of the 11.9 GiB of activations
(MEM_TRACE_DIR, mem_peak.py). They are not needed until the chunk's backward, which under a
pipelined schedule runs long after its forward, so they can wait on the host:

* forward: when chunk k+1 of a sequence (one model chunk's forward over one micro-batch) is
  checkpointed, chunk k's input -- no longer read by the GPU -- is copied to pinned host memory on
  a side stream and its GPU block released. The last chunk of a sequence keeps its input on the
  GPU: its backward comes first, with nothing to hide the copy back under.
* backward: chunk k's backward takes its input back (waiting for the copy only if it is not
  there yet) and starts copying chunk k-1's input back, which then runs under chunk k's
  recompute + backward.

Only non-leaf inputs are offloaded (the outputs of the previous chunk, the embedding): a leaf
input -- a pipeline receive buffer -- is also held by the schedule, so the copy would free
nothing. Stays inside one pipeline stage; the tensors the schedule sends are untouched.
"""

import os

import torch

ENABLED = os.environ.get("RECOMPUTE_OFFLOAD", "0") == "1"
# Also the last chunk of every sequence (once the next sequence starts): its backward comes first,
# so it is fetched back ahead -- the model chunk's backward runs its micro-batches in forward
# order, and the start of one sequence's backward fetches the next one's (RECOMPUTE_OFFLOAD_LAST=1;
# for wide checkpoints, e.g. the LDM state: 16 x 0.7 GB on the first stage at 16k).
# "1": every stage; a comma list of pipeline ranks ("0,1"): only those (the first stages hold the
# most micro-batches in flight; later ones can keep their last chunks on the GPU)
_OFFLOAD_LAST_SPEC = os.environ.get("RECOMPUTE_OFFLOAD_LAST", "0")
OFFLOAD_LAST = _OFFLOAD_LAST_SPEC not in ("", "0")


def _offload_last_here() -> bool:
    if not OFFLOAD_LAST:
        return False
    if _OFFLOAD_LAST_SPEC == "1":
        return True
    from megatron.core import parallel_state

    return parallel_state.get_pipeline_model_parallel_rank() in {
        int(r) for r in _OFFLOAD_LAST_SPEC.split(",")
    }
MIN_BYTES = int(os.environ.get("RECOMPUTE_OFFLOAD_MIN_MB", "16")) << 20

# One stream per direction: PCIe is full duplex, and on one stream the backward's fetches queue
# behind the forward's offloads (mHC-4 at 16k, PP4 x VPP2: ~0.3 TB each way per step, the offloads
# finishing a median 130-180 ms after their Python free, 4 GB of pending blocks at the peak)
_streams = {}
_last = None  # the running sequence's newest slot (its input still on the GPU)
_armed = False


def _side_stream(direction):
    if direction not in _streams:
        _streams[direction] = torch.cuda.Stream()
    return _streams[direction]


_key = None
_queues = {}  # sequence key (the model chunk) -> its offloaded last slots, in forward order


def new_sequence(key=None):
    """A new sequence of chunks starts (checkpointed_forward of model chunk ``key``): its first
    chunk has no previous. With OFFLOAD_LAST the previous sequence's last chunk goes to the host."""
    global _last, _key
    if OFFLOAD_LAST and _last is not None and _last.gpu is not None and _offload_last_here():
        _last.offload()
        _last.last_key = _key
        _queues.setdefault(_key, []).append(_last)
    _last, _key = None, key


def last_fetched(slot):
    """The backward took ``slot``: if it was a sequence's offloaded last chunk, start copying back
    the next one of its model chunk (its backward is next in line)."""
    key = getattr(slot, "last_key", False)
    if key is False:
        return
    queue = _queues.get(key, [])
    if slot in queue:
        queue.remove(slot)
    if queue:
        queue[0].fetch()


def arm():
    """The next CheckpointFunction is a recompute chunk of the running sequence (not a checkpoint
    nested inside a layer)."""
    global _armed
    _armed = ENABLED


def take_armed():
    global _armed
    armed, _armed = _armed, False
    return armed


class Slot:
    """One checkpoint input: on the GPU, on its way to / on the host, or on its way back."""

    def __init__(self, tensor, prev):
        self.gpu, self.host, self.event, self.prev = tensor, None, None, prev
        self.offloaded = None  # the D2H copy's event: the H2D copy back reads the host buffer
        self.requires_grad = tensor.requires_grad

    def offload(self):
        t, side = self.gpu, _side_stream("d2h")
        side.wait_stream(torch.cuda.current_stream())
        self.host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        with torch.cuda.stream(side):
            self.host.copy_(t, non_blocking=True)
            self.offloaded = side.record_event()
        t.record_stream(side)  # the allocator keeps the block until the copy is done
        self.gpu = None

    def fetch(self):
        """Start copying the input back (side stream); no-op if it is on the GPU."""
        if self.gpu is not None or self.host is None:
            return
        side = _side_stream("h2d")
        g = torch.empty(self.host.shape, dtype=self.host.dtype, device="cuda")
        side.wait_stream(torch.cuda.current_stream())  # g's block is free for the side stream
        side.wait_event(self.offloaded)
        with torch.cuda.stream(side):
            g.copy_(self.host, non_blocking=True)
            self.event = side.record_event()
        self.gpu, self.host = g, None

    def get(self):
        """The input, on the GPU, as the checkpoint saved it (a fetched copy is a new leaf: without
        requires_grad the recompute would cut the gradient to the earlier chunks)."""
        self.fetch()
        if self.event is not None:
            torch.cuda.current_stream().wait_event(self.event)
        g, self.gpu, self.event = self.gpu, None, None
        if g.is_leaf and g.requires_grad != self.requires_grad:
            g.requires_grad_(self.requires_grad)
        return g


def register(tensor):
    """A slot for this recompute chunk's input, or None (not offloadable). Offloads the previous
    chunk's input of the sequence, which the GPU no longer reads."""
    global _last
    if (
        not isinstance(tensor, torch.Tensor)
        or not tensor.is_cuda
        or tensor.grad_fn is None
        or tensor.numel() * tensor.element_size() < MIN_BYTES
    ):
        _last = None
        return None
    prev = _last
    if prev is not None and prev.gpu is not None:
        prev.offload()
    _last = Slot(tensor, prev)
    return _last
