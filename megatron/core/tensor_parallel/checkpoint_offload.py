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
MIN_BYTES = int(os.environ.get("RECOMPUTE_OFFLOAD_MIN_MB", "16")) << 20

_stream = None
_last = None  # the running sequence's newest slot (its input still on the GPU)
_armed = False


def _side_stream():
    global _stream
    if _stream is None:
        _stream = torch.cuda.Stream()
    return _stream


def new_sequence():
    """A new sequence of chunks starts (checkpointed_forward): its first chunk has no previous."""
    global _last
    _last = None


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
        self.requires_grad = tensor.requires_grad

    def offload(self):
        t, side = self.gpu, _side_stream()
        side.wait_stream(torch.cuda.current_stream())
        self.host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        with torch.cuda.stream(side):
            self.host.copy_(t, non_blocking=True)
        t.record_stream(side)  # the allocator keeps the block until the copy is done
        self.gpu = None

    def fetch(self):
        """Start copying the input back (side stream); no-op if it is on the GPU."""
        if self.gpu is not None or self.host is None:
            return
        side = _side_stream()
        g = torch.empty(self.host.shape, dtype=self.host.dtype, device="cuda")
        side.wait_stream(torch.cuda.current_stream())  # g's block is free for the side stream
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
