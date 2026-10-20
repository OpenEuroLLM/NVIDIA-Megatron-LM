# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""mHC per transformer layer (``mhc_per_layer``): one stream read and write per block of HybridStack
layers instead of per layer.

A HybridStack layer is one sublayer (``L`` CKDA, ``*`` attention, ``N`` Engram, ``E`` MoE, ...),
so per-sublayer mHC reads and writes the n-stream state around every one of them. With
``mhc_per_layer`` the layers up to and including each MLP / MoE layer form one block -- a
transformer layer, e.g. ``L E`` or ``* E`` -- and the block is the mHC layer:

.. code-block:: text

    x = H_pre S                       (first layer of the block; its mixer is the block's)
    h = x;  h = h + F_1(h);  ...      (the block's layers, plain residual on [s, b, h])
    S' = H_res S + H_post^T (h - x)   (last layer)

That halves the mHC applications of a mixer + MoE pattern (the bandwidth-bound part of an mHC-4
step). The in-block state (the streams, ``x``, the read's coefficients) goes from the first to the
last layer through a :class:`BlockState` within one forward -- and within one recompute replay:
recompute chunks are snapped to block ends (``megatron/core/recompute.py``) and a pipeline stage
(or virtual chunk) must end with a block, so what crosses chunks and stages is always the
n-stream state as before.
"""

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols


class BlockState:
    """Shared by the layers of one block: the block's mixer and, during a forward, the streams,
    the read ``x`` and the read's pending coefficients."""

    __slots__ = ("mixer", "streams", "x", "pending")

    def __init__(self, mixer):
        self.mixer = mixer
        self.streams = self.x = self.pending = None


def _mixers(layer):
    return [
        getattr(layer, name)
        for name in ("mhc_attn_mixer", "mhc_mlp_mixer", "mhc_mixer")
        if getattr(layer, name, None) is not None
    ]


def _drop_mixers(layer):
    for name in ("mhc_attn_mixer", "mhc_mlp_mixer", "mhc_mixer"):
        if getattr(layer, name, None) is not None:
            setattr(layer, name, None)


def assign_blocks(stack):
    """Group a HybridStack's layers into blocks ending at each MLP / MoE layer and give every layer
    of a block of two or more its role; a one-layer block stays plain per-sublayer mHC."""
    from megatron.core.transformer.transformer_layer import TransformerLayer

    types = stack.layer_type_list
    if types and types[-1] not in (Symbols.MLP, Symbols.MOE):
        raise ValueError(
            "mhc_per_layer: every pipeline stage / virtual chunk must end with an MLP or MoE layer "
            f"(a whole block), got the layer pattern {''.join(types)}"
        )
    blocks, cur = [], []
    for i, t in enumerate(types):
        cur.append(i)
        if t in (Symbols.MLP, Symbols.MOE):
            blocks.append(cur)
            cur = []
    for block in blocks:
        if len(block) == 1:
            continue
        layers = [stack.layers[i] for i in block]
        for pos, layer in enumerate(layers):
            if (pos == 0 or pos == len(layers) - 1) and not isinstance(layer, TransformerLayer):
                raise NotImplementedError(
                    f"mhc_per_layer: a block must start and end with a TransformerLayer, got "
                    f"{type(layer).__name__} in {''.join(types[i] for i in block)}"
                )
        first, last = layers[0], layers[-1]
        mixer = _mixers(first)[0]
        _drop_mixers(first)
        first.mhc_block_mixer = mixer  # registered on the first layer: its parameters live there
        state = BlockState(mixer)
        for layer in layers:
            if layer is not first:
                _drop_mixers(layer)
            layer._mhc_block = state
            layer._mhc_block_first = layer is first
            layer._mhc_block_last = layer is last


def block_forward(layer, inner, args, kwargs):
    """A TransformerLayer's forward inside an mHC block (see module doc)."""
    state = layer._mhc_block
    if "hidden_states" in kwargs:
        hidden_states = kwargs["hidden_states"]
    else:
        hidden_states, args = args[0], args[1:]
    if layer._mhc_block_first:
        x, pending = layer._mhc_read(state.mixer, hidden_states)
        state.streams, state.x, state.pending = hidden_states, x, pending
        hidden_states = x
    kwargs["hidden_states"] = hidden_states
    out, context = inner(*args, **kwargs)
    if layer._mhc_block_last:
        streams, x, pending = state.streams, state.x, state.pending
        state.streams = state.x = state.pending = None
        update = (out.float() - x.float()).to(out.dtype)  # the block's F(x); fp32 difference
        h_post, h_res, *link = pending
        out = state.mixer.write(streams, update, h_post, h_res, link=link[0] if link else None)
    return out, context


def is_block_end(layer):
    """False only for a layer inside an mHC block before its last layer (recompute snapping)."""
    return getattr(layer, "_mhc_block", None) is None or layer._mhc_block_last
