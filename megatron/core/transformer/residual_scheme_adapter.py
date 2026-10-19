# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""The depth-wise residual schemes for layers that are not a TransformerLayer.

:class:`~megatron.core.transformer.transformer_layer.TransformerLayer` routes each of
its sublayers through mHC, Block AttnRes or the nGPT LERP residual itself. A
HybridStack also holds layers that do their own ``h + f(norm(h))`` --
:class:`~megatron.core.ssm.mamba_layer.MambaLayer` (`M`) and
:class:`~megatron.core.ssm.engram.EngramLayer` (`N`). Such a layer owns one
:class:`ResidualSchemeAdapter`, which turns that into the same read / write the
TransformerLayer sublayers do::

    x, ctx = adapter.read(hidden_states, depth_state)   # the sublayer's input
    out = f(x)                                           # the layer, without residual
    hidden_states = adapter.write(out, ctx, depth_state, layer_number)

With no scheme active the adapter is not built and the layer keeps its own residual.
"""

from typing import Optional

import torch
from torch import Tensor, nn


def residual_scheme_active(config) -> bool:
    """True when mHC, Block AttnRes or the nGPT LERP residual replaces ``h + f``."""
    return bool(
        getattr(config, "mhc", False)
        or getattr(config, "attn_res", False)
        or getattr(config, "depth_memory", False)
        or getattr(config, "ngpt", False)
        or getattr(config, "post_sublayer_norm", False)
    )


class ResidualSchemeAdapter(nn.Module):
    """mHC / Block AttnRes / nGPT residual around one single-sublayer layer.

    Holds the same per-sublayer modules a TransformerLayer builds for its
    attention slot (``mhc_mixer``, ``attnres``, ``ngpt_residual``), so a layer of
    either kind costs and behaves the same in a stack.
    """

    def __init__(self, config, layer_number: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        self.mhc_mixer = None
        self.attnres = None
        self.ldm = None
        self.ngpt_residual = None
        self.post_norm = None
        if getattr(config, "post_sublayer_norm", False):
            from megatron.core.transformer.post_norm import build_post_norm

            self.post_norm = build_post_norm(config)
        if getattr(config, "ngpt", False):
            from megatron.core.transformer.ngpt import NGPTResidual

            self.ngpt_residual = NGPTResidual(
                hidden_size=config.hidden_size,
                alpha_init_value=config.ngpt_alpha_init_value,
                base_scale=config.ngpt_base_scale,
                alpha_activation=config.ngpt_alpha_activation,
                lerp_normalization=config.ngpt_lerp_normalization_attn
                or config.ngpt_lerp_normalization,
                post_layer_norm=config.ngpt_post_layer_norm_attn,
                eps=config.layernorm_epsilon,
            )
        if getattr(config, "mhc", False):
            from megatron.core.transformer.routed_residual import build_stream_mixer

            self.mhc_mixer = build_stream_mixer(config)
        elif getattr(config, "attn_res", False):
            from megatron.core.transformer.attn_res import BlockAttnRes

            self.attnres = BlockAttnRes(
                hidden_size=config.hidden_size,
                eps=config.layernorm_epsilon,
                query_init_std=config.attn_res_query_init_std,
            )
        elif getattr(config, "depth_memory", False):
            from megatron.core.transformer.depth_memory import DepthMemorySite

            self.ldm = DepthMemorySite(config)
        from megatron.core.transformer import arch_metrics

        for module in (self.mhc_mixer, self.attnres, self.ngpt_residual):
            arch_metrics.tag(module, "mixer", layer_number, config)
        if self.ldm is not None:
            arch_metrics.tag(self.ldm.read, "mixer", layer_number, config)
            arch_metrics.tag(self.ldm.write, "mixer", layer_number, config)
        # Per-token modules: under sequence parallelism each TP rank sees only its own
        # tokens, so their gradients are summed over TP like a norm's.
        for param in self.parameters():
            param.sequence_parallel = config.sequence_parallel

    def _lerp(self, use_lerp: bool):
        if not use_lerp:
            return None
        if self.ngpt_residual is None:
            raise ValueError("mhc_lerp / attn_res_lerp require ngpt=True.")
        return self.ngpt_residual

    def read(self, hidden_states: Tensor, depth_state=None) -> tuple[Tensor, tuple]:
        """The sublayer's ``[s, b, h]`` input, plus what :meth:`write` needs."""
        if self.mhc_mixer is not None:
            streams = hidden_states  # [s, b, n, h]
            h_pre, h_post, h_res = self.mhc_mixer(streams)
            from megatron.core.transformer import arch_metrics

            if arch_metrics.active():
                self.mhc_mixer.record_metrics(streams, (h_pre, h_post, h_res))
            return self.mhc_mixer.read(streams, h_pre), (streams, h_post, h_res)
        if self.attnres is not None:
            if depth_state is None:
                raise RuntimeError("Block AttnRes layer called without the stack's depth state.")
            return self.attnres(depth_state.sources()), ()
        if self.ldm is not None:
            if depth_state is None:
                raise RuntimeError("depth_memory layer called without the stack's depth state.")
            return self.ldm.read_input(depth_state), ()
        return hidden_states, (hidden_states,)

    def write(
        self, out: Tensor, ctx: tuple, depth_state=None, layer_number: Optional[int] = None
    ) -> Tensor:
        """Fold the sublayer output ``out`` back into the residual state."""
        if self.post_norm is not None:
            out = self.post_norm(out)
        if self.mhc_mixer is not None:
            streams, h_post, h_res = ctx
            return self.mhc_mixer.write(
                streams, out, h_post, h_res, lerp=self._lerp(self.config.mhc_lerp)
            )
        if self.ldm is not None:
            return self.ldm.write_output(depth_state, out)
        if self.attnres is not None:
            from megatron.core.transformer.attn_res import accumulate_block

            accumulate_block(depth_state, out, lerp=self._lerp(self.config.attn_res_lerp))
            if layer_number is not None and depth_state.should_seal(layer_number):
                depth_state.seal_block()
            return depth_state.partial if depth_state.partial is not None else out
        (residual,) = ctx
        if self.ngpt_residual is None:  # post_sublayer_norm alone: the plain residual
            return residual + out
        return self.ngpt_residual(out, residual)
