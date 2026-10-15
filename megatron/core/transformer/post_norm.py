# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Post-sublayer ("sandwich" / peri-LN) normalization.

With ``post_sublayer_norm`` every sublayer output is normalized, with a learnable
gain, before it enters the residual: ``h <- h + Norm_out(f(Norm_in(h)))`` (Gemma 2/3,
OLMo 2, Peri-LN). Unlike nGPT the residual stream itself stays unnormalized and the
update stays additive; with mHC / routed streams / Block AttnRes / nGPT the
normalized output is what they mix in.
"""

from typing import Optional, Tuple

import torch


def build_post_norm(config) -> torch.nn.Module:
    """A norm of ``config.normalization`` over ``hidden_size``, TE when available
    (sequence parallel aware), gain initialized to one."""
    try:
        from megatron.core.extensions.transformer_engine import HAVE_TE, TENorm
    except ImportError:  # pragma: no cover
        HAVE_TE = False
    if HAVE_TE:
        norm = TENorm(config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon)
    else:
        from megatron.core.transformer.torch_norm import WrappedTorchNorm

        norm = WrappedTorchNorm(
            config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
        )
    # Acts per token: under SP each TP rank sees its own tokens.
    for p in norm.parameters():
        p.sequence_parallel = config.sequence_parallel
    return norm


def apply_post_norm(
    norm: Optional[torch.nn.Module], output_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]]
):
    """``(out, bias)`` -> ``(norm(out + bias), None)``; unchanged without a norm."""
    if norm is None:
        return output_with_bias
    out, bias = output_with_bias
    if bias is not None:
        out = out + bias
    return norm(out), None
