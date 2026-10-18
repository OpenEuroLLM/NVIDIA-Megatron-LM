# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Transformer Engine linears for the projections of an fla layer (``linear_te_projections``).

fla layers own plain ``nn.Linear`` projections. Under Megatron that costs twice: their weight
gradients are accumulated into the fp32 ``main_grad`` by a separate autograd ``AccumulateGrad``
add per micro-batch (Megatron's TE linears fuse that accumulation into the wgrad GEMM:
``gradient_accumulation_fusion``), and they never run in fp8 (only TE modules do). Measured on
the one-node A3B slice (docs/hybrid_moe_3b_campaign.md §11.2): the AccumulateGrad adds were ~5 %
of the bf16 step, and 40 of the 48 mixer layers of a 5:1 CKDA hybrid stayed bf16 under fp8.

``swap_linears_to_te`` replaces every ``nn.Linear`` below a module (nested ones included, e.g.
the low-rank gate ``nn.Sequential``) by a TE linear with the same in / out features, bias and
weights, registered under the same attribute name -- so parameter names, shapes and the init
are unchanged and checkpoints keep loading. The replacement returns a plain tensor, as the
fla layer's call sites expect.
"""

import torch
from torch import nn

from megatron.core.extensions.transformer_engine import TELinear


class TELinearTensorOut(TELinear):
    """Megatron's ``TELinear`` (duplicated: no tensor parallelism) returning the output tensor
    alone, as ``nn.Linear`` does."""

    def forward(self, x):  # pylint: disable=arguments-differ
        out, bias = super().forward(x)
        return out if bias is None else out + bias


def _copy_param_attrs(src: nn.Parameter, dst: nn.Parameter) -> None:
    """Carry over the attributes Megatron / fla tag parameters with (e.g. weight-decay or
    optimizer routing flags), without overwriting TE's own."""
    for key, value in vars(src).items():
        if not key.startswith("_") and not hasattr(dst, key):
            setattr(dst, key, value)


def swap_linears_to_te(module: nn.Module, config) -> int:
    """Replace every ``nn.Linear`` below ``module`` by a ``TELinearTensorOut`` with the same
    weights. Returns the number of linears swapped."""
    swapped = 0
    for name, child in list(module.named_children()):
        if type(child) is nn.Linear:  # pylint: disable=unidiomatic-typecheck
            has_bias = child.bias is not None
            te = TELinearTensorOut(
                child.in_features,
                child.out_features,
                parallel_mode="duplicated",
                config=config,
                init_method=lambda w: None,
                bias=has_bias,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                is_expert=False,
            )
            te = te.to(device=child.weight.device)
            with torch.no_grad():
                te.weight.copy_(child.weight.to(te.weight.dtype))
                if has_bias:
                    te.bias.copy_(child.bias.to(te.bias.dtype))
            _copy_param_attrs(child.weight, te.weight)
            if has_bias:
                _copy_param_attrs(child.bias, te.bias)
            setattr(module, name, te)
            swapped += 1
        else:
            swapped += swap_linears_to_te(child, config)
    return swapped
