# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Canon layers (Allen-Zhu, "Physics of Language Models: Part 4.1", 2025).

A Canon layer adds a short causal depthwise convolution to a hidden representation,
``h_t <- h_t + sum_j w_j * h_{t-j}`` (kernel ``canon_kernel_size``, weights initialized to
zero, so a model starts exactly as without it). ``canon_layers`` selects the positions,
as in the paper:

* ``A`` -- the input of the token-mixer sublayer (the residual stream before its norm),
* ``B`` -- the attention q/k/v projection output (linear mixers carry their own conv),
* ``C`` -- the input of the MLP sublayer,
* ``D`` -- the MLP fc1 output, fused with the SwiGLU activation (fused_canon.py).

A and C feed the sublayer only; the residual stream itself is unchanged. The work is
in the Triton kernels of megatron/core/fusions/fused_canon.py; this module handles
parallelism:

* sequence parallelism on the residual stream (A, C): each TP rank holds a sequence
  slice, so the ``K-1`` rows before it come from the previous rank (a halo exchange;
  B and D act after the column-parallel gather and see the whole sequence);
* context parallelism: as ShortConv, all-to-all to the whole sequence on a slice of
  the channels and back.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.core.fusions.fused_canon import canon_conv, canon_swiglu


class _PrevRankHalo(torch.autograd.Function):
    """The last ``k`` sequence rows of the previous rank of ``group`` (zeros on rank 0)."""

    @staticmethod
    def forward(ctx, x, k, group):
        ctx.group, ctx.k, ctx.shape = group, k, x.shape
        tail = x[-k:].contiguous()
        tails = [torch.empty_like(tail) for _ in range(group.size())]
        dist.all_gather(tails, tail, group=group)
        rank = group.rank()
        return tails[rank - 1] if rank > 0 else torch.zeros_like(tail)

    @staticmethod
    def backward(ctx, grad):
        grads = [torch.empty_like(grad) for _ in range(ctx.group.size())]
        dist.all_gather(grads, grad.contiguous(), group=ctx.group)
        rank, size = ctx.group.rank(), ctx.group.size()
        dx = grad.new_zeros(ctx.shape)
        if rank + 1 < size:  # my tail was the next rank's halo
            dx[-ctx.k :] = grads[rank + 1]
        return dx, None, None


class Canon(nn.Module):
    """One Canon layer over ``[S, B, channels]`` (see the module docstring).

    Args:
        config: TransformerConfig (``canon_kernel_size``, ``sequence_parallel``).
        channels: local channel count (already divided by TP where the input is).
        tp_group, cp_group: process groups.
        sequence_sharded: the input is sharded on the sequence over TP (A, C under SP).
        tp_sharded_channels: the channels are a TP slice (B, D): the weight is then a
            TP-sharded parameter; otherwise it is replicated (summed over TP under SP).
        swiglu: fuse with SwiGLU (D): ``[S, B, 2F]`` in, ``[S, B, F]`` out.
    """

    def __init__(
        self,
        config,
        channels: int,
        tp_group,
        cp_group,
        sequence_sharded: bool = False,
        tp_sharded_channels: bool = False,
        swiglu: bool = False,
    ):
        super().__init__()
        self.k = config.canon_kernel_size
        self.tp_group, self.cp_group = tp_group, cp_group
        self.sequence_sharded = sequence_sharded and tp_group is not None and tp_group.size() > 1
        self.swiglu = swiglu
        # SwiGLU (D): [2, F, K] -- the gate and up halves -- so the TP-sharded global
        # tensor is [gate | up] for every TP size (a flat [2F/tp, K] shard would
        # concatenate to a TP-dependent interleaving).
        shape = (2, channels // 2, self.k) if swiglu else (channels, self.k)
        self.weight = nn.Parameter(
            torch.zeros(shape, device=torch.cuda.current_device(), dtype=config.params_dtype)
        )
        if tp_sharded_channels:
            self.weight.tensor_model_parallel = True
            self.weight.partition_dim = 1 if swiglu else 0
            self.weight.partition_stride = 1
        else:
            # Acts per token: under SP each TP rank sees only its slice of the tokens.
            self.weight.sequence_parallel = config.sequence_parallel
        self.tp_sharded_channels = tp_sharded_channels

    def _flat_weight(self):
        return self.weight.reshape(-1, self.k)  # D: [gate | up] channels

    def _conv(self, x, weight, prefix):
        if self.swiglu:
            return canon_swiglu(x, weight, prefix)
        return canon_conv(x, weight, prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cp = self.cp_group.size() if self.cp_group is not None else 1
        if cp > 1:
            return self._forward_cp(x)
        prefix = None
        if self.sequence_sharded:
            prefix = _PrevRankHalo.apply(x, self.k - 1, self.tp_group)
        return self._conv(x, self._flat_weight(), prefix)

    def _forward_cp(self, x: torch.Tensor) -> torch.Tensor:
        from megatron.core.ssm.gated_delta_net.common import (
            get_parameter_local_cp,
            tensor_a2a_cp2hp,
            tensor_a2a_hp2cp,
        )
        from megatron.core.tensor_parallel.mappings import (
            gather_from_sequence_parallel_region,
            scatter_to_sequence_parallel_region,
        )

        if self.sequence_sharded:
            # The CP exchange wants this CP rank's whole chunk; every TP rank then
            # computes the same thing, so the gather's backward keeps the local slice
            # and the scatter's backward all-gathers.
            x = gather_from_sequence_parallel_region(
                x, tensor_parallel_output_grad=False, group=self.tp_group
            )
        if self.swiglu:
            # The fused kernel splits channels into [gate | up] halves; on a CP channel
            # slice they would interleave, so convolve per half and apply SwiGLU after.
            a, g = x.chunk(2, dim=-1)
            wa, wg = self.weight[0], self.weight[1]
            a = self._cp_conv(a, wa, tensor_a2a_cp2hp, tensor_a2a_hp2cp, get_parameter_local_cp)
            g = self._cp_conv(g, wg, tensor_a2a_cp2hp, tensor_a2a_hp2cp, get_parameter_local_cp)
            out = torch.nn.functional.silu(a.float()) * g.float()
            out = out.to(x.dtype)
        else:
            out = self._cp_conv(
                x, self.weight, tensor_a2a_cp2hp, tensor_a2a_hp2cp, get_parameter_local_cp
            )
        if self.sequence_sharded:
            out = scatter_to_sequence_parallel_region(out, group=self.tp_group)
        return out

    def _cp_conv(self, x, weight, cp2hp, hp2cp, local_cp):
        x = cp2hp(x, seq_dim=0, head_dim=-1, cp_group=self.cp_group)  # [s, b, C/cp]
        w = local_cp(weight, dim=0, cp_group=self.cp_group)
        y = canon_conv(x, w, None)
        return hp2cp(y, seq_dim=0, head_dim=-1, cp_group=self.cp_group)
