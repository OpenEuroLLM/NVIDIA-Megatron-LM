# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""--megatron-fsdp-grad-comm-dtype-experts-only: the comm dtype per parameter group, and the CLI plumbing."""

import argparse
import types

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    ParamAndGradBuffer,
)


def _dtype_for(policy, is_expert):
    buffer = types.SimpleNamespace(mp_policy=policy)
    group = types.SimpleNamespace(is_expert_param=is_expert)
    return ParamAndGradBuffer.grad_comm_dtype_for(buffer, group)


def test_comm_dtype_applies_to_every_group_by_default():
    policy = MixedPrecisionPolicy(main_grads_dtype=torch.float32, grad_comm_dtype=torch.bfloat16)
    assert _dtype_for(policy, is_expert=True) == torch.bfloat16
    assert _dtype_for(policy, is_expert=False) == torch.bfloat16


def test_experts_only_keeps_dense_groups_in_the_main_grad_dtype():
    policy = MixedPrecisionPolicy(
        main_grads_dtype=torch.float32,
        grad_comm_dtype=torch.bfloat16,
        grad_comm_dtype_experts_only=True,
    )
    assert _dtype_for(policy, is_expert=True) == torch.bfloat16
    assert _dtype_for(policy, is_expert=False) is None


def test_experts_only_without_a_comm_dtype_is_a_no_op():
    policy = MixedPrecisionPolicy(main_grads_dtype=torch.float32, grad_comm_dtype_experts_only=True)
    assert _dtype_for(policy, is_expert=True) is None
    assert _dtype_for(policy, is_expert=False) is None


def test_cli_flag_reaches_the_ddp_config():
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.training.arguments import _add_experimental_args

    parser = argparse.ArgumentParser()
    _add_experimental_args(parser)
    args = parser.parse_args(
        ["--megatron-fsdp-grad-comm-dtype", "bf16", "--megatron-fsdp-grad-comm-dtype-experts-only"]
    )
    assert args.megatron_fsdp_grad_comm_dtype_experts_only is True
    assert parser.parse_args([]).megatron_fsdp_grad_comm_dtype_experts_only is False
    cfg = DistributedDataParallelConfig(megatron_fsdp_grad_comm_dtype_experts_only=True)
    assert cfg.megatron_fsdp_grad_comm_dtype_experts_only is True
