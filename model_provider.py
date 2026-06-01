# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Common functions used in train_*.py and pretrain_*.py scripts."""

from typing import Callable, Optional, Union

import torch

from megatron.core.models.gpt import GPTModel
from megatron.core.models.mamba import MambaModel
from megatron.training import get_args, print_rank_0

try:
    from megatron.post_training.model_builder import modelopt_gpt_mamba_builder
    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

import megatron.legacy.model  # isort: skip

# NOTE: Loading `megatron.legacy.model` earlier fails due to circular import


def model_provider(
    model_builder: Callable, pre_process=True, post_process=True, vp_stage: Optional[int] = None, config=None, pg_collection=None,
) -> Union[GPTModel, megatron.legacy.model.GPTModel, MambaModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        model_builder: A callable that builds the actual model, its signature is the same as model_provider's with an exception of the first argument which is a builder itself. In addition might take a config passed from outside to skip its own config loading. See gpt_builder or mamba_builder for an example, see _gpt_model_builder in train_rl.py to see how to augment a default gpt builder and pass the config from outside
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to compute output logits/loss. Defaults to True.

    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel, MambaModel]: The returned model
    """
    args = get_args()

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')

            filename = f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}"
            torch.cuda.memory._dump_snapshot(filename)

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):
        # [ModelOpt]: Use custom builder + spec when modelopt is enabled
        model_builder = modelopt_gpt_mamba_builder

    model = model_builder(args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)

    router_lr_spec = getattr(args, 'router_lr_spec', None)
    if router_lr_spec is not None:
        from megatron.core.optimizer import _parse_router_lr_spec
        groups = _parse_router_lr_spec(router_lr_spec)
        frozen_count = 0
        for layer_indices, lr in groups:
            if lr != 0.0:
                continue
            for i, layer in enumerate(model.decoder.layers):
                if layer_indices is not None and i not in layer_indices:
                    continue
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'router'):
                    layer.mlp.router.weight.requires_grad = False
                    frozen_count += 1
        if frozen_count:
            print_rank_0(f"[router_lr] frozen {frozen_count} router layer(s) with lr=0")

    freeze_layers_spec = getattr(args, 'freeze_layers', None)
    if freeze_layers_spec is not None:
        from megatron.core.optimizer import _parse_layer_list
        frozen_layer_indices = _parse_layer_list(freeze_layers_spec)
        frozen_param_count = 0
        for i, layer in enumerate(model.decoder.layers):
            if i not in frozen_layer_indices:
                continue
            for param in layer.parameters():
                param.requires_grad_(False)
                frozen_param_count += 1
        print_rank_0(f"[freeze_layers] froze {frozen_param_count} params in layers {freeze_layers_spec}")

    return model


def count_parameters_in_layer(model, layer_name):
    num_params = 0
    for name, param in model.named_parameters():
        if layer_name in name:
            num_params += param.numel()
            print_rank_0(f" - {name}: {param.numel()}")
    return num_params
