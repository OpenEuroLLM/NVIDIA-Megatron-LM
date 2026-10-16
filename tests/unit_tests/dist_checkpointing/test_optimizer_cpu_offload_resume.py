# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Resume of the CPU-offloaded distributed optimizer (HybridDeviceOptimizer) from a
distributed checkpoint: the steps after the reload equal the uninterrupted steps."""

from functools import partial
from unittest import mock

import pytest
import torch

from megatron.core.dist_checkpointing import load, save
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.utils import unwrap_model
from megatron.training.arguments import parse_args
from megatron.training.training import get_model
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.dist_checkpointing.utils import init_basic_mock_args, initialize_gpt_model
from tests.unit_tests.test_utilities import Utils


def _setup(seed, offload_fraction, precision_aware):
    args = parse_args(ignore_unknown_args=True)
    with mock.patch('megatron.training.training.get_args', new=lambda: args):
        init_basic_mock_args(args, 1, 1, bf16=True)
        args.gradient_accumulation_fusion = False
        model = get_model(partial(initialize_gpt_model, seed=seed, bf16=True))
    config = OptimizerConfig(
        bf16=True, params_dtype=torch.bfloat16, use_distributed_optimizer=True, lr=1e-2,
        optimizer_cpu_offload=True, optimizer_offload_fraction=offload_fraction,
        overlap_cpu_optimizer_d2h_h2d=True, use_precision_aware_optimizer=precision_aware,
    )
    optimizer = get_megatron_optimizer(config, model)
    return model, optimizer


def _train(model, optimizer, steps):
    for step in steps:
        model[0].zero_grad_buffer()
        gen = torch.Generator(device="cuda").manual_seed(1000 + step)
        for p in model[0].parameters():
            p.main_grad.copy_(torch.randn(p.shape, generator=gen, device="cuda") * 1e-2)
        model[0].finish_grad_sync()
        optimizer.step()
    torch.cuda.synchronize()


def _params(model):
    return {n: p.detach().float().clone() for n, p in unwrap_model(model[0]).named_parameters()}


class TestCpuOffloadResume:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("offload_fraction", [1.0, 0.5])
    @pytest.mark.parametrize("precision_aware", [True, False])
    def test_resume_matches_uninterrupted(self, tmp_path_dist_ckpt, offload_fraction, precision_aware):
        Utils.initialize_model_parallel(1, 1)
        with TempNamedDir(tmp_path_dist_ckpt / 'cpu_offload_resume', sync=True) as ckpt_dir:
            model_a, opt_a = _setup(1, offload_fraction, precision_aware)
            _train(model_a, opt_a, range(3))
            model_sd = unwrap_model(model_a[0]).sharded_state_dict()
            save({"model": model_sd, "optimizer": opt_a.sharded_state_dict(model_sd)}, ckpt_dir)
            _train(model_a, opt_a, range(3, 6))
            ref = _params(model_a)

            model_b, opt_b = _setup(7, offload_fraction, precision_aware)  # other init
            model_sd = unwrap_model(model_b[0]).sharded_state_dict()
            loaded = load(
                {"model": model_sd, "optimizer": opt_b.sharded_state_dict(model_sd, is_loading=True)},
                ckpt_dir,
            )
            unwrap_model(model_b[0]).load_state_dict(loaded["model"])
            opt_b.load_state_dict(loaded["optimizer"])
            _train(model_b, opt_b, range(3, 6))
            got = _params(model_b)
        for name in ref:
            torch.testing.assert_close(got[name], ref[name], rtol=0, atol=0,
                                       msg=lambda m, name=name: f"{name}: {m}")
