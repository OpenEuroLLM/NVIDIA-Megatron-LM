# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Gated short convolution (LFM2), the HybridModel `C` mixer."""

import sys

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.short_conv import ShortConv
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN = 64
KERNEL = 3


@pytest.mark.internal
class TestShortConv:

    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        self.config = TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN,
            num_attention_heads=4,
            normalization="RMSNorm",
            add_bias_linear=False,
            short_conv_kernel_size=KERNEL,
            transformer_impl="transformer_engine",
        )
        pg_collection = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        )
        spec = hybrid_stack_spec.submodules.short_conv_layer.submodules.self_attention
        self.conv = ShortConv(
            self.config, spec.submodules, layer_number=1, pg_collection=pg_collection
        ).cuda()
        yield
        Utils.destroy_model_parallel()

    def test_matches_lfm2_reference(self):
        """Same weights, same output as Hugging Face's Lfm2ShortConv after the input norm."""
        lfm2 = pytest.importorskip("transformers.models.lfm2.modeling_lfm2")
        ref = lfm2.Lfm2ShortConv(
            lfm2.Lfm2Config(hidden_size=HIDDEN, conv_L_cache=KERNEL, conv_bias=False), layer_idx=0
        ).cuda()
        with torch.no_grad():
            ref.in_proj.weight.copy_(self.conv.in_proj.weight)
            ref.conv.weight.copy_(self.conv.conv1d.weight)
            ref.out_proj.weight.copy_(self.conv.out_proj.weight)

        seq, batch = 16, 2
        hidden = torch.randn(seq, batch, HIDDEN, device="cuda")
        out, bias = self.conv(hidden)
        assert out.shape == hidden.shape and bias is None

        # in_proj fuses the pre-mixer RMSNorm; the reference expects normed input.
        eps = self.config.layernorm_epsilon
        normed = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + eps)
        normed = normed * self.conv.in_proj.layer_norm_weight
        expected = ref.slow_forward(normed.transpose(0, 1)).transpose(0, 1)
        torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-3)

    def test_causal(self):
        """Changing a token changes no earlier output."""
        hidden = torch.randn(16, 1, HIDDEN, device="cuda")
        changed = hidden.clone()
        changed[10] += 1.0
        a, _ = self.conv(hidden)
        b, _ = self.conv(changed)
        torch.testing.assert_close(a[:10], b[:10])
        assert not torch.allclose(a[10], b[10])

    def test_sharded_state_dict_splits_streams(self):
        sharded = self.conv.sharded_state_dict(prefix="mixer.")
        assert isinstance(sharded["mixer.in_proj.weight"], ShardedTensorFactory)
        assert isinstance(sharded["mixer.conv1d.weight"], ShardedTensor)
        assert sharded["mixer.conv1d.weight"].global_shape == (HIDDEN, 1, KERNEL)


class TestShortConvFlops:
    def test_flops_term(self):
        from megatron.training.arguments import parse_args
        from megatron.training.training import num_floating_point_operations

        def flops(pattern):
            argv = sys.argv
            sys.argv = [
                "pretrain_hybrid.py",
                f"--hybrid-layer-pattern={pattern}",
                "--hidden-size",
                "1024",
                "--num-attention-heads",
                "8",
                "--seq-length",
                "4096",
                "--max-position-embeddings",
                "4096",
                "--micro-batch-size",
                "1",
                "--ffn-hidden-size",
                "4096",
            ]
            try:
                args = parse_args(ignore_unknown_args=True)
            finally:
                sys.argv = argv
            args.padded_vocab_size = 32000
            args.kv_channels = 128
            return num_floating_point_operations(args, 2)

        h, k, tokens = 1024, 3, 2 * 4096
        per_layer = 3 * 2 * tokens * (h * 3 * h + k * h + h * h)  # fwd+bwd, FMA
        assert flops("C-C-*-") - flops("--*-") == 2 * per_layer
