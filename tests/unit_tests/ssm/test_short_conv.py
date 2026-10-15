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


def _short_conv(config, tp_group, cp_group):
    spec = hybrid_stack_spec.submodules.short_conv_layer.submodules.self_attention
    pg = ProcessGroupCollection(tp=tp_group, cp=cp_group)
    return ShortConv(config, spec.submodules, layer_number=1, pg_collection=pg).cuda()


def _thd(cu):
    from megatron.core.packed_seq_params import PackedSeqParams

    cu = torch.tensor(cu, dtype=torch.int32, device="cuda")
    return PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)


@pytest.mark.internal
class TestShortConvPacked:
    """A packed batch convolves each document on its own."""

    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN,
            num_attention_heads=4,
            normalization="RMSNorm",
            add_bias_linear=False,
            short_conv_kernel_size=KERNEL,
            transformer_impl="transformer_engine",
        )
        self.conv = _short_conv(
            config,
            parallel_state.get_tensor_model_parallel_group(),
            parallel_state.get_context_parallel_group(),
        )
        yield
        Utils.destroy_model_parallel()

    def test_documents_do_not_mix(self):
        lengths = [5, 7, 4]
        cu = [0, 5, 12, 16]
        hidden = torch.randn(sum(lengths), 1, HIDDEN, device="cuda")
        packed, _ = self.conv(hidden, packed_seq_params=_thd(cu))
        separate = torch.cat([self.conv(hidden[a:b])[0] for a, b in zip(cu[:-1], cu[1:])], dim=0)
        torch.testing.assert_close(packed, separate)


def _cp_chunks(t: torch.Tensor, cp_size: int, rank: int) -> torch.Tensor:
    """This rank's load-balanced sequence chunks (r and 2cp-1-r) of [s, ...]."""
    chunks = t.chunk(2 * cp_size, dim=0)
    return torch.cat([chunks[rank], chunks[2 * cp_size - 1 - rank]], dim=0)


@pytest.mark.internal
@pytest.mark.skipif(Utils.world_size < 2, reason="needs 2 GPUs")
class TestShortConvContextParallel:
    """CP=2 computes exactly what CP=1 computes on the whole sequence."""

    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=2
        )
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN,
            num_attention_heads=4,
            normalization="RMSNorm",
            add_bias_linear=False,
            short_conv_kernel_size=KERNEL,
            context_parallel_size=2,
            transformer_impl="transformer_engine",
        )
        tp = parallel_state.get_tensor_model_parallel_group()
        self.cp_group = parallel_state.get_context_parallel_group()
        rank = torch.distributed.get_rank()
        # A CP=1 twin with identical weights, as the reference.
        solo = [torch.distributed.new_group([r]) for r in range(torch.distributed.get_world_size())]
        self.conv = _short_conv(config, tp, self.cp_group)
        self.ref = _short_conv(config, tp, solo[rank])
        self.ref.load_state_dict(self.conv.state_dict())
        yield
        Utils.destroy_model_parallel()

    def test_matches_full_sequence(self):
        cp, rank = self.cp_group.size(), self.cp_group.rank()
        torch.manual_seed(0)  # same full input on every rank
        hidden = torch.randn(32, 2, HIDDEN, device="cuda")
        out, _ = self.conv(_cp_chunks(hidden, cp, rank))
        ref, _ = self.ref(hidden)
        torch.testing.assert_close(out, _cp_chunks(ref, cp, rank))

    def test_packed_matches_full_sequence(self):
        cp, rank = self.cp_group.size(), self.cp_group.rank()
        cu = [0, 12, 32]  # each document divisible by 2 * cp
        torch.manual_seed(0)
        hidden = torch.randn(32, 1, HIDDEN, device="cuda")
        # THD context parallelism load-balances every document on its own.
        local = torch.cat(
            [_cp_chunks(hidden[a:b], cp, rank) for a, b in zip(cu[:-1], cu[1:])], dim=0
        )
        out, _ = self.conv(local, packed_seq_params=_thd(cu))
        ref, _ = self.ref(hidden, packed_seq_params=_thd(cu))
        expected = torch.cat(
            [_cp_chunks(ref[a:b], cp, rank) for a, b in zip(cu[:-1], cu[1:])], dim=0
        )
        torch.testing.assert_close(out, expected)
