# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Engram n-gram memory, the HybridModel `N` layer (megatron/core/ssm/engram.py)."""

import sys

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.engram import (
    EngramLayer,
    engram_hash,
    engram_hash_multipliers,
    engram_table_sizes,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN = 64


def _config(**kw):
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        engram_ngram_orders=[2, 3],
        engram_num_heads=2,
        engram_table_size=1000,
        engram_embed_dim=16,
        **kw,
    )


class TestHashing:
    def test_sizes_are_distinct_primes(self):
        sizes = engram_table_sizes(_config())
        assert len(sizes) == 4 and len(set(sizes)) == 4 and min(sizes) >= 1000
        assert all(all(n % d for d in range(2, int(n**0.5) + 1)) for n in sizes)

    def test_rows_in_range_and_deterministic(self):
        cfg = _config()
        sizes, mult = engram_table_sizes(cfg), engram_hash_multipliers(cfg)
        ids = torch.randint(0, 50000, (2, 64))
        rows = engram_hash(ids, [2, 3], 2, mult, sizes)
        assert rows.shape == (2, 64, 4)
        for i, m in enumerate(sizes):
            assert int(rows[..., i].min()) >= 0 and int(rows[..., i].max()) < m
        assert torch.equal(rows, engram_hash(ids, [2, 3], 2, engram_hash_multipliers(cfg), sizes))

    def test_ngram_sees_only_its_last_n_tokens(self):
        cfg = _config()
        sizes, mult = engram_table_sizes(cfg), engram_hash_multipliers(cfg)
        ids = torch.randint(0, 50000, (1, 32))
        changed = ids.clone()
        changed[0, 10] += 1
        a = engram_hash(ids, [2, 3], 2, mult, sizes)
        b = engram_hash(changed, [2, 3], 2, mult, sizes)
        bigram, trigram = slice(0, 2), slice(2, 4)  # lookups per order (2 heads each)
        assert torch.equal(a[:, :10], b[:, :10])  # causal
        # x_10 is in the bigrams at t = 10, 11 and the trigrams at t = 10, 11, 12.
        for t in (10, 11):
            assert (a[0, t, bigram] != b[0, t, bigram]).all()
        for t in (10, 11, 12):
            assert (a[0, t, trigram] != b[0, t, trigram]).all()
        assert torch.equal(a[:, 12, bigram], b[:, 12, bigram])
        assert torch.equal(a[:, 13:], b[:, 13:])


@pytest.mark.internal
class TestEngramLayer:

    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        pg = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        )
        self.layer = EngramLayer(_config(), layer_number=1, pg_collection=pg).cuda()
        yield
        Utils.destroy_model_parallel()

    def test_table_marked_for_its_optimizer_group(self):
        assert self.layer.table.is_engram_table
        others = [p for n, p in self.layer.named_parameters() if n != "table"]
        assert others and not any(getattr(p, "is_engram_table", False) for p in others)

    def test_starts_as_gated_value_and_is_causal(self):
        torch.manual_seed(0)
        ids = torch.randint(0, 50000, (2, 24), device="cuda")
        h = torch.randn(24, 2, HIDDEN, device="cuda")
        self.layer.set_input_ids(ids)
        out = self.layer(h)
        assert out.shape == h.shape
        # Zero-initialised conv: Y = SiLU(0) + v~ = v~.
        e = self.layer._memory(ids)
        k, v = self.layer.key_proj(e), self.layer.value_proj(e)
        gate = (self.layer.hidden_norm(h) * self.layer.key_norm(k)).sum(-1, keepdim=True)
        expected = h + torch.sigmoid(gate / HIDDEN**0.5) * v
        torch.testing.assert_close(out, expected)
        # Causal: a later token changes no earlier output.
        with torch.no_grad():
            self.layer.conv1d.weight.normal_()
        base = self.layer(h)
        ids2 = ids.clone()
        ids2[:, 15] += 1
        self.layer.set_input_ids(ids2)
        changed = self.layer(h)
        torch.testing.assert_close(base[:15], changed[:15])
        assert not torch.allclose(base[15], changed[15])

    def test_table_stays_fp32_through_the_bf16_cast(self):
        self.layer.bfloat16()  # what Float16Module does to the whole model
        assert self.layer.table.dtype == torch.float32
        assert self.layer.key_proj.weight.dtype == torch.bfloat16
        ids = torch.randint(0, 50000, (2, 16), device="cuda")
        self.layer.set_input_ids(ids)
        out = self.layer(torch.randn(16, 2, HIDDEN, device="cuda", dtype=torch.bfloat16))
        assert out.dtype == torch.bfloat16

    def test_missing_token_ids_raise(self):
        self.layer.set_input_ids(None)
        with pytest.raises(RuntimeError, match="needs the token ids"):
            self.layer(torch.randn(8, 1, HIDDEN, device="cuda"))


@pytest.mark.internal
@pytest.mark.skipif(Utils.world_size < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("sequence_parallel", [False, True])
class TestEngramTensorParallel:
    """TP=2 (table sharded by rows) computes what one rank with the whole table computes."""

    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self, sequence_parallel):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        self.sp = sequence_parallel
        self.tp_group = parallel_state.get_tensor_model_parallel_group()
        cp = parallel_state.get_context_parallel_group()
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        solo = [torch.distributed.new_group([r]) for r in range(world)]
        self.layer = EngramLayer(
            _config(tensor_model_parallel_size=2, sequence_parallel=sequence_parallel),
            layer_number=1,
            pg_collection=ProcessGroupCollection(tp=self.tp_group, cp=cp),
        ).cuda()
        self.ref = EngramLayer(
            _config(), layer_number=1, pg_collection=ProcessGroupCollection(tp=solo[rank], cp=cp)
        ).cuda()
        # The reference holds the whole table: gather the TP shards.
        shards = [torch.empty_like(self.layer.table) for _ in range(self.tp_group.size())]
        torch.distributed.all_gather(shards, self.layer.table.detach(), group=self.tp_group)
        ref_params = dict(self.ref.named_parameters())
        with torch.no_grad():
            ref_params["table"].copy_(torch.cat(shards, dim=0))
            # A non-zero conv (it starts at zero), the same on every rank.
            gen = torch.Generator(device="cuda").manual_seed(7)
            conv = torch.randn(self.layer.conv1d.weight.shape, generator=gen, device="cuda")
            self.layer.conv1d.weight.copy_(conv)
            for name, p in self.layer.named_parameters():
                if name != "table":
                    ref_params[name].copy_(p)
        yield
        Utils.destroy_model_parallel()

    def test_matches_whole_table(self):
        tp, rank = self.tp_group.size(), self.tp_group.rank()
        torch.manual_seed(0)
        ids = torch.randint(0, 50000, (2, 32), device="cuda")
        h = torch.randn(32, 2, HIDDEN, device="cuda")
        self.layer.set_input_ids(ids)
        self.ref.set_input_ids(ids)
        local = h.chunk(tp, dim=0)[rank] if self.sp else h
        out = self.layer(local)
        ref = self.ref(h)
        expected = ref.chunk(tp, dim=0)[rank] if self.sp else ref
        torch.testing.assert_close(out, expected)


class TestEngramAccounting:
    def test_flops_ignore_the_table(self):
        from megatron.training.arguments import parse_args
        from megatron.training.training import num_floating_point_operations

        def flops(pattern, table_size):
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
                "--engram-table-size",
                str(table_size),
            ]
            try:
                args = parse_args(ignore_unknown_args=True)
            finally:
                sys.argv = argv
            args.padded_vocab_size = 32000
            args.kv_channels = 128
            return num_floating_point_operations(args, 2)

        h, tokens, mem = 1024, 2 * 4096, 64 * 2 * 4  # defaults: dim 64, orders [2, 3], 4 heads
        per_layer = 3 * 2 * tokens * (2 * mem * h + h + 4 * h)
        assert flops("N*-", 1000) - flops("*-", 1000) == per_layer
        assert flops("N*-", 1000) == flops("N*-", 10_000_000)  # table size costs no FLOPs

    def test_optimizer_group(self):
        from megatron.core.optimizer import OptimizerConfig, get_standard_config_overrides

        overrides = get_standard_config_overrides(
            OptimizerConfig(lr=1e-3, min_lr=1e-4, engram_lr_mult=5.0)
        )
        (override,) = [v for k, v in overrides.items() if k.attr == "is_engram_table"]
        assert override == {"wd_mult": 0.0, "max_lr": 5e-3, "min_lr": 5e-4}
