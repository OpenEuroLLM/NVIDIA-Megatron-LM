# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""engram_table_parallel="dp": tables sharded over the whole stage, sparse row updates."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.optimizer.engram_optimizer import EngramTableOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.engram import EngramLayer
from megatron.core.ssm.engram_sharded import balanced_row_range, row_owner
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN = 64
SEQ = 32


def _config(**kw):
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        engram_ngram_orders=[2, 3],
        engram_num_heads=2,
        engram_table_size=500,
        engram_embed_dim=8,
        **kw,
    )


def _pg():
    return ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
        tp_dp_cp=parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True),
    )


def test_balanced_rows_and_owner():
    for total, size in ((10, 3), (2007, 4), (5, 8)):
        ranges = [balanced_row_range(total, r, size) for r in range(size)]
        assert ranges[0][0] == 0 and ranges[-1][1] == total
        assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
        rows = torch.arange(total)
        owners = row_owner(rows, total, size)
        for r, (lo, hi) in enumerate(ranges):
            assert bool((owners[lo:hi] == r).all())


class TestEngramShardedSingle:
    """One rank: the sharded path must equal the "tp" path on the same table.

    Under torchrun with several ranks each runs this on its own one-rank groups."""

    @pytest.fixture(autouse=True)
    def setup(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        rank = torch.distributed.get_rank()
        self.solo = [torch.distributed.new_group([r]) for r in range(Utils.world_size)][rank]
        yield
        Utils.destroy_model_parallel()

    def _pg(self):
        return ProcessGroupCollection(
            tp=self.solo, cp=parallel_state.get_context_parallel_group(), tp_dp_cp=self.solo
        )

    def test_matches_tp_mode_forward_and_grad(self):
        torch.manual_seed(0)
        ref = EngramLayer(_config(), layer_number=1, pg_collection=self._pg()).cuda()
        dp = EngramLayer(
            _config(engram_table_parallel="dp"), layer_number=1, pg_collection=self._pg()
        ).cuda()
        assert not dp.table.requires_grad and dp.total_rows == sum(dp.sizes)
        with torch.no_grad():
            ref.conv1d.weight.normal_()
            for name, p in dp.named_parameters():
                if name == "table":
                    p.copy_(ref.table[: dp.total_rows])
                else:
                    p.copy_(dict(ref.named_parameters())[name])
        ids = torch.randint(0, 5000, (2, SEQ), device="cuda")
        h = torch.randn(SEQ, 2, HIDDEN, device="cuda")
        for layer in (ref, dp):
            layer.set_input_ids(ids)
        out_ref, out_dp = ref(h), dp(h)
        torch.testing.assert_close(out_dp, out_ref)
        g = torch.randn_like(out_ref)
        out_ref.backward(g)
        out_dp.backward(g)
        torch.testing.assert_close(dp.table_grad, ref.table.grad[: dp.total_rows].float())

    def test_optimizer_is_adam_at_the_engram_lr(self):
        torch.manual_seed(0)
        layer = EngramLayer(
            _config(engram_table_parallel="dp"), layer_number=1, pg_collection=self._pg()
        ).cuda()
        config = OptimizerConfig(optimizer="adam", lr=1e-3, min_lr=1e-4, engram_lr_mult=5.0)
        opt = EngramTableOptimizer([layer.table], config)
        reference = torch.nn.Parameter(layer.table.detach().clone())
        ref_opt = torch.optim.Adam(
            [reference], lr=5e-3, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_eps
        )
        grad = torch.randn_like(layer.table)
        layer.table_grad.copy_(grad)
        reference.grad = grad.clone()
        opt.prepare_grads()
        opt.step_with_ready_grads()
        ref_opt.step()
        torch.testing.assert_close(layer.table.data, reference.data)
        assert opt.param_groups[0]["max_lr"] == pytest.approx(5e-3)
        assert opt.param_groups[0]["wd_mult"] == 0.0


@pytest.mark.skipif(Utils.world_size < 2, reason="needs 2+ GPUs")
@pytest.mark.parametrize("tp", [1, 2])
def test_sharded_matches_whole_table(tp):
    """Every rank feeds different tokens; each shard's gradient equals the slice of a
    single-rank reference holding the whole table, summed over all ranks' tokens."""
    if Utils.world_size % tp:
        pytest.skip("world size not divisible by tp")
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
    try:
        model_parallel_cuda_manual_seed(123)
        rank = torch.distributed.get_rank()
        pg = _pg()
        layer = EngramLayer(
            _config(engram_table_parallel="dp", tensor_model_parallel_size=tp),
            layer_number=1,
            pg_collection=pg,
        ).cuda()
        # The whole table, gathered, and a solo reference holding it.
        group = pg.tp_dp_cp
        sizes = [
            balanced_row_range(layer.total_rows, r, group.size())[1]
            - balanced_row_range(layer.total_rows, r, group.size())[0]
            for r in range(group.size())
        ]
        shards = [torch.empty(n, layer.embed_dim, device="cuda") for n in sizes]
        torch.distributed.all_gather(shards, layer.table.data.contiguous(), group=group)
        full = torch.cat(shards)
        # new_group must be called identically on every rank
        solo = [torch.distributed.new_group([r]) for r in range(Utils.world_size)][rank]
        ref = EngramLayer(
            _config(engram_table_parallel="dp"),
            layer_number=1,
            pg_collection=ProcessGroupCollection(tp=solo, cp=pg.cp, tp_dp_cp=solo),
        ).cuda()
        with torch.no_grad():
            ref.table.copy_(full)
            gen = torch.Generator(device="cuda").manual_seed(5)
            layer.conv1d.weight.copy_(
                torch.randn(layer.conv1d.weight.shape, generator=gen, device="cuda")
            )
            for name, p in layer.named_parameters():
                if name != "table":
                    dict(ref.named_parameters())[name].copy_(p)

        # Tokens differ per data-parallel rank and are identical within a TP group.
        dp_rank = parallel_state.get_data_parallel_rank()
        gen = torch.Generator(device="cuda").manual_seed(100 + dp_rank)
        ids = torch.randint(0, 5000, (2, SEQ), device="cuda", generator=gen)
        h = torch.randn(SEQ, 2, HIDDEN, device="cuda", generator=gen)
        layer.set_input_ids(ids)
        ref.set_input_ids(ids)
        out, out_ref = layer(h), ref(h)
        torch.testing.assert_close(out, out_ref)
        g = torch.randn(out.shape, device="cuda", generator=gen)
        out.backward(g)
        out_ref.backward(g)
        # Reference gradients from every data-parallel rank's tokens, summed (once per DP
        # rank: TP ranks of a group computed the same thing).
        ref_grad = ref.table_grad.clone()
        if tp > 1 and parallel_state.get_tensor_model_parallel_rank() != 0:
            ref_grad.zero_()
        torch.distributed.all_reduce(ref_grad)
        lo, hi = balanced_row_range(layer.total_rows, group.rank(), group.size())
        torch.testing.assert_close(layer.table_grad, ref_grad[lo:hi], rtol=1e-4, atol=1e-5)
    finally:
        Utils.destroy_model_parallel()
