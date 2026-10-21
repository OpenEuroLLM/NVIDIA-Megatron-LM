# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""engram_table_parallel="dp": tables sharded over the whole stage, sparse row updates."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.optimizer import engram_optimizer
from megatron.core.optimizer.engram_optimizer import EngramTableOptimizer, TableAdam
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



def _grad(layer):
    """The table gradient as a dense tensor (dense buffer or ENGRAM_SPARSE_GRAD's SparseRowGrad)."""
    g = layer.table_grad
    return g.to_dense() if hasattr(g, "to_dense") else g

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

    def test_lean_states_memory_and_checkpoint_shapes(self):
        """Row-wise second moment + bf16 momentum: 10 B per table parameter instead of 16
        (table + gradient buffer + moments), and checkpoint templates of those shapes."""
        layer = EngramLayer(
            _config(engram_table_parallel="dp"), layer_number=1, pg_collection=self._pg()
        ).cuda()
        nbytes = lambda opt: sum(  # noqa: E731
            t.numel() * t.element_size()
            for st in opt.optimizer.state.values()
            for t in st.values()
            if t.dim()
        )
        params = layer.table.numel()
        fixed = 4 + 4  # fp32 table + fp32 gradient buffer
        lean = {"engram_table_rowwise_v": True, "engram_table_momentum_dtype": "bf16"}
        # row-wise v: 4 B per row = 4 / embed_dim B per parameter
        for kw, per_param in (({}, 16), (lean, 10 + 4 / layer.embed_dim)):
            config = OptimizerConfig(optimizer="adam", lr=1e-3, **kw)
            opt = EngramTableOptimizer([layer.table], config)
            sharded = layer.sharded_state_dict()
            sd = opt.sharded_state_dict(sharded, is_loading=True)
            state = opt.optimizer.state[layer.table]
            bytes_per = fixed + nbytes(opt) / params
            assert bytes_per == pytest.approx(per_param)
            (entry,) = sd["param_state"].values()
            assert entry["exp_avg_sq"].local_shape == tuple(state["exp_avg_sq"].shape)
            assert entry["exp_avg"].dtype == state["exp_avg"].dtype
            assert entry["exp_avg_sq"].global_shape[0] == layer.total_rows


def test_table_adam_matches_torch_adam_over_chunks(monkeypatch):
    monkeypatch.setattr(engram_optimizer, "_CHUNK_ELEMENTS", 40)  # several row chunks
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(23, 8))
    ref = torch.nn.Parameter(w.detach().clone())
    opt = TableAdam([w], lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    ref_opt = torch.optim.Adam([ref], lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    for _ in range(3):
        g = torch.randn(23, 8)
        w.grad, ref.grad = g.clone(), g.clone()
        opt.step()
        ref_opt.step()
    torch.testing.assert_close(w.data, ref.data)


def test_table_adam_rowwise_v_and_bf16_momentum(monkeypatch):
    monkeypatch.setattr(engram_optimizer, "_CHUNK_ELEMENTS", 40)
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(23, 8))
    w0 = w.detach().clone()
    opt = TableAdam([w], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, rowwise_v=True,
                    momentum_dtype=torch.bfloat16)
    m, v = torch.zeros(23, 8), torch.zeros(23, 1)
    for t in range(1, 4):
        g = torch.randn(23, 8)
        w.grad = g.clone()
        opt.step()
        m = 0.9 * m + 0.1 * g
        v = 0.95 * v + 0.05 * g.square().mean(-1, keepdim=True)
        w0 -= 1e-2 / (1 - 0.9**t) * m / ((v / (1 - 0.95**t)).sqrt() + 1e-8)
    state = opt.state[w]
    assert state["exp_avg"].dtype == torch.bfloat16 and state["exp_avg_sq"].shape == (23, 1)
    torch.testing.assert_close(w.data, w0, atol=2e-4, rtol=0)  # bf16 momentum storage


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


@pytest.mark.skipif(Utils.world_size < 2, reason="needs 2+ GPUs")
@pytest.mark.parametrize("sparse", [False, True])
def test_step_lookup_matches_per_microbatch(monkeypatch, sparse):
    """ENGRAM_STEP_LOOKUP: one row exchange per step (prefetch_step_lookup) and one gradient
    exchange at the flush give the per-micro-batch path's outputs and table gradients, over
    three packed (document-masked) micro-batches with different tokens per rank."""
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.ssm.engram_sharded import flush_step_lookups

    monkeypatch.setenv("ENGRAM_SPARSE_GRAD", "1" if sparse else "0")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(123)
        layer = EngramLayer(
            _config(engram_table_parallel="dp"), layer_number=1, pg_collection=_pg()
        ).cuda()
        with torch.no_grad():
            gen = torch.Generator(device="cuda").manual_seed(5)
            layer.conv1d.weight.copy_(
                torch.randn(layer.conv1d.weight.shape, generator=gen, device="cuda")
            )
        dp_rank = parallel_state.get_data_parallel_rank()
        gen = torch.Generator(device="cuda").manual_seed(100 + dp_rank)
        T = 3 * SEQ
        mbs = []
        for k in range(3):
            ids = torch.randint(0, 5000, (1, T), device="cuda", generator=gen)
            cu = torch.tensor([0, 17 + k, 40, 77, T - 2], dtype=torch.int32, device="cuda")
            psp = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                                  max_seqlen_q=40, max_seqlen_kv=40)
            h = torch.randn(T, 1, HIDDEN, device="cuda", generator=gen)
            g = torch.randn(T, 1, HIDDEN, device="cuda", generator=gen)
            mbs.append((ids, cu, psp, h, g))

        def run():
            outs = []
            for ids, cu, psp, h, g in mbs:
                layer.set_input_ids(ids)
                out = layer(h, packed_seq_params=psp)
                out.backward(g)
                outs.append(out.detach())
            return outs

        layer.table_grad.zero_()
        ref_outs = run()
        ref_grad = _grad(layer).clone()
        layer.table_grad.zero_()
        # cu_seqlens as training hands them over ([1, docs + 1], maybe_prefetch_cu_seqlens)
        layer.prefetch_step_lookup([m[0] for m in mbs], [m[1].view(1, -1) for m in mbs])
        outs = run()
        assert _grad(layer).abs().sum() == 0  # nothing exchanged before the flush
        flush_step_lookups()
        for a, b in zip(outs, ref_outs):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(_grad(layer), ref_grad, rtol=1e-4, atol=1e-5)
        assert layer._step_lookup is None
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(Utils.world_size < 2, reason="needs 2+ GPUs")
@pytest.mark.parametrize("sparse", [False, True])
def test_step_lookup_groups_match_per_microbatch(monkeypatch, sparse):
    """ENGRAM_STEP_GROUP: the step lookup in groups of 2 of 5 micro-batches, under a pipelined,
    fully recomputed order (no-grad forwards running ahead, recompute + backward later): the
    per-micro-batch path's outputs and table gradients; a group's gradients reach their owners
    once its last backward ran."""
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.ssm.engram_sharded import PENDING_STEP_LOOKUPS, flush_step_lookups

    monkeypatch.setenv("ENGRAM_SPARSE_GRAD", "1" if sparse else "0")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(123)
        layer = EngramLayer(
            _config(engram_table_parallel="dp"), layer_number=1, pg_collection=_pg()
        ).cuda()
        with torch.no_grad():
            gen = torch.Generator(device="cuda").manual_seed(5)
            layer.conv1d.weight.copy_(
                torch.randn(layer.conv1d.weight.shape, generator=gen, device="cuda")
            )
        dp_rank = parallel_state.get_data_parallel_rank()
        gen = torch.Generator(device="cuda").manual_seed(200 + dp_rank)
        T = 3 * SEQ
        mbs = []
        for k in range(5):
            ids = torch.randint(0, 5000, (1, T), device="cuda", generator=gen)
            cu = torch.tensor([0, 11 + k, 40, 77, T - 2], dtype=torch.int32, device="cuda")
            psp = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                                  max_seqlen_q=40, max_seqlen_kv=40)
            h = torch.randn(T, 1, HIDDEN, device="cuda", generator=gen)
            g = torch.randn(T, 1, HIDDEN, device="cuda", generator=gen)
            mbs.append((ids, cu, psp, h, g))

        def fwd(k, grad):
            ids, _, psp, h, _ = mbs[k]
            layer.set_input_ids(ids)
            with torch.set_grad_enabled(grad):
                return layer(h, packed_seq_params=psp)

        layer.table_grad.zero_()
        ref_outs = []
        for k in range(5):
            out = fwd(k, True)
            out.backward(mbs[k][4])
            ref_outs.append(out.detach())
        ref_grad = _grad(layer).clone()

        layer.table_grad.zero_()
        layer.stash_step_groups([m[0] for m in mbs], [m[1].view(1, -1) for m in mbs], 2)
        outs = {}
        # forwards (no grad, as under full recompute) run ahead of the backwards
        order = [("f", 0), ("f", 1), ("f", 2), ("b", 0), ("f", 3), ("b", 1), ("f", 4),
                 ("b", 2), ("b", 3), ("b", 4)]
        for what, k in order:
            if what == "f":
                fwd(k, False)
                continue
            out = fwd(k, True)  # the recompute, then its backward
            out.backward(mbs[k][4])
            outs[k] = out.detach()
            if k == 0:
                assert _grad(layer).abs().sum() == 0  # group {0, 1} still open
            if k == 1:
                assert _grad(layer).abs().sum() > 0  # ... flushed after its last backward
        flush_step_lookups()  # the last group ({4}) flushed already; nothing left
        assert not PENDING_STEP_LOOKUPS and layer._step_groups is None
        for k in range(5):
            torch.testing.assert_close(outs[k], ref_outs[k])
        torch.testing.assert_close(_grad(layer), ref_grad, rtol=1e-4, atol=1e-5)
    finally:
        Utils.destroy_model_parallel()
