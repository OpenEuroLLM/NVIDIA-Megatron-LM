# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Product-key memory, the HybridModel `P` layer (megatron/core/ssm/product_key_memory.py)."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer import OptimizerConfig, get_standard_config_overrides
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.engram_sharded import balanced_row_range
from megatron.core.ssm.product_key_memory import ProductKeyMemoryLayer, product_key_topk
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from tests.unit_tests.test_utilities import Utils

HIDDEN = 64
SEQ = 16


def _config(**kw):
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        pkm_heads=2,
        pkm_num_keys=16,
        pkm_key_dim=16,
        pkm_topk=4,
        pkm_value_dim=32,
        **kw,
    )


def _solo_pg():
    rank = torch.distributed.get_rank()
    solo = [torch.distributed.new_group([r]) for r in range(Utils.world_size)][rank]
    return ProcessGroupCollection(
        tp=solo, cp=parallel_state.get_context_parallel_group(), tp_dp_cp=solo
    )


def test_product_key_search_is_exact():
    torch.manual_seed(0)
    T, H, n, d = 7, 3, 12, 8
    q = torch.randn(T, H, d)
    k1, k2 = torch.randn(H, n, d // 2), torch.randn(H, n, d // 2)
    scores, rows = product_key_topk(q, k1, k2, topk=5)
    # Brute force over all n*n product keys.
    s1 = torch.einsum("thd,hnd->thn", q[..., : d // 2], k1)
    s2 = torch.einsum("thd,hnd->thn", q[..., d // 2 :], k2)
    full = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten(-2)  # row = i * n + j
    ref_scores, ref_rows = full.topk(5, dim=-1)
    torch.testing.assert_close(scores, ref_scores)
    torch.testing.assert_close(rows, ref_rows)


class TestPKMSingle:
    @pytest.fixture(autouse=True)
    def setup(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def test_dp_mode_matches_tp_mode(self):
        torch.manual_seed(0)
        pg = _solo_pg()
        ref = ProductKeyMemoryLayer(_config(), layer_number=1, pg_collection=pg).cuda()
        dp = ProductKeyMemoryLayer(
            _config(pkm_table_parallel="dp"), layer_number=1, pg_collection=pg
        ).cuda()
        assert not dp.values.requires_grad and "table" not in dict(dp.named_parameters())
        with torch.no_grad():
            for name, p in dp.named_parameters():
                p.copy_(dict(ref.named_parameters())[name])
            dp.values.copy_(ref.values)
        h = torch.randn(SEQ, 2, HIDDEN, device="cuda")
        out_ref, out_dp = ref(h), dp(h)
        torch.testing.assert_close(out_dp, out_ref)
        g = torch.randn_like(out_ref)
        out_ref.backward(g)
        out_dp.backward(g)
        torch.testing.assert_close(dp.values_grad, ref.values.grad.float())
        # touched rows only
        touched = (ref.values.grad.abs().sum(-1) > 0).sum().item()
        assert 0 < touched <= SEQ * 2 * 2 * 4
        for name in ("keys1", "keys2", "query_proj.weight", "gate_proj.weight"):
            torch.testing.assert_close(
                dict(dp.named_parameters())[name].grad, dict(ref.named_parameters())[name].grad
            )

    def test_memory_table_optimizer_group(self):
        overrides = get_standard_config_overrides(
            OptimizerConfig(lr=1e-3, min_lr=1e-4, pkm_lr_mult=2.0)
        )
        (override,) = [v for k, v in overrides.items() if k.attr == "is_memory_table"]
        assert override == {"wd_mult": 0.0, "max_lr": 2e-3, "min_lr": 2e-4}

    @pytest.mark.parametrize(
        "scheme", [{}, {"mhc": True, "mhc_expansion_rate": 2}, {"attn_res": True}, {"ngpt": True}]
    )
    def test_hybrid_model_with_p_trains(self, scheme):
        config = TransformerConfig(
            num_layers=6,
            hidden_size=HIDDEN,
            num_attention_heads=4,
            ffn_hidden_size=128,
            normalization="RMSNorm",
            add_bias_linear=False,
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            attention_backend=AttnBackend.unfused,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            is_hybrid_model=True,
            pkm_heads=2,
            pkm_num_keys=16,
            pkm_key_dim=16,
            pkm_topk=4,
            **scheme,
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=32,
            hybrid_layer_pattern="P-*-P-",
            position_embedding_type="none",
            share_embeddings_and_output_weights=False,
        ).cuda()
        ids = torch.randint(0, 128, (2, SEQ), device="cuda")
        logits = model(ids, position_ids=None, attention_mask=None)
        assert torch.isfinite(logits).all()
        logits.float().logsumexp(-1).mean().backward()
        pkm = model.decoder.layers[0]
        assert isinstance(pkm, ProductKeyMemoryLayer)
        assert pkm.values.grad is not None and pkm.keys1.grad is not None


@pytest.mark.skipif(Utils.world_size < 2, reason="needs 2+ GPUs")
@pytest.mark.parametrize("mode", ["tp", "dp"])
@pytest.mark.parametrize("tp", [1, 2])
def test_distributed_matches_whole_table(mode, tp):
    """Every gradient (values, keys, query) matches a single-rank reference holding the
    whole table, with different tokens per data-parallel rank."""
    if mode == "tp" and tp == 1:
        pytest.skip("tp mode at TP=1 is the single-rank case")
    if Utils.world_size % tp:
        pytest.skip("world size not divisible by tp")
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
    try:
        model_parallel_cuda_manual_seed(123)
        rank = torch.distributed.get_rank()
        pg = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
            tp_dp_cp=parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True),
        )
        layer = ProductKeyMemoryLayer(
            _config(pkm_table_parallel=mode, tensor_model_parallel_size=tp),
            layer_number=1,
            pg_collection=pg,
        ).cuda()
        group = pg.tp_dp_cp if mode == "dp" else pg.tp
        sizes = [
            balanced_row_range(layer.total_rows, r, group.size())[1]
            - balanced_row_range(layer.total_rows, r, group.size())[0]
            for r in range(group.size())
        ]
        shards = [torch.empty(n, layer.value_dim, device="cuda") for n in sizes]
        torch.distributed.all_gather(shards, layer.values.data.float().contiguous(), group=group)
        ref = ProductKeyMemoryLayer(_config(), layer_number=1, pg_collection=_solo_pg()).cuda()
        with torch.no_grad():
            ref.values.copy_(torch.cat(shards))
            for name, p in layer.named_parameters():
                if name != "values":
                    dict(ref.named_parameters())[name].copy_(p)

        dp_rank = parallel_state.get_data_parallel_rank()
        gen = torch.Generator(device="cuda").manual_seed(100 + dp_rank)
        h = torch.randn(SEQ, 2, HIDDEN, device="cuda", generator=gen)
        out, out_ref = layer(h), ref(h)
        torch.testing.assert_close(out, out_ref, rtol=1e-4, atol=1e-5)
        g = torch.randn(out.shape, device="cuda", generator=gen)
        out.backward(g)
        out_ref.backward(g)
        # Replicated parameters: TP ranks average their gradients in training, so after
        # that average each must equal the reference.
        for name in ("keys1", "query_proj.weight"):
            grad = dict(layer.named_parameters())[name].grad.clone()
            if tp > 1:
                torch.distributed.all_reduce(grad, group=pg.tp)
                grad /= tp
            torch.testing.assert_close(
                grad, dict(ref.named_parameters())[name].grad, rtol=1e-4, atol=1e-5
            )
        # Values: summed over the data-parallel ranks' tokens, compared on this shard.
        ref_grad = ref.values.grad.float().clone()
        if tp > 1 and parallel_state.get_tensor_model_parallel_rank() != 0:
            ref_grad.zero_()
        if mode == "dp":
            torch.distributed.all_reduce(ref_grad)
            mine = layer.values_grad
        else:  # rows sharded over TP only; DDP sums over DP afterwards
            torch.distributed.all_reduce(ref_grad, group=pg.tp)
            mine = layer.values.grad.float()
        lo, hi = balanced_row_range(layer.total_rows, group.rank(), group.size())
        torch.testing.assert_close(mine, ref_grad[lo:hi], rtol=1e-4, atol=1e-5)
    finally:
        Utils.destroy_model_parallel()
