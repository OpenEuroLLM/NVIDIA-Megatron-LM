# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""The hybrid stack under context parallelism on packed (THD) batches: NoPE attention (TE, THD + p2p CP),
Complex KDA and Engram (fla_cp.ThdStream), MoE and mHC-4 at CP=2 give each rank's tokens the logits and the
(CP-summed) parameter gradients that CP=1 gives them -- documents spanning the ranks' stream pieces; TE's ring
attention, an all-gathered dense attention and page-sparse attention. 2 GPUs:

    python -m torch.distributed.run --nproc-per-node 2 -m pytest -q tests/unit_tests/models/test_hybrid_cp_packed.py
"""

import faulthandler
import sys

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm import fla_cp
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

PATTERN = "*ENLELE"
VOCAB = 128
CP = 2
LENGTHS = (96, 200, 72)  # multiples of 2 cp; the middle document spans the two stream pieces

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPUs"),
    pytest.mark.skipif(Utils.world_size < CP, reason=f"needs {CP} GPUs"),
]


def _model(cp, **kw):
    config = TransformerConfig(**{**dict(
        num_layers=len(PATTERN), hidden_size=128, num_attention_heads=4, num_query_groups=2, kv_channels=32,
        linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=4, linear_num_value_heads=4,
        experimental_attention_variant="complex_kda", linear_beta_max=2.0, linear_output_gate="lowrank",
        linear_gate_init_style="shipped", linear_drop_qkv_silu=True, attention_output_gate=True,
        exclusive_self_attention=True, qk_layernorm=True, normalization="RMSNorm", add_bias_linear=False,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu, hidden_dropout=0.0,
        attention_dropout=0.0, num_moe_experts=4, moe_router_topk=2, moe_ffn_hidden_size=64,
        moe_shared_expert_intermediate_size=64, moe_grouped_gemm=True, moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="none", moe_router_score_function="sigmoid", mhc=True,
        mhc_expansion_rate=4, engram_ngram_orders=[2, 3], engram_num_heads=2, engram_table_size=1000,
        engram_embed_dim=16, bf16=True, params_dtype=torch.bfloat16, is_hybrid_model=True,
        context_parallel_size=cp, cp_comm_type="p2p",
    ), **kw})
    return HybridModel(config=config, hybrid_stack_spec=hybrid_stack_spec, vocab_size=VOCAB,
                       max_sequence_length=512, hybrid_layer_pattern=PATTERN, position_embedding_type="none",
                       share_embeddings_and_output_weights=False).cuda().bfloat16()


def _packed():
    cu = torch.tensor([0, *torch.tensor(LENGTHS).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    return PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, cu_seqlens_q_padded=cu,
                           cu_seqlens_kv_padded=cu, max_seqlen_q=max(LENGTHS), max_seqlen_kv=max(LENGTHS))


def _positions():
    return torch.cat([torch.arange(n) for n in LENGTHS]).cuda().unsqueeze(0)


# bf16 model: the CP layouts round differently (FLA's CP kernels, TE's ring attention, the gathers), so the
# largest logit deviates by up to ~10 % of the logits' spread even with exact all-gathered attention; the mean
# stays below 1-2 % and the gradients of everything but the routed experts (whose routing flips at near ties)
# within ~3 %. A wrong mask or state hand-over shows up as an O(1) deviation.
@pytest.mark.parametrize("attention,max_err,mean_max", [
    ({}, 0.15, 0.02),  # TE's ring attention (THD + p2p CP)
    (dict(page_sparse_selector="dense", page_sparse_backend="gathered"), 0.15, 0.01),  # all-gathered K / V
    # the page selection, like the expert routing, flips at near ties between the layouts' roundings
    (dict(page_sparse_selector="page_mean", page_sparse_page=16, page_sparse_topk=2, page_sparse_window=16),
     0.15, 0.02),
], ids=["te_dense", "allgather_dense", "page_sparse"])
def test_cp_matches_single_rank(attention, max_err, mean_max):
    faulthandler.dump_traceback_later(360, exit=True, file=sys.stderr)  # a hang prints every thread's stack
    # CP=1 reference on every rank (the whole batch)
    Utils.initialize_model_parallel(1, 1)  # sets this rank's device: the inputs go there
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB, (1, sum(LENGTHS)), device="cuda")
    w = torch.randn(1, sum(LENGTHS), VOCAB, device="cuda")
    failed = None
    try:
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(1234)
        ref = _model(1, **attention)
        state = {k: v.detach().clone() for k, v in ref.state_dict().items() if v is not None}
        out_ref = ref(ids, position_ids=_positions(), attention_mask=None, packed_seq_params=_packed())
        (out_ref.float() * w).sum().backward()
        grads_ref = {n: p.grad.detach().clone() for n, p in ref.named_parameters() if p.grad is not None}
        out_ref = out_ref.detach().float()
        del ref
    except Exception as exc:  # noqa: BLE001 -- reported below, after every rank agreed
        import traceback

        print(f"[rank {torch.distributed.get_rank()}] reference phase failed:\n{traceback.format_exc()}",
              flush=True)
        failed = exc
    finally:
        Utils.destroy_model_parallel()
    # every rank fails together (one failing alone would leave the others in the next group creation)
    flag = torch.tensor([float(failed is not None)], device="cuda")
    torch.distributed.all_reduce(flag)
    if flag.item() > 0:
        raise RuntimeError("the CP=1 reference phase failed on a rank") from failed
    Utils.initialize_model_parallel(1, 1, context_parallel_size=CP)
    try:
        model_parallel_cuda_manual_seed(123)
        model = _model(CP, **attention)
        model.load_state_dict(state, strict=False)
        params = _packed()
        st = fla_cp.thd_stream(params.cu_seqlens_q_padded, parallel_state.get_context_parallel_group())
        idx = st.local_idx
        out = model(ids[:, idx], position_ids=_positions()[:, idx], attention_mask=None, packed_seq_params=params)
        (out.float() * w[:, idx]).sum().backward()
        cp_group = parallel_state.get_context_parallel_group()
        scale = out_ref.std()
        diff = (out.float() - out_ref[:, idx]).abs()
        errs = torch.stack([diff.max() / scale, diff.mean() / scale]).detach()
        # every rank checks the same numbers (an assert on one rank only would hang the others' collectives)
        torch.distributed.all_reduce(errs, op=torch.distributed.ReduceOp.MAX, group=cp_group)
        rows = []
        for n, p in model.named_parameters():  # the same all-reduces on every rank, in the same order
            if n not in grads_ref or "expert_bias" in n:
                continue
            g = torch.zeros_like(p, dtype=torch.float32) if p.grad is None else p.grad.float().clone()
            torch.distributed.all_reduce(g, group=cp_group)
            ref_g = grads_ref[n].float()
            rows.append((float((g - ref_g).norm() / ref_g.norm().clamp(min=1e-12)), n, float(ref_g.norm()),
                         float(g.norm())))
        # the gradients that matter: parameters whose reference gradient is not negligible (the Sinkhorn biases
        # get ~1e-8: their relative error is noise); and the error of all of them together
        top = max(r[2] for r in rows)
        rows = sorted([r for r in rows if r[2] > 1e-3 * top], reverse=True)
        worst, names = rows[0][0], [rows[0][1]]
        print(f"[cp test] {attention}: logits max {float(errs[0]):.3e} mean {float(errs[1]):.3e} of their "
              f"spread; gradients (relative norm error, |ref|, |cp|): " + "; ".join(
                  f"{n} {e:.2e} {r:.2e} {c:.2e}" for e, n, r, c in rows[:6]))
        assert errs[1] < mean_max, f"logits: mean error {float(errs[1]):.3e} of their spread"
        assert errs[0] < max_err, f"logits: {float(errs[0]):.3e} of their spread"
        assert worst < 0.15, f"{names}: gradient relative norm error {worst:.3e}"
    finally:
        Utils.destroy_model_parallel()
        faulthandler.cancel_dump_traceback_later()
