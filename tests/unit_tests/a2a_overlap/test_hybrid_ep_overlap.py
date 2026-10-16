# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HybridModel's EP-overlap (combined 1F1B) schedule plan against the eager forward.

The schedule plan runs one implicit layer group (``*E``, ``*-`` ...) per schedule layer
(``megatron/core/models/hybrid/fine_grained_callables.py``). Two model chunks are run as
the combined-1F1B schedule runs them (forward of chunk 0; forward of chunk 1 overlapped
with the backward of chunk 0; backward of chunk 1) and compared with the eager forward /
backward of the same models: losses and every parameter gradient.

With one rank the experts are not distributed (EP 1) and the dispatch nodes run the local
permutation; with more ranks EP is the world size (the real all-to-all path).
"""

import gc
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan
from megatron.core.models.hybrid.fine_grained_callables import split_hybrid_layer_groups
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import float16_to_fp32
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(
    "pattern, groups",
    [
        ("*E*E", [[0, 1], [2, 3]]),
        ("*-*E", [[0, 1], [2, 3]]),
        ("M*E*", [[0, 1, 2], [3]]),
        ("EE", [[0], [1]]),
        ("**", [[0, 1]]),
    ],
)
def test_split_hybrid_layer_groups(pattern, groups):
    assert split_hybrid_layer_groups(list(pattern)) == groups


def _stub_model(**config_overrides):
    """A HybridModel stand-in for the validation (no layers)."""
    config = dict(
        cuda_graph_impl="none", mhc=False, attn_res=False, ngpt=False, mtp_num_layers=None,
        fp8=None, moe_expert_viability_metrics=False,
    )
    config.update(config_overrides)
    model = torch.nn.Module()
    model.config = SimpleNamespace(**config)
    model.mtp_process = False
    model.decoder = SimpleNamespace(
        layer_type_list=[], layers=[], _loop_schedule=None, per_layer_embeddings=None
    )
    return model


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(cuda_graph_impl="local"), "CUDA graphs"),
        (dict(mhc=True), "mhc"),
        (dict(attn_res=True), "attn_res"),
        (dict(mtp_num_layers=1), "multi-token prediction"),
    ],
)
def test_ep_overlap_rejects_unsupported_features(overrides, message):
    with pytest.raises(ValueError, match=message):
        HybridModel._validate_ep_overlap_support(_stub_model(**overrides))
    HybridModel._validate_ep_overlap_support(_stub_model())


def _build(pattern, ep, seed, shared_expert):
    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    config = TransformerConfig(
        num_layers=len(pattern),
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        normalization="RMSNorm",
        num_moe_experts=4,
        moe_ffn_hidden_size=128,
        moe_grouped_gemm=True,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.01,
        moe_shared_expert_intermediate_size=128 if shared_expert else None,
        moe_token_dispatcher_type="alltoall",
        expert_model_parallel_size=ep,
        is_hybrid_model=True,
    )
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=64,
        hybrid_layer_pattern=pattern,
    ).cuda()
    seq_len = 32
    ids = torch.arange(seq_len, dtype=torch.int64, device="cuda").repeat(2, 1) % 128
    data = dict(
        input_ids=ids,
        position_ids=torch.arange(seq_len, device="cuda").repeat(2, 1),
        attention_mask=None,
        labels=(ids + 1) % 128,
    )
    return model, data


@pytest.mark.parametrize("pattern", ["*E*E", "*-*E", "*E*-*"])
@pytest.mark.parametrize("shared_expert", [False, True])
def test_hybrid_ep_overlap_matches_eager(pattern, shared_expert, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    ep = Utils.world_size
    if ep == 1:
        # Without expert parallelism the all-to-all dispatcher hands its input through, so
        # the dispatch node's early input release would free the experts' input as well.
        import megatron.core.models.common.utils as schedule_utils

        monkeypatch.setattr(schedule_utils, "should_free_input", lambda *args: False)
    Utils.initialize_model_parallel(expert_model_parallel_size=ep)
    try:
        set_streams()
        models, datas, refs = [], [], []
        for i in range(2):
            model, data = _build(pattern, ep, seed=11 + i, shared_expert=shared_expert)
            loss = float16_to_fp32(model(**data))
            loss.backward(torch.ones_like(loss))
            refs.append(
                {"loss": loss.detach().clone()}
                | {n: p.grad.detach().clone() for n, p in model.named_parameters()
                   if p.grad is not None}
            )
            model.zero_grad()
            models.append(model)
            datas.append(data)

        plans = [m.build_schedule_plan(**d) for m, d in zip(models, datas)]
        out0 = TransformerModelChunkSchedulePlan.run(plans[0], None)
        out1 = TransformerModelChunkSchedulePlan.run(
            plans[1], plans[0], b_grad=torch.ones_like(out0)
        )
        TransformerModelChunkSchedulePlan.run(None, plans[1], b_grad=torch.ones_like(out1))
        torch.cuda.synchronize()

        for model, ref, out in zip(models, refs, (out0, out1)):
            torch.testing.assert_close(out, ref["loss"], rtol=1e-2, atol=1e-2)
            for name, param in model.named_parameters():
                if name in ref:
                    assert param.grad is not None, name
                    torch.testing.assert_close(
                        param.grad, ref[name], rtol=2e-2, atol=2e-2,
                        msg=lambda m, name=name: f"{name}: {m}",
                    )
    finally:
        models = plans = None
        gc.collect()
        Utils.destroy_model_parallel()
