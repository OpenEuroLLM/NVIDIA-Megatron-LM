# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Canon layers (megatron/core/transformer/canon.py) in models and under SP / CP."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.fusions.fused_canon import canon_conv_reference, canon_swiglu_reference
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.canon import Canon
from megatron.core.transformer.enums import AttnBackend
from tests.unit_tests.test_utilities import Utils

H = 64


def _config(**kw):
    return TransformerConfig(
        num_layers=4,
        hidden_size=H,
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
        **kw,
    )


def _model(config, pattern="*-*-"):
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=64,
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    ).cuda()


class TestCanonSingle:
    @pytest.fixture(autouse=True)
    def setup(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("swiglu", [False, True])
    def test_module_matches_reference(self, swiglu):
        tp, cp = parallel_state.get_tensor_model_parallel_group(), None
        canon = Canon(_config(), 32, tp_group=tp, cp_group=cp, swiglu=swiglu)
        with torch.no_grad():
            canon.weight.normal_()
        x = torch.randn(20, 2, 32, device="cuda", requires_grad=True)
        ref_fn = canon_swiglu_reference if swiglu else canon_conv_reference
        w = canon.weight.detach().reshape(32, -1).clone().requires_grad_()
        y, y_ref = canon(x), ref_fn(x, w)
        torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-5)
        g = torch.randn_like(y)
        y.backward(g)
        y_ref.backward(g)
        torch.testing.assert_close(canon.weight.grad.reshape(32, -1), w.grad, rtol=1e-4, atol=1e-4)

    def test_zero_init_is_the_plain_model(self):
        torch.manual_seed(0)
        model_parallel_cuda_manual_seed(7)
        plain = _model(_config())
        model_parallel_cuda_manual_seed(7)
        canon = _model(_config(canon_layers="ABCD"))
        state = {k: v for k, v in canon.state_dict().items() if "canon" not in k}
        missing = canon.load_state_dict(plain.state_dict(), strict=False)
        assert all("canon" in k for k in missing.missing_keys)
        names = [n for n, _ in canon.named_parameters() if "canon" in n]
        assert any("canon_a" in n for n in names) and any("canon_b" in n for n in names)
        assert any("canon_c" in n for n in names) and any("canon_d" in n for n in names)
        ids = torch.randint(0, 128, (2, 32), device="cuda")
        torch.testing.assert_close(
            canon(ids, position_ids=None, attention_mask=None),
            plain(ids, position_ids=None, attention_mask=None),
        )
        del state

    def test_trains_with_all_positions(self):
        model = _model(_config(canon_layers="ABCD"), pattern="*-C-*-")
        ids = torch.randint(0, 128, (2, 32), device="cuda")
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(3):
            loss = model(ids, position_ids=None, attention_mask=None).float().logsumexp(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        grads = [p.grad for n, p in model.named_parameters() if "canon" in n]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(
            bool(p.detach().abs().sum() > 0) for n, p in model.named_parameters() if "canon" in n
        )


@pytest.mark.skipif(Utils.world_size < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("mode", ["sp", "cp"])
@pytest.mark.parametrize("swiglu", [False, True])
def test_distributed_matches_whole_sequence(mode, swiglu):
    """A sequence-sharded (SP halo) or context-parallel Canon layer equals the conv of the
    whole sequence, in its output slice and in its weight gradient."""
    if mode == "sp":
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    else:
        Utils.initialize_model_parallel(context_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(1)
        tp = parallel_state.get_tensor_model_parallel_group()
        cp = parallel_state.get_context_parallel_group()
        rank = (tp if mode == "sp" else cp).rank()
        canon = Canon(
            _config(
                sequence_parallel=(mode == "sp"),
                tensor_model_parallel_size=2 if mode == "sp" else 1,
            ),
            32,
            tp_group=tp,
            cp_group=cp,
            sequence_sharded=(mode == "sp"),
            swiglu=swiglu,
        )
        gen = torch.Generator(device="cuda").manual_seed(3)
        with torch.no_grad():
            canon.weight.copy_(torch.randn(canon.weight.shape, generator=gen, device="cuda"))
        S = 16
        full = torch.randn(S, 2, 32, device="cuda", generator=gen)
        ref_w = canon.weight.detach().reshape(32, -1).clone().requires_grad_()
        ref_fn = canon_swiglu_reference if swiglu else canon_conv_reference
        y_ref = ref_fn(full, ref_w)
        g_full = torch.randn(y_ref.shape, device="cuda", generator=gen)
        y_ref.backward(g_full)
        if mode == "sp":
            idx = torch.arange(rank * S // 2, (rank + 1) * S // 2, device="cuda")
        else:  # context parallelism splits 2*cp chunks, rank r holds r and 2cp-1-r
            chunks = torch.arange(S, device="cuda").chunk(4)
            idx = torch.cat([chunks[rank], chunks[3 - rank]])
        x = full.index_select(0, idx).clone().requires_grad_()
        y = canon(x)
        torch.testing.assert_close(y, y_ref.index_select(0, idx), rtol=1e-5, atol=1e-5)
        y.backward(g_full.index_select(0, idx))
        wgrad = canon.weight.grad.reshape(32, -1).clone()
        group = tp if mode == "sp" else cp
        torch.distributed.all_reduce(wgrad, group=group)  # SP: finalize sums; CP: DDP sums
        torch.testing.assert_close(wgrad, ref_w.grad, rtol=1e-4, atol=1e-4)
    finally:
        Utils.destroy_model_parallel()
