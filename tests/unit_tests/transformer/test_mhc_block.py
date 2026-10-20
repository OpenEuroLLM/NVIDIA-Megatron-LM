# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""mhc_per_layer (megatron/core/transformer/mhc_block.py): one mHC read / write per block of
HybridStack layers ending at an MLP / MoE layer, plain residual inside the block."""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.mhc import collapse_streams, expand_to_streams
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _model(pattern, **kw):
    config = TransformerConfig(
        num_layers=len(pattern),
        hidden_size=64,
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
        mhc=True,
        mhc_expansion_rate=4,
        mhc_per_layer=True,
        **kw,
    )
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=32,
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    ).cuda()
    for name, p in model.named_parameters():  # input-dependent mixers
        if name.endswith(".phi"):
            with torch.no_grad():
                p.normal_(0, 0.05)
    return model


class TestMHCPerLayer:
    @pytest.fixture(autouse=True)
    def setup(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def test_one_mixer_per_block(self):
        model = _model("*-*--")  # blocks: "*-", "*-", "-" (a one-layer block stays per-sublayer)
        names = [n for n, _ in model.named_parameters() if ".phi" in n]
        assert sorted(names) == [
            "decoder.layers.0.mhc_block_mixer.phi",
            "decoder.layers.2.mhc_block_mixer.phi",
            "decoder.layers.4.mhc_mlp_mixer.phi",
        ], names
        layers = model.decoder.layers
        assert layers[0]._mhc_block is layers[1]._mhc_block
        assert layers[0]._mhc_block_first and layers[1]._mhc_block_last
        assert getattr(layers[4], "_mhc_block", None) is None

    def test_stack_matches_reference(self):
        torch.manual_seed(0)
        model = _model("*-*-")
        stack = model.decoder
        h0 = torch.randn(16, 2, 64, device="cuda")
        out = stack(h0, attention_mask=None)
        # reference: the block written out by hand with the same modules
        s = expand_to_streams(h0, 4)
        for first, last in ((stack.layers[0], stack.layers[1]), (stack.layers[2], stack.layers[3])):
            mixer = first.mhc_block_mixer
            x, _, h_post, h_res = mixer.coeffs_and_read(s)
            kw = dict(attention_mask=None, inference_context=None, rotary_pos_emb=None,
                      sequence_len_offset=None, packed_seq_params=None, padding_mask=None)
            h, _ = first._forward_layer(hidden_states=x, **kw)
            h, _ = last._forward_layer(hidden_states=h, **kw)
            s = mixer.write(s, h - x, h_post, h_res)
        ref = stack.final_norm(collapse_streams(s)) if stack.post_layer_norm else collapse_streams(s)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)

    def test_full_recompute_snaps_to_blocks(self):
        torch.manual_seed(0)
        model = _model("*-*-*-")
        ids = torch.randint(0, 128, (2, 16), device="cuda")
        grads = []
        for recompute in (False, True):
            cfg = model.decoder.config
            cfg.recompute_granularity = "full" if recompute else None
            cfg.recompute_method = "uniform" if recompute else None
            cfg.recompute_num_layers = 1 if recompute else None  # snapped up to whole blocks
            model.zero_grad(set_to_none=True)
            model(ids, position_ids=None, attention_mask=None).float().logsumexp(-1).mean().backward()
            grads.append({n: p.grad.clone() for n, p in model.named_parameters()})
        for name in grads[0]:
            torch.testing.assert_close(grads[1][name], grads[0][name], rtol=1e-4, atol=1e-6,
                                       msg=name)
        assert all(grads[0][n].abs().sum() > 0 for n in grads[0] if n.endswith("block_mixer.phi"))

    def test_trains(self):
        model = _model("*-*-")
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        ids = torch.randint(0, 128, (4, 16), device="cuda")
        losses = []
        for _ in range(8):
            logits = model(ids[:, :-1], position_ids=None, attention_mask=None)
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, 128), ids[:, 1:].reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]

    def test_stage_must_end_with_a_block(self):
        with pytest.raises(ValueError, match="must end with an MLP or MoE layer"):
            _model("*-*")
