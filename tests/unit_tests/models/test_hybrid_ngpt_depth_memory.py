# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""nGPT's residual (normalized stream + LERP) with Linear Depth Memory through a HybridModel
(depth_memory_lerp, torch stream form): the stream enters on the unit sphere without the weight
normalizer, stays there after every layer (exact 'analytic' LERP), and training runs."""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

VOCAB, S, B = 128, 32, 2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _model(pattern, **kw):
    model_parallel_cuda_manual_seed(123)
    base = dict(
        num_layers=len(pattern), hidden_size=64, num_attention_heads=4, ffn_hidden_size=128,
        use_cpu_initialization=True, hidden_dropout=0.0, attention_dropout=0.0,
        normalization="RMSNorm", add_bias_linear=False, gated_linear_unit=True,
        activation_func=torch.nn.functional.silu, is_hybrid_model=True,
        ngpt=True, ngpt_lerp_normalization="analytic",
    )
    config = TransformerConfig(**{**base, **kw})
    return HybridModel(
        config=config, hybrid_stack_spec=hybrid_stack_spec, vocab_size=VOCAB,
        max_sequence_length=S, hybrid_layer_pattern=pattern, position_embedding_type="rope",
        share_embeddings_and_output_weights=False,
    ).cuda()


class TestNgptDepthMemory:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    # "direct": LERP toward the RAW sublayer output, then renormalize exactly (no post-norm of the
    # output: a vanishing output switches its sublayer off instead of amplifying gradients 1/|a|)
    @pytest.mark.parametrize("variant", ["analytic", "direct"])
    @pytest.mark.parametrize("ldm", [False, True])
    def test_stream_stays_on_the_sphere_and_trains(self, ldm, variant):
        extra = dict(depth_memory=True, depth_memory_impl="torch", depth_memory_groups=4,
                     depth_memory_key_dim=8, depth_memory_hidden_dim=16,
                     depth_memory_lerp=True) if ldm else {}
        if variant == "direct":
            extra.update(ngpt_lerp_normalization="direct", ngpt_post_layer_norm_attn=False,
                         ngpt_post_layer_norm_ffn=False, ngpt_alpha_activation="half_sigmoid")
        model = _model("*-*-", **extra)
        norms = []

        def hook(_m, _inp, out):
            y = out[0] if isinstance(out, tuple) else out
            x = y[..., 0, :] if y.dim() == 4 else y  # LDM streams: slot 0 is the stream x
            norms.append(x.detach().float().norm(dim=-1))

        for layer in model.decoder.layers:
            layer.register_forward_hook(hook)
        ids = torch.randint(0, VOCAB, (B, S), device="cuda")
        pos = torch.arange(S, device="cuda").unsqueeze(0).expand(B, -1)
        for p in model.parameters():
            p.main_grad = torch.zeros_like(p, dtype=torch.float32)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(8):
            norms.clear()
            out = model(ids, position_ids=pos, attention_mask=None, labels=ids)
            loss = out.float().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
            assert len(norms) == 4
            for n in norms:
                torch.testing.assert_close(n, torch.ones_like(n), rtol=0, atol=2e-2)
        assert all(torch.isfinite(torch.tensor(losses)))
        assert losses[-1] < losses[0]
