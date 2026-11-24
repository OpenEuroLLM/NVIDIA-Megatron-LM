# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Per-stream Engram gates under mHC (engram_branch_gates, megatron/core/ssm/engram.py)."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.engram import EngramLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN = 64
N = 4
EXTRA = ("key_proj_extra", "hidden_norm_extra", "key_norm_extra", "gate_terms")


def _config(**kw):
    kw.setdefault("mhc", True)
    kw.setdefault("engram_conv_prenorm", True)
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        engram_ngram_orders=[2, 3],
        engram_num_heads=2,
        engram_table_size=1000,
        engram_embed_dim=16,
        engram_gate_signed_sqrt=True,
        mhc_expansion_rate=N,
        mhc_sinkhorn_iters=4,
        **kw,
    )


def _packed(lens):
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32).cuda()
    return PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                           max_seqlen_q=max(lens), max_seqlen_kv=max(lens))


class TestEngramBranchGates:
    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)
        self.pg = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        )
        yield
        Utils.destroy_model_parallel()

    def _layer(self, **kw):
        return EngramLayer(_config(**kw), layer_number=1, pg_collection=self.pg).cuda()

    def _base(self, **kw):
        """A trained-looking layer without per-stream gates."""
        base = self._layer(**kw)
        with torch.no_grad():
            base.table.normal_()
            base.conv1d.weight.normal_(std=0.3)
            base.hidden_norm.weight.uniform_(0.5, 1.5)
            base.key_norm.weight.uniform_(0.5, 1.5)
            for p in base.residual_scheme.parameters():
                p.add_(torch.randn_like(p) * 0.3)
        return base

    def _warm(self, base, **kw):
        """``base`` continued with per-stream gates: its state, stream 0's parameters copied to the
        other streams (what add_engram_branch_gates.py does to a checkpoint)."""
        layer = self._layer(engram_branch_gates=True, **kw)
        missing, unexpected = layer.load_state_dict(base.state_dict(), strict=False)
        assert not unexpected, unexpected
        assert missing and all(k.startswith(EXTRA) for k in missing), missing
        for name in ("key_proj", "hidden_norm", "key_norm"):
            for extra in getattr(layer, name + "_extra"):
                extra.load_state_dict(getattr(layer, name).state_dict())
        return layer

    def _inputs(self, seed, s=24, b=2):
        torch.manual_seed(seed)
        ids = torch.randint(0, 50000, (b, s), device="cuda")
        return ids, torch.randn(s, b, N, HIDDEN, device="cuda")

    def test_parameters(self):
        layer = self._layer(engram_branch_gates=True, engram_branch_query="learned",
                            engram_gate_scale=True)
        assert len(layer.key_proj_extra) == len(layer.hidden_norm_extra) == N - 1
        assert torch.equal(layer.gate_terms.query_mix, torch.zeros(N, device="cuda"))
        assert torch.equal(layer.gate_terms.scale, torch.ones(N, device="cuda"))
        # fresh layers draw every stream's key projection
        assert not torch.equal(layer.key_proj.weight, layer.key_proj_extra[0].weight)
        # without the option the state is the old one
        assert not any(k.startswith(EXTRA) for k in self._layer().state_dict())

    @pytest.mark.parametrize("query", ["read", "learned"])
    @pytest.mark.parametrize("packed", [False, True])
    @pytest.mark.parametrize("prenorm", [True, False])
    def test_copies_continue_the_layer(self, query, packed, prenorm):
        base = self._base(engram_conv_prenorm=prenorm)
        layer = self._warm(base, engram_branch_query=query, engram_gate_scale=True,
                           engram_conv_prenorm=prenorm)
        if packed:
            lens = [5, 1, 11, 7]
            ids, x = self._inputs(0, s=sum(lens), b=1)
            kw = dict(packed_seq_params=_packed(lens))
        else:
            ids, x = self._inputs(0)
            kw = {}
        base.set_input_ids(ids)
        layer.set_input_ids(ids)
        torch.testing.assert_close(layer(x, **kw), base(x, **kw), rtol=1e-5, atol=1e-5)

    def test_new_parameters_learn(self):
        base = self._base()
        layer = self._warm(base, engram_branch_query="learned", engram_gate_scale=True)
        ids, x = self._inputs(1)
        layer.set_input_ids(ids)
        layer(x).square().sum().backward()
        assert layer.gate_terms.query_mix.grad.abs().min() > 0
        assert layer.gate_terms.scale.grad.abs().min() > 0
        for name in EXTRA[:3]:
            for extra in getattr(layer, name):
                assert extra.weight.grad.abs().sum() > 0, name

    def test_stream_query(self):
        """Stream m's output is the single-gate Engram of stream m with stream m's parameters,
        written to stream m only."""
        base = self._base()
        layer = self._warm(base, engram_branch_query="stream")
        with torch.no_grad():
            layer.key_proj_extra[0].weight.normal_(std=0.1)
            layer.hidden_norm_extra[0].weight.uniform_(0.5, 1.5)
            layer.key_norm_extra[2].weight.uniform_(0.5, 1.5)
        ids, x = self._inputs(2)
        layer.set_input_ids(ids)
        got = layer(x)

        mixer = layer.residual_scheme.mhc_mixer
        h_pre, h_post, h_res = mixer(x)
        outs = []
        for m in range(N):
            if m:
                for name in ("key_proj", "hidden_norm", "key_norm"):
                    getattr(base, name).load_state_dict(getattr(layer, name + "_extra")[m - 1].state_dict())
            outs.append(base._engram(x[:, :, m], ids, None))
        want = mixer.write_streams(x, torch.stack(outs, dim=2), h_post, h_res)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
        # the streams differ, so do their gates
        assert not torch.allclose(outs[1], outs[3])

    def test_write_streams_of_one_output_is_write(self):
        mixer = self._layer().residual_scheme.mhc_mixer
        torch.manual_seed(3)
        x, out = torch.randn(8, 2, N, HIDDEN, device="cuda"), torch.randn(8, 2, HIDDEN, device="cuda")
        _, h_post, h_res = mixer(x)
        torch.testing.assert_close(
            mixer.write_streams(x, out.unsqueeze(2).expand(-1, -1, N, -1), h_post, h_res),
            mixer.write(x, out, h_post, h_res), rtol=1e-5, atol=1e-5,
        )

    def test_needs_dense_mhc(self):
        with pytest.raises(ValueError):
            self._layer(engram_branch_gates=True, mhc=False)
        with pytest.raises(ValueError):
            self._layer(engram_branch_gates=True, engram_branch_query="mean")

    def test_logit_scale_parameter_and_single_gate_invariance(self):
        """engram_gate_logit_scale: the query norm is scale-invariant (eps ~1e-12), and the scalar
        scales the logit before the signed square root."""
        layer = self._layer(engram_gate_logit_scale=True, engram_gate_query_eps=1e-12)
        assert torch.equal(layer.gate_terms.logit_scale, torch.ones(1, device="cuda"))
        with torch.no_grad():
            layer.gate_terms.logit_scale.fill_(0.5)
        ids, x = self._inputs(4)
        h = x[:, :, 0]
        out1 = layer._engram(h, ids, None)
        for c in (3.0, 1e-4):
            torch.testing.assert_close(layer._engram(h * c, ids, None), out1, rtol=1e-5, atol=1e-5)
        # the scalar is applied before the sqrt, so a larger value changes the gate
        with torch.no_grad():
            layer.gate_terms.logit_scale.fill_(8.0)
        assert not torch.allclose(layer._engram(h, ids, None), out1)

    def test_branch_query_scale_invariant(self):
        layer = self._layer(engram_branch_gates=True, engram_gate_query_eps=1e-12)
        ids, x = self._inputs(5)
        e = layer._memory(ids, None).to(x.dtype)
        want = layer._branch_gate_logits(x, e)
        torch.testing.assert_close(layer._branch_gate_logits(x * 3.0, e), want, rtol=1e-5, atol=1e-5)
        # hn ~ 1e-4: mean(q^2) ~ 1e-8, so eps=1e-12 leaves a ~1e-4 relative residual
        torch.testing.assert_close(layer._branch_gate_logits(x * 1e-4, e), want, rtol=1e-3, atol=1e-4)

    def test_logit_scale_learns(self):
        base = self._base()
        layer = self._warm(base, engram_branch_query="read", engram_gate_logit_scale=True)
        assert torch.equal(layer.gate_terms.logit_scale, torch.ones(N, device="cuda"))
        ids, x = self._inputs(6)
        layer.set_input_ids(ids)
        layer(x).square().sum().backward()
        assert layer.gate_terms.logit_scale.grad.abs().sum() > 0
