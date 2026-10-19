# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Block AttnRes in its stream form (attn_res_impl streams) through a HybridModel: the same
outputs and gradients as the state form, and CUDA-graph replay = eager."""

import pytest
import torch

from megatron.core.tensor_parallel.random import initialize_rng_tracker
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    _CudagraphGlobalRecord,
    create_cudagraphs,
    delete_cuda_graphs,
)
from tests.unit_tests.models.test_hybrid_depth_memory_graphs import SCOPES, B, S, VOCAB
from tests.unit_tests.models.test_hybrid_depth_memory_graphs import _fwd_bwd
from tests.unit_tests.models.test_hybrid_depth_memory_graphs import _model as _ldm_model
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _model(pattern, impl, graphs=None, blocks=2):
    return _ldm_model(pattern, "torch", graphs=graphs, depth_memory=False, attn_res=True,
                      attn_res_impl=impl, attn_res_num_blocks=blocks, attn_res_query_init_std=0.0)


def _perturb(model):
    with torch.no_grad():  # off the uniform init, so the depth attention matters
        for name, p in model.named_parameters():
            if name.endswith(".query") or name.endswith(".key_norm_weight"):
                p.add_(torch.randn_like(p) * 0.3)


class TestAttnResStreams:
    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(1, 1)
        CudaGraphManager.global_mempool = None

    def teardown_method(self, method):
        delete_cuda_graphs()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("pattern,blocks", [("*E*E", 2), ("*E*E*E", 3), ("*E*E*E", 2)])
    def test_streams_match_the_state_form(self, pattern, blocks):
        torch.manual_seed(0)
        ids = torch.randint(0, VOCAB, (B, S), device="cuda")
        state = _model(pattern, "state", blocks=blocks)
        _perturb(state)
        streams = _model(pattern, "streams", blocks=blocks)
        sd = {}
        for k, v in state.state_dict().items():
            k = k.replace("attnres_attn.", "mhc_attn_mixer.attnres.")
            k = k.replace("attnres_mlp.", "mhc_mlp_mixer.attnres.")
            sd[k] = v
        missing, unexpected = streams.load_state_dict(sd, strict=False)
        assert not [m for m in missing if "_extra_state" not in m], missing
        assert not unexpected, unexpected
        out_a, g_a = _fwd_bwd(state, ids)
        out_b, g_b = _fwd_bwd(streams, ids)
        torch.testing.assert_close(out_b, out_a, rtol=1e-5, atol=1e-5)
        for k, g in g_a.items():
            k2 = k.replace("attnres_attn.", "mhc_attn_mixer.attnres.").replace(
                "attnres_mlp.", "mhc_mlp_mixer.attnres.")
            torch.testing.assert_close(g_b[k2], g, rtol=1e-4, atol=1e-5, msg=k)

    @pytest.mark.parametrize("pattern", list(SCOPES))
    def test_graph_replay_matches_eager(self, pattern):
        torch.manual_seed(0)
        ids = torch.randint(0, VOCAB, (B, S), device="cuda")
        eager = _model(pattern, "streams")
        _perturb(eager)
        ref_out, ref_grads = _fwd_bwd(eager, ids)
        graphed = _model(pattern, "streams", graphs=SCOPES[pattern])
        graphed.load_state_dict(eager.state_dict())
        _fwd_bwd(graphed, ids)
        create_cudagraphs()
        assert _CudagraphGlobalRecord.cudagraph_created, "no cudagraphs were created"
        for _ in range(2):
            out, grads = _fwd_bwd(graphed, ids)
            torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
            for k, g in ref_grads.items():
                torch.testing.assert_close(grads[k], g, rtol=0, atol=0, msg=k)
