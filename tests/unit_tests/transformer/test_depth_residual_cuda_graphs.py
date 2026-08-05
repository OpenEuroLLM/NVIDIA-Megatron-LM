# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""CUDA-graph safety for the depth-wise residual schemes.

mHC carries an ``n``-stream residual state across every layer of a transformer
block. CUDA graph capture records tensor *operations*, and
:class:`~megatron.core.transformer.cuda_graphs.CudaGraphManager` only staticizes
tensors it can see in a layer's args and kwargs -- so state reached through a
module attribute has its address baked into the captured graph and a replay reads
whichever buffer happened to be live at capture time. That failure is silent: no
exception, just wrong numbers. This test pins the contract by requiring a graph
replay to reproduce eager results exactly.
"""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    _CudagraphGlobalRecord,
    create_cudagraphs,
    delete_cuda_graphs,
)
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

S, B, H, L, N = 16, 2, 64, 4, 3


def _config(cuda_graph_impl: str) -> TransformerConfig:
    kwargs = dict(
        num_layers=L,
        hidden_size=H,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        # Dropout would make the eager and replay runs consume the RNG
        # differently -- graph capture performs extra warmup steps -- so the
        # comparison has to be free of it to isolate the thing under test.
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mhc=True,
        mhc_expansion_rate=N,
        mhc_sinkhorn_iters=4,
    )
    if cuda_graph_impl != "none":
        kwargs["cuda_graph_impl"] = cuda_graph_impl
        kwargs["cuda_graph_warmup_steps"] = 2
    return TransformerConfig(**kwargs)


def _build(cuda_graph_impl: str) -> TransformerBlock:
    model_parallel_cuda_manual_seed(123)
    spec = get_gpt_layer_with_transformer_engine_spec()
    return TransformerBlock(_config(cuda_graph_impl), spec).cuda()


def _fwd_bwd(block: TransformerBlock, x: torch.Tensor):
    """One forward + backward; returns the output and the parameter gradients."""
    out = block(hidden_states=x.clone().requires_grad_(True), attention_mask=None)
    block.zero_grad(set_to_none=True)
    out.square().mean().backward()
    grads = {k: p.grad.clone() for k, p in block.named_parameters() if p.grad is not None}
    return out.detach().clone(), grads


class TestMHCCudaGraphs:
    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(1, 1)
        CudaGraphManager.global_mempool = None

    def teardown_method(self, method):
        delete_cuda_graphs()
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
    def test_graph_replay_matches_eager(self):
        """A captured-and-replayed mHC block must match the eager block exactly.

        Exact equality is the right bar: both runs execute the same kernels on
        the same inputs, so any drift means the graph is reading or writing the
        wrong buffer rather than accumulating differently.
        """
        torch.manual_seed(0)
        x = torch.randn(S, B, H, device="cuda")

        eager = _build("none")
        ref_out, ref_grads = _fwd_bwd(eager, x)

        graphed = _build("local")
        graphed.load_state_dict(eager.state_dict())

        # The first pass records; create_cudagraphs() captures; the next replays.
        _fwd_bwd(graphed, x)
        create_cudagraphs()
        assert _CudagraphGlobalRecord.cudagraph_created, "no cudagraphs were created"
        replay_out, replay_grads = _fwd_bwd(graphed, x)

        torch.testing.assert_close(replay_out, ref_out, rtol=0, atol=0)
        assert set(replay_grads) == set(ref_grads)
        for name, grad in ref_grads.items():
            torch.testing.assert_close(replay_grads[name], grad, rtol=0, atol=0, msg=name)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
    def test_streams_are_not_stashed_on_the_layer(self):
        """The mHC state must not survive on the module between forwards.

        Anything left on the layer is invisible to the graph manager, which is
        precisely what makes a capture unsafe.
        """
        block = _build("none")
        x = torch.randn(S, B, H, device="cuda")
        block(hidden_states=x, attention_mask=None)
        for layer in block.layers:
            assert layer._depth_state is None
