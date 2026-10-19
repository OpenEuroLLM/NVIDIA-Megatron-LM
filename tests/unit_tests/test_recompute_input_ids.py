# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Full activation recompute restores the token ids of the micro-batch it recomputes.

Engram (`N`) layers read the ids from an attribute that HybridStack sets once per forward.
Under a pipelined schedule the recompute of micro-batch k runs in its backward, after the
forwards of k+1, k+2, ... replaced those ids: without the capture in checkpointed_forward
the recompute (and so the gradients) used another micro-batch's tokens.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.recompute import checkpointed_forward
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils


class _IdLayer(torch.nn.Module):
    """Stands in for an Engram layer: output = hidden * ids (the gradient shows the ids)."""

    def __init__(self):
        super().__init__()
        self._input_ids = None
        self.layer_number = 1

    def set_input_ids(self, input_ids):
        self._input_ids = input_ids

    def forward(self, hidden_states, **kwargs):
        return hidden_states * self._input_ids.to(hidden_states.dtype).t().unsqueeze(-1)


@pytest.mark.internal
def test_recompute_uses_the_ids_of_its_own_micro_batch():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    try:
        layer = _IdLayer().cuda()
        stack = SimpleNamespace(
            layers=[layer],
            num_layers_per_pipeline_rank=1,
            config=SimpleNamespace(
                fp8=False, fp4=False, distribute_saved_activations=False,
                recompute_method="uniform", recompute_num_layers=1, mhc_routed_topk=0,
            ),
            _input_ids=None,
        )

        def forward(ids, h):
            # what HybridStack.forward does before its checkpointed layers
            stack._input_ids = ids
            layer.set_input_ids(ids)
            return checkpointed_forward(
                stack, hidden_states=h, attention_mask=None, context=None, context_mask=None,
                rotary_pos_emb=None, attention_bias=None, packed_seq_params=None,
                use_inner_quantization_context=False,
            )

        b, s, hdim = 1, 6, 4
        ids_a = torch.arange(1, s + 1, device="cuda").view(b, s)
        ids_b = 100 + ids_a
        h_a = torch.randn(s, b, hdim, device="cuda", requires_grad=True)
        h_b = torch.randn(s, b, hdim, device="cuda", requires_grad=True)
        out_a = forward(ids_a, h_a)
        forward(ids_b, h_b)  # the next micro-batch's forward, before a's backward
        out_a.sum().backward()
        expected = ids_a.t().unsqueeze(-1).float().expand(s, b, hdim)
        torch.testing.assert_close(h_a.grad, expected)
    finally:
        Utils.destroy_model_parallel()
