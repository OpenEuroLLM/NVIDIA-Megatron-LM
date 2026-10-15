# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""FLA mixers under TP>1: every TP rank runs the whole layer (tp_replicated.py)."""

import copy

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm import fla_hybrid
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN, HEADS, SEQ, BATCH = 128, 4, 64, 2


@pytest.mark.internal
@pytest.mark.skipif(fla_hybrid._FLAKDA is None, reason="needs fla.layers.kda")
@pytest.mark.skipif(Utils.world_size < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("sequence_parallel", [False, True])
class TestReplicatedKDA:

    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self, sequence_parallel):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        self.sp = sequence_parallel
        self.tp_group = parallel_state.get_tensor_model_parallel_group()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN,
            num_attention_heads=HEADS,
            linear_key_head_dim=HIDDEN // HEADS,
            linear_num_key_heads=HEADS,
            tensor_model_parallel_size=2,
            sequence_parallel=sequence_parallel,
            bf16=True,
            params_dtype=torch.bfloat16,
            experimental_attention_variant="kda",
            is_hybrid_model=True,
        )
        pg = ProcessGroupCollection(
            tp=self.tp_group, cp=parallel_state.get_context_parallel_group()
        )
        self.mixer = fla_hybrid.KDA(config, layer_number=1, pg_collection=pg).cuda()
        # The single-copy reference: the same FLA layer, run on the whole sequence.
        self.ref = copy.deepcopy(self.mixer.layer)
        yield
        Utils.destroy_model_parallel()

    def test_replicas_start_identical(self):
        for p in self.mixer.parameters():
            gathered = [torch.empty_like(p) for _ in range(self.tp_group.size())]
            torch.distributed.all_gather(gathered, p.detach().contiguous(), group=self.tp_group)
            assert all(torch.equal(g, gathered[0]) for g in gathered)
            assert getattr(p, "average_gradients_across_tp_domain", False)

    def test_forward_and_gradients_match_single_copy(self):
        tp, rank = self.tp_group.size(), self.tp_group.rank()
        torch.manual_seed(0)  # same full input on every rank
        full = torch.randn(SEQ, BATCH, HIDDEN, device="cuda", dtype=torch.bfloat16)

        local = full.chunk(tp, dim=0)[rank] if self.sp else full
        local = local.clone().requires_grad_(True)
        out, bias = self.mixer(local)
        assert bias is None
        # Under SP each rank's loss covers its slice; the slices add up to the full loss.
        out.float().sum().backward()

        ref_in = full.clone().requires_grad_(True)
        ref_out = self.ref(hidden_states=ref_in.transpose(0, 1), attention_mask=None)[0]
        ref_out = ref_out.transpose(0, 1)
        ref_out.float().sum().backward()

        expected = ref_out.chunk(tp, dim=0)[rank] if self.sp else ref_out
        torch.testing.assert_close(out, expected.detach(), atol=2e-2, rtol=2e-2)
        expected_grad = ref_in.grad.chunk(tp, dim=0)[rank] if self.sp else ref_in.grad
        # A reduce-scatter in the gather's backward would make this tp times too large.
        torch.testing.assert_close(local.grad, expected_grad, atol=2e-2, rtol=2e-2)
        for (name, p), (_, rp) in zip(
            self.mixer.layer.named_parameters(), self.ref.named_parameters()
        ):
            if rp.grad is None:
                continue
            torch.testing.assert_close(p.grad, rp.grad, atol=5e-2, rtol=5e-2, msg=name)
