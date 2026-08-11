# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Looped (weight-tied) depth.

A looped stack applies one built block several times, so effective depth exceeds
parameter depth. Everything that can go wrong here is silent: an off-by-one in the
schedule changes the architecture without raising, and the recompute and CUDA-graph
paths walk ``self.layers`` once and would happily produce wrong activations for a
layer that runs more than once per step. These tests pin the schedule exactly and
require the incompatible paths to fail loudly.
"""

import logging

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.transformer_block import TransformerBlock, build_loop_schedule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

S, B, H = 8, 2, 64


def _config(**kwargs) -> TransformerConfig:
    base = dict(
        num_layers=kwargs.pop("num_layers", 9),
        hidden_size=H,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    base.update(kwargs)
    return TransformerConfig(**base)


class TestLoopSchedule:
    """The schedule is pure config arithmetic, so it is testable without a GPU."""

    def test_disabled_returns_none(self):
        assert build_loop_schedule(_config(num_layers=4)) is None

    def test_prelude_block_coda_order(self):
        """3 + (3)x4 + 3 over 9 unique layers gives effective depth 18."""
        cfg = _config(
            num_layers=9, loop_prelude_layers=3, loop_block_layers=3, loop_iterations=4,
            loop_coda_layers=3,
        )
        assert build_loop_schedule(cfg) == [
            0, 1, 2,
            3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5,
            6, 7, 8,
        ]

    def test_effective_depth_and_uniqueness(self):
        """The three sweep arms: 3x4 and 6x2 are iso-FLOP with 18 layers, 12x2 is iso-param."""
        for prelude, block, iters, coda, unique, effective in [
            (3, 3, 4, 3, 9, 18),
            (3, 6, 2, 3, 12, 18),
            (3, 12, 2, 3, 18, 30),
        ]:
            cfg = _config(
                num_layers=unique, loop_prelude_layers=prelude, loop_block_layers=block,
                loop_iterations=iters, loop_coda_layers=coda,
            )
            schedule = build_loop_schedule(cfg)
            assert len(schedule) == effective
            assert len(set(schedule)) == unique
            assert max(schedule) == unique - 1

    def test_one_iteration_is_the_identity(self):
        """loop_iterations=1 must reproduce the plain stack, or the arm is not a control."""
        cfg = _config(
            num_layers=6, loop_prelude_layers=1, loop_block_layers=4, loop_iterations=1,
            loop_coda_layers=1,
        )
        assert build_loop_schedule(cfg) == list(range(6))

    def test_group_size_moves_the_recurrence_without_changing_cost(self):
        """Same parameters, same FLOPs, different order -- only the schedule changes.

        Group 1 makes a layer consume its own output immediately, which is the case
        the LERP's magnitude invariance is supposed to make well-posed; group == block
        is outer recurrence, where a layer sees the rest of the block first.
        """
        common = dict(
            num_layers=12, loop_prelude_layers=3, loop_block_layers=6, loop_iterations=2,
            loop_coda_layers=3,
        )
        expected = {
            6: [3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8],
            3: [3, 4, 5, 3, 4, 5, 6, 7, 8, 6, 7, 8],
            2: [3, 4, 3, 4, 5, 6, 5, 6, 7, 8, 7, 8],
            1: [3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
        }
        for group, middle in expected.items():
            schedule = build_loop_schedule(_config(loop_group_size=group, **common))
            assert schedule == [0, 1, 2] + middle + [9, 10, 11], group
            # Every group size is iso-cost: same effective depth, same unique layers.
            assert len(schedule) == 18 and len(set(schedule)) == 12

    def test_group_size_zero_is_whole_block(self):
        cfg = dict(
            num_layers=12, loop_prelude_layers=3, loop_block_layers=6, loop_iterations=2,
            loop_coda_layers=3,
        )
        assert build_loop_schedule(_config(loop_group_size=0, **cfg)) == build_loop_schedule(
            _config(loop_group_size=6, **cfg)
        )

    def test_group_size_must_divide_the_block(self):
        with pytest.raises(ValueError, match="must divide"):
            _config(
                num_layers=12, loop_prelude_layers=3, loop_block_layers=6, loop_iterations=2,
                loop_coda_layers=3, loop_group_size=4,
            )

    def test_num_layers_must_count_unique_layers(self):
        with pytest.raises(ValueError, match="unique layers"):
            _config(
                num_layers=18, loop_prelude_layers=3, loop_block_layers=3, loop_iterations=4,
                loop_coda_layers=3,
            )

    def test_pipeline_parallel_rejected(self):
        with pytest.raises(ValueError, match="pipeline_model_parallel_size"):
            _config(
                num_layers=9, loop_prelude_layers=3, loop_block_layers=3, loop_iterations=4,
                loop_coda_layers=3, pipeline_model_parallel_size=2,
            )


class TestLoopedBlock:
    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
    def test_looped_block_trains_and_shares_weights(self):
        """The looped layers must be the same objects and accumulate grad from every pass."""
        model_parallel_cuda_manual_seed(123)
        cfg = _config(
            num_layers=5, loop_prelude_layers=1, loop_block_layers=3, loop_iterations=3,
            loop_coda_layers=1, ngpt=True, ngpt_lerp_normalization="analytic",
        )
        block = TransformerBlock(cfg, get_gpt_layer_with_transformer_engine_spec()).cuda()
        assert block._loop_schedule == [0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4]
        assert len(block.layers) == 5, "looping must not build the block more than once"

        x = torch.randn(S, B, H, device="cuda", requires_grad=True)
        out = block(hidden_states=x, attention_mask=None)
        out.square().mean().backward()
        assert torch.isfinite(out).all()
        # A looped layer sees three backward paths; a prelude layer sees one. Both must
        # receive gradient -- a schedule that skipped the loop would leave the block dry.
        for name, param in block.layers[1].named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"looped layer param without grad: {name}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
    def test_recompute_and_cuda_graphs_are_rejected(self):
        """Both walk self.layers once; silently wrong numbers are worse than a crash."""
        spec = get_gpt_layer_with_transformer_engine_spec()
        loop = dict(
            num_layers=5, loop_prelude_layers=1, loop_block_layers=3, loop_iterations=3,
            loop_coda_layers=1,
        )
        with pytest.raises(NotImplementedError, match="recompute"):
            TransformerBlock(
                _config(recompute_granularity="full", recompute_method="uniform",
                        recompute_num_layers=1, **loop),
                spec,
            )
        with pytest.raises(NotImplementedError, match="CUDA graphs"):
            TransformerBlock(_config(cuda_graph_impl="local", **loop), spec)


class _StubHybridStack:
    """Minimal stand-in for the inputs of ``HybridStack._validate_loop_schedule``.

    The validator reads only ``config``, ``layer_type_list`` and ``layers``.
    Building a real HybridStack would need a GPU, a process group and the
    mLSTM/Mamba kernels, none of which the validation logic depends on.
    """

    def __init__(self, config, pattern):
        self.config = config
        self.layer_type_list = list(pattern)
        self.layers = [object()] * len(pattern)


def _looped_pattern(config, pattern):
    """The layer-type string a looped hybrid stack actually executes."""
    schedule = build_loop_schedule(config)
    return ''.join(pattern[i] for i in schedule)


class TestHybridLoopedStack:
    """HybridStack shares build_loop_schedule but indexes heterogeneous layers.

    In a hybrid stack an index selects a layer *type* as well as a layer, so the
    schedule decides the mixer/MLP composition of the repeated region. These pin
    what a given (prelude, block, iterations, coda) does to the pattern.
    """

    def test_looping_repeats_the_block_pattern(self):
        """3 softmax + (6 alternating)x2 + 3 softmax, over unique layers."""
        pattern = "*-*-*-" + "M-M-M-" + "*-*-*-"
        config = _config(
            num_layers=18, loop_prelude_layers=6, loop_block_layers=6,
            loop_iterations=2, loop_coda_layers=6,
        )
        assert _looped_pattern(config, pattern) == "*-*-*-" + "M-M-M-M-M-M-" + "*-*-*-"

    def test_only_the_linear_block_is_repeated(self):
        """The point of looping a hybrid: softmax layers stay out of the loop.

        KV cache scales with effective depth, so repeating a softmax layer costs
        cache; a linear layer's recurrent state is sequence-independent and costs
        essentially nothing. The softmax count must therefore not change.
        """
        pattern = "*-*-*-" + "M-M-M-" + "*-*-*-"
        config = _config(
            num_layers=18, loop_prelude_layers=6, loop_block_layers=6,
            loop_iterations=2, loop_coda_layers=6,
        )
        looped = _looped_pattern(config, pattern)
        assert looped.count("*") == pattern.count("*") == 6
        assert looped.count("M") == 2 * pattern.count("M") == 6

    def test_alternating_block_does_not_warn(self, caplog):
        """``M-`` repeats cleanly: the block ends on '-' and restarts on 'M'."""
        from megatron.core.models.hybrid.hybrid_block import HybridStack

        config = _config(
            num_layers=18, loop_prelude_layers=6, loop_block_layers=6,
            loop_iterations=2, loop_coda_layers=6,
        )
        stub = _StubHybridStack(config, "*-*-*-" + "M-M-M-" + "*-*-*-")
        with caplog.at_level(logging.WARNING):
            HybridStack._validate_loop_schedule(stub)
        assert "seam" not in caplog.text and "starts and ends" not in caplog.text

    def test_block_starting_and_ending_on_same_type_warns(self, caplog):
        """A block ``M-M`` splices 'M' onto 'M' at every seam -- usually off-by-one."""
        from megatron.core.models.hybrid.hybrid_block import HybridStack

        config = _config(
            num_layers=11, loop_prelude_layers=4, loop_block_layers=3,
            loop_iterations=2, loop_coda_layers=4,
        )
        stub = _StubHybridStack(config, "*-*-" + "M-M" + "-*-*")
        with caplog.at_level(logging.WARNING):
            HybridStack._validate_loop_schedule(stub)
        assert "starts and ends" in caplog.text

    def test_layer_type_count_must_match_built_layers(self):
        from megatron.core.models.hybrid.hybrid_block import HybridStack

        config = _config(
            num_layers=18, loop_prelude_layers=6, loop_block_layers=6,
            loop_iterations=2, loop_coda_layers=6,
        )
        stub = _StubHybridStack(config, "*-*-*-" + "M-M-M-" + "*-*-*-")
        stub.layers = stub.layers[:-1]
        with pytest.raises(ValueError, match="one layer type per built layer"):
            HybridStack._validate_loop_schedule(stub)

    def test_recompute_and_cuda_graphs_are_rejected(self):
        """Same reason as TransformerBlock: both walk self.layers once."""
        from megatron.core.models.hybrid.hybrid_block import HybridStack

        loop = dict(
            num_layers=18, loop_prelude_layers=6, loop_block_layers=6,
            loop_iterations=2, loop_coda_layers=6,
        )
        pattern = "*-*-*-" + "M-M-M-" + "*-*-*-"
        with pytest.raises(NotImplementedError, match="recompute"):
            HybridStack._validate_loop_schedule(
                _StubHybridStack(
                    _config(recompute_granularity="full", recompute_method="uniform",
                            recompute_num_layers=1, **loop),
                    pattern,
                )
            )
        with pytest.raises(NotImplementedError, match="CUDA graphs"):
            HybridStack._validate_loop_schedule(
                _StubHybridStack(_config(cuda_graph_impl="local", **loop), pattern)
            )
