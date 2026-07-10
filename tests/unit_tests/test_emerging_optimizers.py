# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.emerging_optimizers import (
    HAVE_EMERGING_OPTIMIZERS,
    TensorParallelAdaptiveMuon,
    TensorParallelAngularMuown,
    TensorParallelMuon,
    get_supported_coefficient_types,
    validate_coefficient_type,
)
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

if HAVE_EMERGING_OPTIMIZERS:
    from emerging_optimizers.scalar_optimizers import Lion
    from emerging_optimizers.soap import SOAP
else:
    SOAP = None
    Lion = None

# Skip all tests in this file for LTS versions or when emerging_optimizers is missing
pytestmark = [
    pytest.mark.skipif(
        Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
        reason="Skip emerging optimizer tests for LTS test",
    ),
    pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package is not installed"
    ),
]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(80, 48)
        self.fc2 = nn.Linear(48, 32)
        self.fc3 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 16)
        self.fc5 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# ===========================================================================
# Muon optimizer tests
# ===========================================================================


def test_muon_optimizer_smoke():
    """Smoke test for TensorParallelMuon optimizer."""
    # Create a simple linear model for testing
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    # Create TensorParallelMuon optimizer
    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.01,
        use_decoupled_weight_decay=True,
        split_qkv=False,
        fp32_matmul_prec="medium",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        pg_collection=None,
        tp_mode="duplicated",
    )

    # Test basic properties
    assert optimizer is not None, "Optimizer should not be None"
    assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
    assert len(optimizer.param_groups) > 0, "Optimizer should have at least one parameter group"

    # Test forward and backward pass
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    # Store original weight
    original_weight = model.weight.data.clone()

    # Test optimizer step
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    # Test zero_grad
    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    # Test state_dict and load_state_dict
    state_dict = optimizer.state_dict()
    assert 'state' in state_dict, "State dict should contain state"
    assert 'param_groups' in state_dict, "State dict should contain param_groups"

    # Load state dict should not raise error
    optimizer.load_state_dict(state_dict)


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonOptimizerMultiRank:
    """Test class for Muon optimizer with multi-rank setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        """Wrap model in DDP.

        Args:
            model: Model to wrap

        Returns:
            DDP-wrapped model
        """
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    def test_get_megatron_optimizer_smoke(self):
        """Smoke test for get_megatron_optimizer function."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # Ensure all parameters require gradients
        for param in model.parameters():
            assert param.requires_grad, "All parameters should require gradients"

        # Create optimizer config for Muon
        optimizer_config = OptimizerConfig(
            optimizer='muon',  # This will be changed internally to 'adam' for non-linear params
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,  # Muon doesn't support distributed optimizer
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
        )

        # Test creating the optimizer
        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        # Test basic properties
        assert optimizer is not None, "Optimizer should not be None"
        assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
        assert hasattr(optimizer, 'chained_optimizers'), "Should be a ChainedOptimizer"
        assert len(optimizer.chained_optimizers) >= 1, "Should have at least one chained optimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        # Test optimizer step
        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated after optimizer step"

        # Test zero_grad
        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None or torch.all(
                param.grad == 0
            ), f"Gradients should be zeroed for all parameters"

        # Test state_dict and load_state_dict
        state_dict = optimizer.state_dict()
        assert isinstance(state_dict, list), "State dict should be a list"

        # Load state dict should not raise error
        optimizer.load_state_dict(state_dict)

    def test_get_megatron_optimizer_validation(self):
        """Test validation logic for get_megatron_optimizer."""
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.bfloat16, device='cuda')
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # Test 1: FP16 should raise exception
        optimizer_config_fp16 = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            fp16=True,  # This should cause an exception
            use_distributed_optimizer=False,
        )

        with pytest.raises(Exception, match='emerging optimizer with fp16 is not supported'):
            get_megatron_optimizer(config=optimizer_config_fp16, model_chunks=[model])

        # Test 3: Invalid num_ns_steps should raise exception
        optimizer_config_invalid_ns = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_num_ns_steps=0,  # This should cause an exception
        )

        with pytest.raises(ValueError, match='num_ns_steps must be at least 1'):
            get_megatron_optimizer(config=optimizer_config_invalid_ns, model_chunks=[model])

    def test_get_megatron_optimizer_layer_wise(self):
        """Test get_megatron_optimizer with layer-wise distributed optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_layer_wise_distributed_optimizer=True,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
        )

        # use_layer_wise_distributed_optimizer=True triggers LayerWiseDistributedOptimizer
        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        # Verify it's a LayerWiseDistributedOptimizer
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        assert isinstance(
            optimizer, LayerWiseDistributedOptimizer
        ), "Should return LayerWiseDistributedOptimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Test optimizer step
        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"
        assert grad_norm is not None or grad_norm is None, "Grad norm should be returned"

    def test_get_megatron_muon_optimizer_backward_compatible(self):
        """Test get_megatron_muon_optimizer with backward compatible layer-wise distributed optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_layer_wise_distributed_optimizer=True,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
        )

        with pytest.raises(ValueError, match="dist_ prefix"):
            get_megatron_muon_optimizer(
                config=optimizer_config, model_chunks=[model], layer_wise_distributed_optimizer=True
            )

        optimizer_config.optimizer = 'dist_muon'
        optimizer = get_megatron_muon_optimizer(
            config=optimizer_config, model_chunks=[model], layer_wise_distributed_optimizer=True
        )

        # Verify it's a LayerWiseDistributedOptimizer
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        assert isinstance(
            optimizer, LayerWiseDistributedOptimizer
        ), "Should return LayerWiseDistributedOptimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Test optimizer step
        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"
        assert grad_norm is not None or grad_norm is None, "Grad norm should be returned"


@pytest.mark.parametrize("mode", ["duplicated", "blockwise", "distributed"])
def test_muon_optimizer_different_modes_single_rank(mode):
    """Test TensorParallelMuon optimizer with different modes on single rank.

    When TP size is 1, all modes should produce the same result.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.normal_(0, 0.02)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.0,  # Disable weight decay for deterministic comparison
        num_ns_steps=5,
        pg_collection=None,
        tp_mode=mode,
    )

    # Use fixed input for deterministic results
    torch.manual_seed(42)
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with mode={mode}"


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonOptimizerMultiRankTP:
    """Test class for Muon optimizer with multi-rank and tensor parallel setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test with tensor parallel."""
        world = int(os.getenv('WORLD_SIZE', '1'))
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(world, 2))
        yield
        Utils.destroy_model_parallel()

    def create_tp_model_and_optimizer(self, mode):
        """Create model with TP and optimizer.

        Args:
            mode: Muon optimizer mode

        Returns:
            tuple: (model, optimizer, pg_collection)
        """
        rank = int(os.getenv('RANK', '0'))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        # Create model with partition_dim for TP
        torch.manual_seed(42 + rank)
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
        model.requires_grad_(True)
        model.weight.data.normal_(0, 0.02)
        model.weight.partition_dim = 0  # Set partition dimension for TP

        optimizer = TensorParallelMuon(
            params=[model.weight],
            lr=0.01,
            momentum=0.95,
            weight_decay=0.0,
            num_ns_steps=5,
            pg_collection=pg_collection,
            tp_mode=mode,
        )

        return model, optimizer

    @pytest.mark.parametrize("mode", ["duplicated", "distributed"])
    def test_muon_optimizer_modes_multirank_same_result(self, mode):
        """Test that duplicated and distributed modes produce same results with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer(mode)

        # Use fixed input for deterministic results
        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        # Verify weight was updated
        assert not torch.equal(
            model.weight.data, original_weight
        ), f"Weight should be updated with mode={mode}"

    def test_muon_optimizer_blockwise_mode_different_result(self):
        """Test that blockwise mode produces different results than duplicated/distributed with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer("blockwise")

        # Use fixed input for deterministic results
        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        # Verify weight was updated
        assert not torch.equal(
            model.weight.data, original_weight
        ), "Weight should be updated with mode=blockwise"


# All non-custom coefficient types supported by emerging_optimizers.
_TESTABLE_COEFFICIENT_TYPES = (
    [t for t in get_supported_coefficient_types() if t != "custom"]
    if HAVE_EMERGING_OPTIMIZERS
    else []
)

# A reasonable default NS step count for testing; get_coefficient_iterator
# cycles/repeats coefficients so any step count works with any type.
_DEFAULT_NS_STEPS = 5


@pytest.mark.parametrize("coefficient_type", _TESTABLE_COEFFICIENT_TYPES)
def test_muon_optimizer_coefficient_types(coefficient_type):
    """Test TensorParallelMuon optimizer with different coefficient types."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        coefficient_type=coefficient_type,
        num_ns_steps=_DEFAULT_NS_STEPS,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with coefficient_type={coefficient_type}"


@pytest.mark.parametrize("scale_mode", ["spectral", "unit_rms_norm", "shape_scaling"])
def test_muon_optimizer_scale_modes(scale_mode):
    """Test TensorParallelMuon optimizer with different scale modes."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        scale_mode=scale_mode,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with scale_mode={scale_mode}"


@pytest.mark.parametrize("nesterov", [True, False])
def test_muon_optimizer_nesterov(nesterov):
    """Test TensorParallelMuon optimizer with and without Nesterov momentum."""
    model = torch.nn.Linear(50, 25, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.9,
        nesterov=nesterov,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 50, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with nesterov={nesterov}"


def test_muon_optimizer_multiple_steps():
    """Test TensorParallelMuon optimizer across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    # Verify weights changed at each step
    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


def test_muon_optimizer_qkv_split():
    """Test TensorParallelMuon optimizer with QKV splitting."""
    # Create a model with QKV-like parameter
    qkv_size = 3 * 64 * 16  # Combined Q, K, V dimensions, 16 heads x 64 per head
    hidden_size = 1024
    model = torch.nn.Linear(hidden_size, qkv_size, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    # Mark parameter as QKV
    model.weight.is_qkv = True

    # QKV split shapes: [Q_size, K_size, V_size]
    qkv_split_shapes = (64, 64, 64)

    # Test with split_qkv=True
    optimizer_split = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, hidden_size, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer_split.step()
    weight_with_split = model.weight.data.clone()

    assert not torch.equal(
        weight_with_split, original_weight
    ), "QKV weight should be updated with split_qkv=True"

    # Reset model and test with split_qkv=False
    model.weight.data.fill_(1.0)
    optimizer_no_split = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=False,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    optimizer_no_split.step()
    weight_without_split = model.weight.data.clone()

    assert not torch.equal(
        weight_without_split, original_weight
    ), "QKV weight should be updated with split_qkv=False"

    # Ensure the two results are different
    assert not torch.equal(
        weight_with_split, weight_without_split
    ), "Weights should be different between split_qkv=True and split_qkv=False"


def test_muon_optimizer_extra_scale_factor():
    """Test TensorParallelMuon optimizer with different extra_scale_factor values."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        extra_scale_factor=2.0,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated with extra_scale_factor"


def test_muon_batched_step_matches_per_param():
    """The opt-in batched Muon step must reproduce the upstream per-param path.

    Two identical optimizers (batched_step=True vs the default False) over a
    mix of shapes — same-shape stacks, a singleton batch, and a fused-QKV
    weight taking the split path — with decoupled weight decay and Nesterov
    momentum on. With fp32_matmul_prec="highest" the Newton-Schulz runs in
    fp32, so bmm-vs-mm only differ by fp32 reduction order and tolerances can
    be tight. Params and momentum buffers are compared after every step.
    """

    def make_params(seed):
        torch.manual_seed(seed)
        spec = [((96, 64), False)] * 3 + [((64, 48), False)] * 2
        spec += [((128, 64), False), ((96, 64), True)]  # singleton + qkv
        params = []
        for shape, is_qkv in spec:
            p = torch.nn.Parameter(torch.randn(shape, device='cuda') * 0.02)
            p.is_qkv = is_qkv
            params.append(p)
        return params

    def make_opt(params, batched):
        return TensorParallelMuon(
            params=params,
            lr=0.01,
            momentum=0.95,
            nesterov=True,
            weight_decay=0.1,
            use_decoupled_weight_decay=True,
            split_qkv=True,
            is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
            qkv_split_shapes=(32, 8, 8),
            num_ns_steps=5,
            coefficient_type="simple",
            fp32_matmul_prec="highest",
            pg_collection=None,
            tp_mode="duplicated",
            batched_step=batched,
        )

    params_ref = make_params(11)
    params_batched = make_params(11)
    opt_ref = make_opt(params_ref, batched=False)
    opt_batched = make_opt(params_batched, batched=True)

    for step in range(3):
        torch.manual_seed(200 + step)
        for p in params_ref:
            p.grad = torch.randn_like(p) * 1e-3
        torch.manual_seed(200 + step)
        for p in params_batched:
            p.grad = torch.randn_like(p) * 1e-3
        opt_ref.step()
        opt_batched.step()

        for p_ref, p_b in zip(params_ref, params_batched):
            torch.testing.assert_close(p_b, p_ref, rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(
                opt_batched.state[p_b]["momentum_buffer"],
                opt_ref.state[p_ref]["momentum_buffer"],
                rtol=1e-5,
                atol=1e-7,
            )


def test_get_supported_coefficient_types_returns_tuple():
    """Test that get_supported_coefficient_types returns a non-empty tuple of strings."""
    supported = get_supported_coefficient_types()
    assert isinstance(supported, tuple)
    assert len(supported) > 0
    for t in supported:
        assert isinstance(t, str)


def test_get_supported_coefficient_types_contains_known_types():
    """Test that the known coefficient types are present in the supported set."""
    supported = get_supported_coefficient_types()
    for expected in ("simple", "quintic", "polar_express"):
        assert expected in supported, f"Expected '{expected}' in supported types {supported}"


def test_validate_coefficient_type_accepts_valid():
    """Test that validate_coefficient_type does not raise for valid types."""
    for t in get_supported_coefficient_types():
        validate_coefficient_type(t)  # should not raise


def test_validate_coefficient_type_rejects_invalid():
    """Test that validate_coefficient_type raises ValueError for an invalid type."""
    with pytest.raises(ValueError, match="Unsupported muon coefficient type"):
        validate_coefficient_type("nonexistent_type_xyz")


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonCoefficientTypeMultiRank:
    """Test coefficient_type integration through get_megatron_optimizer."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    @pytest.mark.parametrize("coefficient_type", _TESTABLE_COEFFICIENT_TYPES)
    def test_get_megatron_optimizer_coefficient_type(self, coefficient_type):
        """Test that coefficient_type flows through get_megatron_optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_coefficient_type=coefficient_type,
            muon_num_ns_steps=_DEFAULT_NS_STEPS,
            muon_tp_mode="duplicated",
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        assert optimizer is not None

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()


@pytest.mark.parametrize("num_ns_steps", [5, 15, 25])
def test_muon_optimizer_num_ns_steps(num_ns_steps):
    """Test TensorParallelMuon optimizer with different numbers of Newton-Schulz steps."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        coefficient_type="quintic",
        num_ns_steps=num_ns_steps,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with num_ns_steps={num_ns_steps}"


# ===========================================================================
# Adaptive Muon optimizer tests
# ===========================================================================


def test_adaptive_muon_optimizer_smoke():
    """Smoke test for TensorParallelAdaptiveMuon optimizer."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.01,
        use_decoupled_weight_decay=True,
        split_qkv=False,
        fp32_matmul_prec="medium",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        pg_collection=None,
        tp_mode="duplicated",
        moment2_method="adamuon",
        beta2=0.95,
        eps=1e-8,
    )

    assert optimizer is not None
    assert hasattr(optimizer, 'param_groups')
    assert len(optimizer.param_groups) > 0

    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    state_dict = optimizer.state_dict()
    assert 'state' in state_dict
    assert 'param_groups' in state_dict
    optimizer.load_state_dict(state_dict)


@pytest.mark.parametrize("mode", ["duplicated", "blockwise", "distributed"])
def test_adaptive_muon_optimizer_different_modes_single_rank(mode):
    """Test TensorParallelAdaptiveMuon with different modes on single rank."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.normal_(0, 0.02)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.0,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode=mode,
    )

    torch.manual_seed(42)
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with mode={mode}"


@pytest.mark.parametrize("moment2_method", ["adamuon", "normuon"])
def test_adaptive_muon_optimizer_moment2_methods(moment2_method):
    """Test TensorParallelAdaptiveMuon with different moment2 methods."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
        moment2_method=moment2_method,
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with moment2_method={moment2_method}"


@pytest.mark.parametrize("beta2", [0.5, 0.95, 0.999])
def test_adaptive_muon_optimizer_beta2(beta2):
    """Test TensorParallelAdaptiveMuon with different beta2 values."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
        beta2=beta2,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with beta2={beta2}"


def test_adaptive_muon_optimizer_multiple_steps():
    """Test TensorParallelAdaptiveMuon across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


@pytest.mark.parametrize("nesterov", [True, False])
def test_adaptive_muon_optimizer_nesterov(nesterov):
    """Test TensorParallelAdaptiveMuon with and without Nesterov momentum."""
    model = torch.nn.Linear(50, 25, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.9,
        nesterov=nesterov,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 50, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with nesterov={nesterov}"


def test_adaptive_muon_optimizer_qkv_split():
    """Test TensorParallelAdaptiveMuon with QKV splitting."""
    qkv_size = 3 * 64 * 16  # Combined Q, K, V dimensions
    hidden_size = 1024
    model = torch.nn.Linear(hidden_size, qkv_size, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    model.weight.is_qkv = True
    qkv_split_shapes = (64, 64, 64)

    optimizer_split = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, hidden_size, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer_split.step()
    weight_with_split = model.weight.data.clone()

    assert not torch.equal(
        weight_with_split, original_weight
    ), "QKV weight should be updated with split_qkv=True"

    model.weight.data.fill_(1.0)
    optimizer_no_split = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=False,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    optimizer_no_split.step()
    weight_without_split = model.weight.data.clone()

    assert not torch.equal(
        weight_without_split, original_weight
    ), "QKV weight should be updated with split_qkv=False"

    assert not torch.equal(
        weight_with_split, weight_without_split
    ), "Weights should be different between split_qkv=True and split_qkv=False"


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestAdaptiveMuonOptimizerMultiRank:
    """Test class for Adaptive Muon optimizer with multi-rank setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        """Wrap model in DDP."""
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    def test_get_megatron_optimizer_adaptive_muon_smoke(self):
        """Smoke test for get_megatron_optimizer with adaptive_muon."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        for param in model.parameters():
            assert param.requires_grad

        optimizer_config = OptimizerConfig(
            optimizer='adaptive_muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
            adaptive_muon_moment2_method="adamuon",
            adaptive_muon_beta2=0.95,
            adaptive_muon_eps=1e-8,
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        assert optimizer is not None
        assert hasattr(optimizer, 'param_groups')
        assert hasattr(optimizer, 'chained_optimizers')
        assert len(optimizer.chained_optimizers) >= 1

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        optimizer.step()

        params_updated = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated after optimizer step"

        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None or torch.all(
                param.grad == 0
            ), "Gradients should be zeroed for all parameters"

        state_dict = optimizer.state_dict()
        assert isinstance(state_dict, list)
        optimizer.load_state_dict(state_dict)

    def test_get_megatron_optimizer_adaptive_muon_validation(self):
        """Test validation logic for get_megatron_optimizer with adaptive_muon."""
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.bfloat16, device='cuda')
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer_config_fp16 = OptimizerConfig(
            optimizer='adaptive_muon', lr=0.01, fp16=True, use_distributed_optimizer=False
        )

        with pytest.raises(Exception, match='emerging optimizer with fp16 is not supported'):
            get_megatron_optimizer(config=optimizer_config_fp16, model_chunks=[model])


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestAdaptiveMuonOptimizerMultiRankTP:
    """Test class for Adaptive Muon optimizer with multi-rank and tensor parallel setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test with tensor parallel."""
        world = int(os.getenv('WORLD_SIZE', '1'))
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(world, 2))
        yield
        Utils.destroy_model_parallel()

    def create_tp_model_and_optimizer(self, mode):
        """Create model with TP and optimizer."""
        rank = int(os.getenv('RANK', '0'))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        torch.manual_seed(42 + rank)
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
        model.requires_grad_(True)
        model.weight.data.normal_(0, 0.02)
        model.weight.partition_dim = 0

        optimizer = TensorParallelAdaptiveMuon(
            params=[model.weight],
            lr=0.01,
            momentum=0.95,
            weight_decay=0.0,
            num_ns_steps=5,
            pg_collection=pg_collection,
            tp_mode=mode,
        )

        return model, optimizer

    @pytest.mark.parametrize("mode", ["duplicated", "distributed"])
    def test_adaptive_muon_optimizer_modes_multirank_same_result(self, mode):
        """Test that duplicated and distributed modes produce same results with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer(mode)

        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        assert not torch.equal(
            model.weight.data, original_weight
        ), f"Weight should be updated with mode={mode}"

    def test_adaptive_muon_optimizer_blockwise_mode(self):
        """Test that blockwise mode works with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer("blockwise")

        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        assert not torch.equal(
            model.weight.data, original_weight
        ), "Weight should be updated with mode=blockwise"


# ===========================================================================
# SOAP optimizer tests
# ===========================================================================

skip_no_soap = pytest.mark.skipif(
    not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package not installed"
)


@skip_no_soap
def test_soap_optimizer_smoke():
    """Smoke test for SOAP optimizer."""

    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(
        params=[model.weight],
        lr=0.01,
        betas=(0.9, 0.999),
        shampoo_beta=0.95,
        weight_decay=0.01,
        precondition_frequency=1,
    )

    # Test basic properties
    assert optimizer is not None, "Optimizer should not be None"
    assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
    assert len(optimizer.param_groups) > 0, "Optimizer should have at least one parameter group"

    # Test forward and backward pass
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    # Store original weight
    original_weight = model.weight.data.clone()

    # Test optimizer step
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    # Test zero_grad
    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    # Test state_dict and load_state_dict
    state_dict = optimizer.state_dict()
    assert 'state' in state_dict, "State dict should contain state"
    assert 'param_groups' in state_dict, "State dict should contain param_groups"

    # Load state dict should not raise error
    optimizer.load_state_dict(state_dict)


@skip_no_soap
def test_soap_optimizer_multiple_steps():
    """Test SOAP optimizer across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(
        params=[model.weight],
        lr=0.01,
        betas=(0.9, 0.999),
        shampoo_beta=0.95,
        weight_decay=0.01,
        precondition_frequency=1,
    )

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    # Verify weights changed at each step
    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


@skip_no_soap
@pytest.mark.parametrize("precondition_frequency", [1, 5, 10])
def test_soap_optimizer_precondition_frequency(precondition_frequency):
    """Test SOAP optimizer with different precondition frequencies."""

    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(
        params=[model.weight],
        lr=0.01,
        betas=(0.9, 0.999),
        shampoo_beta=0.95,
        precondition_frequency=precondition_frequency,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with precondition_frequency={precondition_frequency}"


@skip_no_soap
@pytest.mark.parametrize("use_kl_shampoo", [True, False])
def test_soap_optimizer_kl_shampoo(use_kl_shampoo):
    """Test SOAP optimizer with and without KL-Shampoo preconditioner."""

    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(
        params=[model.weight],
        lr=0.01,
        betas=(0.9, 0.999),
        shampoo_beta=0.95,
        use_kl_shampoo=use_kl_shampoo,
        precondition_frequency=1,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with use_kl_shampoo={use_kl_shampoo}"


@skip_no_soap
@pytest.mark.parametrize("shampoo_beta", [0.5, 0.9, 0.99])
def test_soap_optimizer_shampoo_beta(shampoo_beta):
    """Test SOAP optimizer with different shampoo_beta values."""

    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(
        params=[model.weight],
        lr=0.01,
        betas=(0.9, 0.999),
        shampoo_beta=shampoo_beta,
        precondition_frequency=1,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with shampoo_beta={shampoo_beta}"


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestSoapOptimizerMultiRank:
    """Test class for SOAP optimizer with multi-rank setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        """Wrap model in DDP."""
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    def test_get_megatron_optimizer_soap_smoke(self):
        """Smoke test for get_megatron_optimizer with SOAP."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        for param in model.parameters():
            assert param.requires_grad, "All parameters should require gradients"

        optimizer_config = OptimizerConfig(
            optimizer='soap',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            soap_shampoo_beta=0.95,
            soap_precondition_frequency=1,
            soap_use_kl_shampoo=True,
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        assert optimizer is not None, "Optimizer should not be None"
        assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
        assert hasattr(optimizer, 'chained_optimizers'), "Should be a ChainedOptimizer"
        assert len(optimizer.chained_optimizers) >= 1, "Should have at least one chained optimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        # Test optimizer step
        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated after optimizer step"

        # Test zero_grad
        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None or torch.all(
                param.grad == 0
            ), "Gradients should be zeroed for all parameters"

        # Test state_dict and load_state_dict
        state_dict = optimizer.state_dict()
        assert isinstance(state_dict, list), "State dict should be a list"
        optimizer.load_state_dict(state_dict)

    def test_get_megatron_optimizer_soap_validation(self):
        """Test validation logic for get_megatron_optimizer with SOAP."""
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.bfloat16, device='cuda')
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # FP16 should raise exception
        optimizer_config_fp16 = OptimizerConfig(
            optimizer='soap', lr=0.01, fp16=True, use_distributed_optimizer=False
        )

        with pytest.raises(Exception, match='emerging optimizer with fp16 is not supported'):
            get_megatron_optimizer(config=optimizer_config_fp16, model_chunks=[model])


# ===========================================================================
# Lion optimizer tests
# ===========================================================================

skip_no_lion = pytest.mark.skipif(
    not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package not installed"
)


@skip_no_lion
def test_lion_optimizer_smoke():
    """Smoke test for Lion optimizer."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(params=[model.weight], lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)

    assert optimizer is not None
    assert hasattr(optimizer, 'param_groups')
    assert len(optimizer.param_groups) > 0

    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    state_dict = optimizer.state_dict()
    assert 'state' in state_dict
    assert 'param_groups' in state_dict
    optimizer.load_state_dict(state_dict)


@skip_no_lion
def test_lion_optimizer_multiple_steps():
    """Test Lion optimizer across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(params=[model.weight], lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


@skip_no_lion
@pytest.mark.parametrize("betas", [(0.9, 0.99), (0.95, 0.999), (0.5, 0.9)])
def test_lion_optimizer_betas(betas):
    """Test Lion optimizer with different beta values."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(params=[model.weight], lr=1e-4, betas=betas)

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with betas={betas}"


@skip_no_lion
@pytest.mark.parametrize("weight_decay", [0.0, 0.01, 0.1])
def test_lion_optimizer_weight_decay(weight_decay):
    """Test Lion optimizer with different weight decay values."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(params=[model.weight], lr=1e-4, betas=(0.9, 0.99), weight_decay=weight_decay)

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with weight_decay={weight_decay}"


@skip_no_lion
@pytest.mark.parametrize("weight_decay_method", ["decoupled", "l2"])
def test_lion_optimizer_weight_decay_method(weight_decay_method):
    """Test Lion optimizer with different weight decay methods."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(
        params=[model.weight],
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        weight_decay_method=weight_decay_method,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with weight_decay_method={weight_decay_method}"


@skip_no_lion
def test_lion_optimizer_multi_layer_net():
    """Test Lion optimizer with the multi-layer Net model."""
    model = Net().cuda()
    model.requires_grad_(True)

    optimizer = Lion(params=model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_params = {name: p.data.clone() for name, p in model.named_parameters()}
    optimizer.step()

    params_updated = 0
    for name, param in model.named_parameters():
        if not torch.equal(param.data, original_params[name]):
            params_updated += 1

    assert params_updated > 0, "At least some parameters should be updated after optimizer step"


# ===========================================================================
# AngularMuown optimizer tests
# ===========================================================================


def test_angular_muown_optimizer_smoke():
    """Smoke test for TensorParallelAngularMuown optimizer."""
    torch.manual_seed(123)
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.normal_(0, 0.02)

    optimizer = TensorParallelAngularMuown(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        nesterov=True,
        betas=(0.9, 0.95),
        adam_eps=1e-8,
        num_ns_steps=5,
        coefficient_type="simple",
        scale_mode="shape_scaling",
        pg_collection=None,
        tp_mode="duplicated",
    )

    assert optimizer is not None
    assert len(optimizer.param_groups) > 0

    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    # AngularMuown invariant: row norms of W equal |g| (rows of U have unit norm).
    g = optimizer.state[model.weight]["g"]
    row_norms = model.weight.data.norm(dim=1, keepdim=True)
    assert torch.allclose(row_norms, g.abs(), rtol=1e-5, atol=1e-6), (
        "Row norms of W should equal |g| after a AngularMuown step"
    )

    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(model.weight.grad == 0)

    state_dict = optimizer.state_dict()
    assert 'state' in state_dict and 'param_groups' in state_dict
    optimizer.load_state_dict(state_dict)


def test_angular_muown_optimizer_rejects_non_2d():
    """AngularMuown should reject non-2D parameters."""
    bias_like = torch.nn.Parameter(torch.randn(16, device='cuda'))
    with pytest.raises(ValueError, match='only supports 2D parameters'):
        TensorParallelAngularMuown(params=[bias_like], lr=0.01, pg_collection=None)


def test_angular_muown_state_dict_round_trip():
    """state_dict expands per-row states to the weight shape; load collapses them back.

    Exercises the torch_dist-checkpoint-compatibility logic: g/m_g/v_g are stored
    live as (rows, 1) but saved as the weight's full (rows, cols) so they inherit
    the weight's sharding metadata. The expanded columns are identical copies and
    the load round-trip is exact.
    """
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(32, 64, dtype=torch.float32, device='cuda') * 0.02)
    optimizer = TensorParallelAngularMuown(
        params=[weight], lr=0.01, pg_collection=None, tp_mode="duplicated"
    )
    weight.grad = torch.randn_like(weight)
    optimizer.step()

    rows, cols = weight.shape
    live_state = optimizer.state[weight]
    # Live state stays reduced (rows, 1); m_u matches the weight shape.
    for key in ("g", "m_g", "v_g"):
        assert live_state[key].shape == (rows, 1)
    assert live_state["m_u"].shape == (rows, cols)

    saved = optimizer.state_dict()
    saved_state = saved["state"][0]
    # Per-row states are expanded to full shape with identical columns.
    for key in ("g", "m_g", "v_g"):
        assert saved_state[key].shape == (rows, cols)
        assert torch.equal(saved_state[key], live_state[key].expand(-1, cols))
    assert saved_state["m_u"].shape == (rows, cols)
    # Saving must not mutate the live reduced-shape states.
    for key in ("g", "m_g", "v_g"):
        assert live_state[key].shape == (rows, 1)

    # A fresh optimizer loads the expanded checkpoint and collapses it exactly.
    weight2 = torch.nn.Parameter(weight.detach().clone())
    optimizer2 = TensorParallelAngularMuown(
        params=[weight2], lr=0.01, pg_collection=None, tp_mode="duplicated"
    )
    optimizer2.load_state_dict(saved)
    loaded_state = optimizer2.state[weight2]
    for key in ("g", "m_g", "v_g"):
        assert loaded_state[key].shape == (rows, 1)
        assert torch.equal(loaded_state[key], live_state[key])
    assert torch.equal(loaded_state["m_u"], live_state["m_u"])
    assert loaded_state["step"] == live_state["step"]


def test_angular_muown_optimizer_u_decay_schedules():
    """U-decay schedule multipliers should decay as configured."""
    model = torch.nn.Linear(32, 16, bias=False, dtype=torch.float32, device='cuda')
    model.weight.data.normal_(0, 0.02)

    optimizer = TensorParallelAngularMuown(
        params=[model.weight],
        lr=0.01,
        pg_collection=None,
        tp_mode="duplicated",
        u_decay_schedule="poly",
        u_decay_scale=1.0,
        u_decay_p=1.0,
    )

    multipliers = []
    for _ in range(3):
        model.weight.grad = torch.randn_like(model.weight)
        optimizer.step()
        multipliers.append(optimizer.param_groups[0]["u_lr_multiplier"])

    # (1 + steps_after_warmup) ** -1 evaluated at steps 0, 1, 2.
    assert multipliers == pytest.approx([1.0, 0.5, 1.0 / 3.0])

    with pytest.raises(ValueError, match="requires u_decay_steps"):
        TensorParallelAngularMuown(
            params=[torch.nn.Parameter(torch.randn(8, 8, device='cuda'))],
            u_decay_schedule="cosine",
            pg_collection=None,
        )


def test_angular_muown_optimizer_qkv_split():
    """Test TensorParallelAngularMuown with QKV splitting."""
    qkv_size = 3 * 64 * 16
    hidden_size = 1024
    model = torch.nn.Linear(hidden_size, qkv_size, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.normal_(0, 0.02)
    model.weight.is_qkv = True

    optimizer_split = TensorParallelAngularMuown(
        params=[model.weight],
        lr=0.01,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=(64, 64, 64),
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, hidden_size, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer_split.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), "QKV weight should be updated with split_qkv=True"

    g = optimizer_split.state[model.weight]["g"]
    row_norms = model.weight.data.norm(dim=1, keepdim=True)
    assert torch.allclose(row_norms, g.abs(), rtol=1e-5, atol=1e-6)


def test_angular_muown_batched_step_matches_per_param():
    """The batched step must reproduce the per-parameter reference path.

    Builds two identical optimizers (one with batched_step=True, one False)
    over a mix of shapes — several same-shape weights that form stacks, a
    singleton batch, and a fused-QKV weight taking the split path — and
    checks params and all optimizer state stay equal across steps. With
    fp32_matmul_prec="highest" the Newton-Schulz runs in fp32, so bmm-vs-mm
    only differ by fp32 reduction order and tolerances can be tight.
    """

    def make_params(seed):
        torch.manual_seed(seed)
        spec = [((96, 64), False)] * 3 + [((64, 48), False)] * 2
        spec += [((128, 64), False), ((96, 64), True)]  # singleton + qkv
        params = []
        for shape, is_qkv in spec:
            p = torch.nn.Parameter(torch.randn(shape, device='cuda') * 0.02)
            p.is_qkv = is_qkv
            params.append(p)
        return params

    def make_opt(params, batched):
        return TensorParallelAngularMuown(
            params=params,
            lr=0.01,
            momentum=0.95,
            nesterov=True,
            betas=(0.9, 0.95),
            split_qkv=True,
            is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
            qkv_split_shapes=(32, 8, 8),
            num_ns_steps=5,
            coefficient_type="simple",
            fp32_matmul_prec="highest",
            pg_collection=None,
            tp_mode="duplicated",
            u_decay_schedule="poly",
            u_decay_scale=0.001,
            batched_step=batched,
        )

    params_ref = make_params(7)
    params_batched = make_params(7)
    opt_ref = make_opt(params_ref, batched=False)
    opt_batched = make_opt(params_batched, batched=True)

    for step in range(3):
        torch.manual_seed(100 + step)
        for p in params_ref:
            p.grad = torch.randn_like(p) * 1e-3
        torch.manual_seed(100 + step)
        for p in params_batched:
            p.grad = torch.randn_like(p) * 1e-3
        opt_ref.step()
        opt_batched.step()

        for p_ref, p_b in zip(params_ref, params_batched):
            torch.testing.assert_close(p_b, p_ref, rtol=1e-5, atol=1e-7)
            s_ref, s_b = opt_ref.state[p_ref], opt_batched.state[p_b]
            assert s_ref["step"] == s_b["step"]
            for key in ("g", "m_u", "m_g", "v_g"):
                torch.testing.assert_close(s_b[key], s_ref[key], rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestAngularMuownOptimizerMultiRank:
    """Test class for AngularMuown optimizer with multi-rank setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        """Wrap model in DDP."""
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    def _make_config(self, **kwargs):
        return OptimizerConfig(
            optimizer='angular_muown',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            angular_muown_momentum=0.95,
            angular_muown_nesterov=True,
            angular_muown_fp32_matmul_prec="medium",
            angular_muown_num_ns_steps=5,
            angular_muown_scale_mode="shape_scaling",
            angular_muown_tp_mode="duplicated",
            **kwargs,
        )

    def test_get_megatron_optimizer_smoke(self):
        """Hybrid AngularMuown + Adam optimizer via get_megatron_optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer = get_megatron_optimizer(
            config=self._make_config(), model_chunks=[model], use_gloo_process_groups=True
        )

        assert optimizer is not None
        assert hasattr(optimizer, 'chained_optimizers'), "Should be a ChainedOptimizer"
        assert len(optimizer.chained_optimizers) == 2, "Should chain AngularMuown and Adam"

        # Verify the hybrid split: AngularMuown gets only 2D params, Adam gets the rest.
        inner_types = {}
        for sub in optimizer.chained_optimizers:
            inner = sub.optimizer
            inner_types[type(inner).__name__] = inner
        assert 'TensorParallelAngularMuown' in inner_types, f"Got {list(inner_types)}"
        angular_muown_inner = inner_types['TensorParallelAngularMuown']
        for group in angular_muown_inner.param_groups:
            for p in group['params']:
                assert p.ndim == 2, "AngularMuown bucket should only contain 2D params"
        adam_inner = [v for k, v in inner_types.items() if k != 'TensorParallelAngularMuown'][0]
        adam_ndims = {p.ndim for g in adam_inner.param_groups for p in g['params']}
        assert 1 in adam_ndims, "Adam bucket should hold the biases"

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_params = {name: p.data.clone() for name, p in model.named_parameters()}
        optimizer.step()

        params_updated = sum(
            0 if torch.equal(p.data, original_params[name]) else 1
            for name, p in model.named_parameters()
        )
        assert params_updated > 0, "Parameters should be updated after optimizer step"

        optimizer.zero_grad()
        state_dict = optimizer.state_dict()
        optimizer.load_state_dict(state_dict)

    def test_moe_router_and_experts_routing(self):
        """Router/gate weights must go to Adam; per-expert 2D weights to AngularMuown."""

        class MoeLikeNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = nn.Linear(64, 8, bias=False)  # name: router.weight [8, 64]
                self.gate_weight = nn.Parameter(torch.randn(1, 64))
                self.experts = nn.ModuleList(nn.Linear(64, 64, bias=False) for _ in range(4))
                self.proj = nn.Linear(64, 64, bias=True)

        model = MoeLikeNet().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer = get_megatron_optimizer(
            config=self._make_config(), model_chunks=[model], use_gloo_process_groups=True
        )

        angular_muown_params, adam_params = set(), set()
        for sub in optimizer.chained_optimizers:
            inner = sub.optimizer
            target = (
                angular_muown_params if isinstance(inner, TensorParallelAngularMuown) else adam_params
            )
            for group in inner.param_groups:
                for p in group['params']:
                    # Map main (fp32) params back to model params via shape+id walk.
                    target.add(p.data_ptr())

        name_to_main_ptr = {
            name: getattr(p, 'main_param', p).data_ptr()
            for name, p in model.named_parameters()
        }
        assert name_to_main_ptr['module.router.weight'] in adam_params, "router must use Adam"
        assert name_to_main_ptr['module.gate_weight'] in adam_params, "gate must use Adam"
        assert name_to_main_ptr['module.proj.bias'] in adam_params, "bias must use Adam"
        for i in range(4):
            assert (
                name_to_main_ptr[f'module.experts.{i}.weight'] in angular_muown_params
            ), f"expert {i} weight must use AngularMuown"
        assert name_to_main_ptr['module.proj.weight'] in angular_muown_params

    def test_get_megatron_optimizer_layer_wise(self):
        """AngularMuown through the layer-wise distributed optimizer (AngularMuownDP path)."""
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer = get_megatron_optimizer(
            config=self._make_config(use_layer_wise_distributed_optimizer=True),
            model_chunks=[model],
            use_gloo_process_groups=True,
        )

        assert isinstance(
            optimizer, LayerWiseDistributedOptimizer
        ), "Should return LayerWiseDistributedOptimizer"

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        update_successful, grad_norm, num_zeros = optimizer.step()
        assert update_successful, "Optimizer step should be successful"

        # After step + allgather, all ranks must agree on every parameter.
        for name, p in model.named_parameters():
            p_max = p.data.clone()
            torch.distributed.all_reduce(p_max, op=torch.distributed.ReduceOp.MAX)
            p_min = p.data.clone()
            torch.distributed.all_reduce(p_min, op=torch.distributed.ReduceOp.MIN)
            assert torch.equal(p_max, p_min), f"Param {name} diverged across ranks"


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestAngularMuownOptimizerColumnParallel:
    """AngularMuown correctness for column-sharded (partition_dim=1) weights.

    A RowParallelLinear weight (linear_proj / linear_fc2 / MoE down-proj) is
    sharded along its columns, so every rank holds only a slice of each row.
    The AngularMuown per-row scalars (``g`` seed, ``grad_g``, ``u_step`` norm)
    are ``dim=1`` reductions and become partial per rank unless all-reduced
    across the TP group. These tests shard a weight along ``dim=1`` and assert
    the ``duplicated``/``distributed`` update reproduces the single-rank
    (global) reference, which only holds once those reductions are TP-aware.
    """

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up tensor parallelism across the whole world (up to size 2)."""
        world = int(os.getenv('WORLD_SIZE', '1'))
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(world, 2))
        yield
        Utils.destroy_model_parallel()

    def _make_optimizer(self, params, mode, pg_collection):
        # fp32 matmuls (not "medium"/bf16) so the reference vs sharded
        # comparison is not dominated by bf16 rounding in Newton-Schulz.
        return TensorParallelAngularMuown(
            params=params,
            lr=0.01,
            momentum=0.95,
            nesterov=True,
            betas=(0.9, 0.95),
            adam_eps=1e-8,
            num_ns_steps=5,
            coefficient_type="simple",
            scale_mode="shape_scaling",
            fp32_matmul_prec="highest",
            pg_collection=pg_collection,
            tp_mode=mode,
        )

    @pytest.mark.parametrize("mode", ["duplicated", "distributed"])
    def test_column_sharded_matches_single_rank(self, mode):
        """A column-sharded (partition_dim=1) update must match the single-rank reference."""
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        tp_group = pg_collection.tp
        tp_size = torch.distributed.get_world_size(group=tp_group)
        tp_rank = torch.distributed.get_rank(group=tp_group)

        rows, cols_local = 48, 40
        cols_full = cols_local * tp_size

        # Identical full weight + grad on every rank (same seed, no rank offset).
        torch.manual_seed(1234)
        full_w = torch.randn(rows, cols_full, dtype=torch.float32, device='cuda') * 0.02
        full_grad = torch.randn(rows, cols_full, dtype=torch.float32, device='cuda')

        # Single-rank (global) reference: no TP, full matrix.
        ref_w = torch.nn.Parameter(full_w.clone())
        ref_opt = self._make_optimizer([ref_w], "duplicated", None)
        ref_w.grad = full_grad.clone()
        for _ in range(3):
            ref_opt.step()

        # Column-sharded run: this rank owns the group-rank-th column slice.
        # newton_schulz_tp all-gathers/cats shards in group-rank order, so the
        # slice assignment must follow the same order for the round-trip to line up.
        col = slice(tp_rank * cols_local, (tp_rank + 1) * cols_local)
        shard_w = torch.nn.Parameter(full_w[:, col].contiguous())
        shard_w.partition_dim = 1
        shard_opt = self._make_optimizer([shard_w], mode, pg_collection)
        shard_grad = full_grad[:, col].contiguous()
        for _ in range(3):
            shard_w.grad = shard_grad.clone()
            shard_opt.step()

        # Gather the column shards back to a full matrix and compare to reference.
        gathered = [torch.empty_like(shard_w.data) for _ in range(tp_size)]
        torch.distributed.all_gather(gathered, shard_w.data.contiguous(), group=tp_group)
        full_result = torch.cat(gathered, dim=1)

        torch.testing.assert_close(
            full_result,
            ref_w.data,
            rtol=1e-3,
            atol=1e-4,
            msg=lambda m: f"Column-sharded (mode={mode}) update diverged from single-rank reference\n\n{m}",
        )

        # The per-row states must be replicated across the column-shard ranks
        # (identical on every rank) and equal the single-rank reference; this is
        # what keeps the state_dict expand/collapse round-trip exact under TP.
        ref_state = ref_opt.state[ref_w]
        shard_state = shard_opt.state[shard_w]
        for key in ("g", "m_g", "v_g"):
            tensor = shard_state[key]
            t_max = tensor.clone()
            torch.distributed.all_reduce(t_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
            t_min = tensor.clone()
            torch.distributed.all_reduce(t_min, op=torch.distributed.ReduceOp.MIN, group=tp_group)
            assert torch.equal(t_max, t_min), f"State '{key}' not replicated across column-shard ranks"
            torch.testing.assert_close(
                tensor,
                ref_state[key],
                rtol=1e-3,
                atol=1e-4,
                msg=lambda m, key=key: f"State '{key}' diverged from single-rank reference\n\n{m}",
            )

    def test_blockwise_rejected(self):
        """blockwise is not supported: it would make the row geometry per-shard local.

        With Muon's ``tp_mode="blockwise"`` each column shard would run an
        independent local update (per-shard ``g`` from the local column norm),
        which is a different optimizer than AngularMuown, so the constructor
        must reject it.
        """
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        w = torch.nn.Parameter(torch.randn(48, 40, dtype=torch.float32, device='cuda'))
        with pytest.raises(ValueError, match="blockwise"):
            self._make_optimizer([w], "blockwise", pg_collection)
