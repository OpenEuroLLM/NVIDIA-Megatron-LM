# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""fsdp_fp8_param_gather: FP8 E4M3 (1 x 128 fp32 scales) all-gather of bf16 FSDP weights.

    torchrun --nproc-per-node 4 -m pytest tests/unit_tests/distributed/mfsdp_v1/test_fp8_param_gather.py
"""
import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.fp8_param_gather import (
    BLOCK,
    HAVE_TRITON,
    dequantize_unpack,
    fp8_all_gather_into_tensor,
    packed_layout,
    quantize_dequantize_reference,
    quantize_pack,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="needs CUDA and Triton"
)


def _init_dist():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())


def _block_error(out, x):
    """max |out - x| / (the amax of the 128-element block of x), per rank-shard blocking."""
    n = x.numel()
    nb = (n + BLOCK - 1) // BLOCK
    pad = nb * BLOCK - n
    xb = torch.nn.functional.pad(x.float(), (0, pad)).view(nb, BLOCK)
    ob = torch.nn.functional.pad(out.float(), (0, pad)).view(nb, BLOCK)
    amax = xb.abs().amax(1, keepdim=True).clamp_min(1e-30)
    return ((ob - xb).abs() / amax).max().item()


@pytest.mark.parametrize("n", [1, 127, 128, 129, 1000, 4096 * 37 + 5])
def test_kernels_match_reference(n):
    """Packing on 3 'ranks' and the Triton dequantization equal the plain-torch round trip
    bit for bit; the error stays within E4M3's 2^-4 of the block maximum."""
    torch.manual_seed(n)
    world = 3
    shards = []
    for _ in range(world):
        x = torch.randn(n, device="cuda") * (10 ** torch.empty(n, device="cuda").uniform_(-3, 1))
        if n > 300:
            x[128:256] = 0  # an all-zero block
        shards.append(x.to(torch.bfloat16))
    packed = torch.cat([quantize_pack(s) for s in shards])
    assert packed.numel() == world * packed_layout(n)[2]
    out = torch.empty(world * n, dtype=torch.bfloat16, device="cuda")
    dequantize_unpack(packed, n, world, out)
    ref = torch.cat([quantize_dequantize_reference(s) for s in shards])
    assert torch.equal(out, ref)
    for r in range(world):
        assert _block_error(out[r * n : (r + 1) * n], shards[r]) <= 2.0**-4 + 1e-3


def test_fp8_all_gather_matches_reference():
    """fp8_all_gather_into_tensor == concatenation of every rank's reference round trip, the
    same on every rank."""
    _init_dist()
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    n = 3 * BLOCK + 17
    torch.manual_seed(1234 + rank)
    shard = (torch.randn(n, device="cuda") * (rank + 1)).to(torch.bfloat16)
    out = torch.empty(world * n, dtype=torch.bfloat16, device="cuda")
    fp8_all_gather_into_tensor(out, shard, group=None).wait()
    exact = torch.empty_like(out)
    torch.distributed.all_gather_into_tensor(exact, shard)
    ref = torch.cat([quantize_dequantize_reference(exact[r * n : (r + 1) * n]) for r in range(world)])
    assert torch.equal(out, ref)
    assert not torch.equal(out, exact)  # the FP8 path is really taken
    # identical on every rank
    sums = torch.zeros(world, dtype=torch.float64, device="cuda")
    sums[rank] = out.double().sum()
    torch.distributed.all_reduce(sums)
    assert torch.allclose(sums, sums[0].expand_as(sums), rtol=0, atol=0)


class _Uniform(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden, hidden, bias=False)
        self.linear2 = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


@pytest.mark.parametrize("double_buffer", [False, True])
def test_fsdp_training_step_close_to_bf16(double_buffer):
    """A Megatron-FSDP model with fsdp_fp8_param_gather="all": the weights used in forward are
    the FP8 round trip of the bf16 weights (not the bf16 weights themselves, within E4M3's
    error), identical on all ranks; one optimizer step stays close to the bf16 baseline."""
    from tests.unit_tests.test_utilities import Utils

    _init_dist()
    world = torch.distributed.get_world_size()
    if world < 2:
        pytest.skip("needs >= 2 ranks")
    Utils.initialize_model_parallel()
    try:
        _training_step_check(double_buffer, world)
    finally:
        Utils.destroy_model_parallel()


def _training_step_check(double_buffer, world):
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
    from megatron.core.hyper_comm_grid import HyperCommGrid
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer import TransformerConfig

    grid = HyperCommGrid([world], ["dp"])
    pg = ProcessGroupCollection()
    pg.dp = grid.create_pg("dp")
    hidden = 256

    def build(mode, base=None):
        torch.manual_seed(0)
        m = _Uniform(hidden).cuda().to(torch.bfloat16)
        if base is not None:
            for p, q in zip(m.parameters(), base.parameters()):
                p.data.copy_(q.data)
        cfg = DistributedDataParallelConfig(
            data_parallel_sharding_strategy="optim_grads_params",
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            bucket_size=10000,
            use_megatron_fsdp=True,
            fsdp_double_buffer=double_buffer,
            fsdp_fp8_param_gather=mode,
        )
        fsdp = FullyShardedDataParallel(
            config=TransformerConfig(num_attention_heads=1, num_layers=1, context_parallel_size=1),
            ddp_config=cfg,
            module=m,
            fsdp_unit_modules=[torch.nn.Linear],
        )
        opt = DistributedOptimizer(
            optimizer=None,
            config=OptimizerConfig(optimizer="adam", lr=1e-3, bf16=True),
            grad_scaler=None,
            init_state_fn=None,
            model_chunks=[fsdp],
            per_model_buffers={0: [fsdp.param_and_grad_buffer]},
            data_parallel_group=fsdp.megatron_fsdp_dist_index.get_dp_group(),
            data_parallel_group_gloo=None,
            data_parallel_group_idx=0,
            distributed_optimizer_instance_id=0 if mode == "none" else 1,
        )
        return m, fsdp, opt

    raw0 = _Uniform(hidden).cuda().to(torch.bfloat16)
    base_mod, base, opt_b = build("none", raw0)
    fp8_mod, fp8, opt_f = build("all", raw0)

    seen = {}

    def capture(tag):
        def hook(mod, args):
            seen.setdefault(tag, []).append(mod.weight.detach().clone())

        return hook

    for tag, model in (("bf16", base_mod), ("fp8", fp8_mod)):
        for i, lin in enumerate((model.linear1, model.linear2)):
            lin.register_forward_pre_hook(capture(f"{tag}{i}"))

    torch.manual_seed(42)
    x = torch.randn(4, hidden, device="cuda", dtype=torch.bfloat16)
    losses = {}
    for tag, model, opt in (("bf16", base, opt_b), ("fp8", fp8, opt_f)):
        opt.zero_grad()
        y = model(x.clone())
        loss = y.float().pow(2).mean()
        loss.backward()
        model.finish_grad_sync()
        opt.step()
        losses[tag] = loss.item()

    for i in range(2):
        wb, wf = seen[f"bf16{i}"][0].float(), seen[f"fp8{i}"][0].float()
        assert not torch.equal(wb, wf), "FP8 path not taken"
        assert (wf - wb).abs().max().item() <= 2.0**-4 * wb.abs().max().item() * 1.01
        s = torch.tensor([wf.double().sum()], device="cuda")
        g = torch.empty(world, dtype=torch.float64, device="cuda")
        torch.distributed.all_gather_into_tensor(g, s)
        assert torch.all(g == g[0]), "unsharded FP8 weights differ between ranks"
    assert abs(losses["fp8"] - losses["bf16"]) <= 0.05 * abs(losses["bf16"]) + 1e-6, losses
