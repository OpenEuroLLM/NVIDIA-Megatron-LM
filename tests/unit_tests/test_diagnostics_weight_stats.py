# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Weight stats and norm gains of `TrainingDiagnostics` over Megatron-FSDP-style shards.

Two gloo ranks on CPU hold DTensors sharded on dim 0 (uneven, and one shard empty), a replicated
plain parameter and a sharded gain; the per-slot sums / counts / min / max / row norms must equal
the same statistics of the FULL tensors. Before the fix the stats were taken per rank and reduced
over the model-parallel group, i.e. they described one shard.

    python -m pytest tests/unit_tests/test_diagnostics_weight_stats.py   (or run the file directly)
"""

import os
import socket
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _full_tensors():
    g = torch.Generator().manual_seed(1234)
    return {
        "a": torch.randn(6, 4, generator=g) * 3.0,  # Shard(0): 3 + 3 rows
        "b": torch.randn(5, 3, generator=g) + 2.0,  # Shard(0): 3 + 2 rows (uneven)
        "e": torch.randn(1, 4, generator=g) - 5.0,  # Shard(0): 1 + 0 rows (empty shard on rank 1)
        "d": torch.randn(3, 3, generator=g),  # replicated plain parameter (complete on every rank)
        "gain": torch.rand(7, generator=g) + 0.5,  # sharded 1-D norm gain
    }


def _expected(tensors_per_slot):
    out = []
    for ts in tensors_per_slot:
        flat = torch.cat([t.reshape(-1) for t in ts]).double()
        rows = torch.cat([t.double().pow(2).sum(1).sqrt() for t in ts if t.dim() == 2])
        out.append((float(flat.sum()), float(flat.pow(2).sum()), float(flat.numel()),
                    float(flat.min()), float(flat.max()), float(rows.sum()), float(rows.numel()),
                    float(rows.max())))
    return out


def _worker(rank: int, world: int, port: int, failures):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Shard, distribute_tensor

        from megatron.training import diagnostics as D

        # mpu: TP = PP = 1 -> the model-parallel and pipeline groups are the rank itself
        own = [dist.new_group([r]) for r in range(world)][rank]
        D.mpu.get_model_parallel_group = lambda: own
        D.mpu.get_pipeline_model_parallel_group = lambda: own
        D.mpu.get_tensor_model_parallel_world_size = lambda: 1

        mesh = init_device_mesh("cpu", (world,))
        full = _full_tensors()

        def fsdp(t):
            p = torch.nn.Parameter(distribute_tensor(t.clone(), mesh, [Shard(0)]))
            p.__fsdp_param__ = True
            return p

        a, b, e, gain = fsdp(full["a"]), fsdp(full["b"]), fsdp(full["e"]), fsdp(full["gain"])
        d = torch.nn.Parameter(full["d"].clone())
        if rank == 1:
            assert e._local_tensor.numel() == 0  # the empty-shard case is exercised

        args = SimpleNamespace(diagnostics_interval=1, num_layers=1, diag_weight_stats=True,
                               diag_norm_gains=True, num_distributed_optimizer_instances=1)
        diag = D.TrainingDiagnostics(args)
        n_slots = 3
        diag._w_sum = torch.zeros(n_slots, D.N_WEIGHT_SUMS)
        diag._w_min = torch.full((n_slots,), float("inf"))
        diag._w_max = torch.full((n_slots,), float("-inf"))
        diag._w_rowmax = torch.zeros(n_slots)
        diag._stat_replica_factor = 1.0
        # slot 0: a + b, slot 1: e + everything in slot 0 (folded, as a family into "all"),
        # slot 2: the replicated d
        diag._weight_params = [(a, [0, 1], True), (b, [0, 1], True), (e, [1], True), (d, [2], False)]
        diag._collect_weights()

        diag._gain_buf = torch.zeros(1, D.N_GAIN_SUMS)
        diag._gain_min = torch.full((1,), float("inf"))
        diag._gain_max = torch.full((1,), float("-inf"))
        diag._gain_params = [(gain, 0, True)]
        diag._collect_gains()

        exp = _expected([[full["a"], full["b"]], [full["a"], full["b"], full["e"]], [full["d"]]])
        for slot, (s, sq, n, lo, hi, rsum, rn, rmax) in enumerate(exp):
            if slot == 1:  # a folded slot: rows stay with the family (only e's own row counts)
                er = full["e"].double().pow(2).sum(1).sqrt()
                rsum, rn, rmax = float(er.sum()), float(er.numel()), float(er.max())
            got = diag._w_sum[slot].double()
            checks = [
                ("sum", float(got[0]), s), ("sumsq", float(got[1]), sq), ("count", float(got[2]), n),
                ("min", float(diag._w_min[slot]), lo), ("max", float(diag._w_max[slot]), hi),
                ("rownorm_sum", float(got[3]), rsum), ("rows", float(got[4]), rn),
                ("rownorm_max", float(diag._w_rowmax[slot]), rmax),
            ]
            for what, have, want in checks:
                if abs(have - want) > 1e-4 * max(1.0, abs(want)):
                    failures.append(f"rank {rank} weight slot {slot} {what}: {have} != {want}")
        g = full["gain"].double()
        gs = diag._gain_buf[0].double()
        for what, have, want in (("sum", gs[0], g.sum()), ("sumsq", gs[1], g.pow(2).sum()),
                                 ("count", gs[2], g.numel()), ("min", diag._gain_min[0], g.min()),
                                 ("max", diag._gain_max[0], g.max())):
            if abs(float(have) - float(want)) > 1e-5 * max(1.0, abs(float(want))):
                failures.append(f"rank {rank} gain {what}: {float(have)} != {float(want)}")
        # end to end through the real slot layout and emit(): production parameter names
        args2 = SimpleNamespace(diagnostics_interval=1, num_layers=2, diag_weight_stats=True,
                                diag_norm_gains=True, num_distributed_optimizer_instances=1)
        d2 = D.TrainingDiagnostics(args2)
        n_w = len(D.WEIGHT_FAMILIES) * 2 + len(D.WEIGHT_EXTRAS)
        n_g = len(D.GAIN_FAMILIES) * 2 + 1
        d2._w_sum = torch.zeros(n_w, D.N_WEIGHT_SUMS)
        d2._w_min = torch.full((n_w,), float("inf"))
        d2._w_max = torch.full((n_w,), float("-inf"))
        d2._w_rowmax = torch.zeros(n_w)
        d2._gain_buf = torch.zeros(n_g, D.N_GAIN_SUMS)
        d2._gain_min = torch.full((n_g,), float("inf"))
        d2._gain_max = torch.full((n_g,), float("-inf"))
        d2._stat_replica_factor = 1.0
        named = [
            ("module.decoder.layers.1.mlp.router.weight", a, 1),
            ("module.decoder.layers.1.mlp.experts.linear_fc1.weight7", b, 1),
            ("module.decoder.layers.0.self_attention.layer.q_proj.weight", d, 0),
            ("module.embedding.word_embeddings.weight", e, None),
        ]
        d2._weight_params = [
            (prm, d2._weight_slots(n, li), bool(getattr(prm, "__fsdp_param__", False))) for n, prm, li in named
        ]
        d2._gain_params = [(gain, d2._gain_slot("pre_mlp_norm", 1), True)]
        d2._collect_weights()
        d2._collect_gains()

        class Writer:
            def __init__(self):
                self.v = {}

            def add_scalar(self, key, value, iteration):
                self.v[key] = value

        w = Writer()
        d2.emit(11, w, None)
        r = full["a"].double()
        want_keys = {
            "diag/weight/router/layer_01/norm": float(r.pow(2).sum().sqrt()),
            "diag/weight/router/layer_01/rownorm_max": float(r.pow(2).sum(1).sqrt().max()),
            "diag/weight/experts_fc1/layer_01/mean": float(full["b"].double().mean()),
            "diag/weight/mixer_qkv/layer_00/max": float(full["d"].max()),
            "diag/weight/embedding/min": float(full["e"].min()),
            "diag/gain/pre_mlp_norm/layer_01/mean": float(full["gain"].double().mean()),
            "diag/gain/pre_mlp_norm/layer_01/std": float(full["gain"].double().std(unbiased=False)),
        }
        tot = sum(float(t.double().pow(2).sum()) for t in (full["a"], full["b"], full["d"], full["e"]))
        want_keys["diag/weight/total_norm"] = tot ** 0.5
        want_keys["diag/weight/all/layer_01/norm"] = (float(full["a"].double().pow(2).sum())
                                                      + float(full["b"].double().pow(2).sum())) ** 0.5
        for k, want in want_keys.items():
            if k not in w.v:
                failures.append(f"rank {rank} emit: missing {k}")
            elif abs(w.v[k] - want) > 1e-4 * max(1.0, abs(want)):
                failures.append(f"rank {rank} emit {k}: {w.v[k]} != {want}")
    except Exception as exc:  # pragma: no cover - reported by the parent
        failures.append(f"rank {rank}: {type(exc).__name__}: {exc}")
    finally:
        dist.destroy_process_group()


def test_weight_stats_and_gains_over_fsdp_shards():
    world = 2
    with mp.Manager() as manager:
        failures = manager.list()
        mp.spawn(_worker, args=(world, _free_port(), failures), nprocs=world, join=True)
        assert not list(failures), "\n".join(failures)


if __name__ == "__main__":
    test_weight_stats_and_gains_over_fsdp_shards()
    print("ok")
