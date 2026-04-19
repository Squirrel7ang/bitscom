import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import bitscom
from bitscom.api import LowBitGroup
from bitscom.quantization import dequantize_tensor, pack_lowbit, quantize_tensor, unpack_lowbit


pytestmark = pytest.mark.integration


TEST_CASES = [
    {
        "name": "linspace_nondiv_bw4",
        "bitwidth": 4,
        "numel": 1031,
        "pattern": "linspace",
        "seed": 11,
    },
    {
        "name": "random_nondiv_bw2",
        "bitwidth": 2,
        "numel": 4099,
        "pattern": "random",
        "seed": 23,
    },
    {
        "name": "alternating_div_bw1",
        "bitwidth": 1,
        "numel": 2048,
        "pattern": "alternating",
        "seed": 37,
    },
    {
        "name": "small_range_div_bw4",
        "bitwidth": 4,
        "numel": 512,
        "pattern": "small_range",
        "seed": 41,
    },
    {
        "name": "sparse_spikes_nondiv_bw2",
        "bitwidth": 2,
        "numel": 777,
        "pattern": "sparse_spikes",
        "seed": 53,
    },
]


STOCHASTIC_CASES = [
    {
        "name": "stochastic_random_nondiv_bw4",
        "bitwidth": 4,
        "numel": 4099,
        "pattern": "random",
        "seed": 101,
    },
    {
        "name": "stochastic_small_range_bw2",
        "bitwidth": 2,
        "numel": 2051,
        "pattern": "small_range",
        "seed": 131,
    },
]


def _simulate_lowbit_allreduce_cpu(inputs: list[torch.Tensor], bitwidth: int) -> list[torch.Tensor]:
    world_size = len(inputs)
    if world_size == 0:
        return []

    flats = [t.contiguous().view(-1).to(torch.float32) for t in inputs]
    original_numel = int(flats[0].numel())
    if original_numel == 0:
        return [flat.clone() for flat in flats]

    pad = (world_size - (original_numel % world_size)) % world_size
    if pad:
        flats = [
            torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)], dim=0) for flat in flats
        ]

    shard_len = flats[0].numel() // world_size
    packed_by_src: list[list[torch.Tensor]] = []
    scale_by_src: list[list[float]] = []

    for flat in flats:
        q, scale = quantize_tensor(flat, bitwidth, stochastic_rounding=False)
        q_shards = list(q.split(shard_len))
        packed_shards = [pack_lowbit(shard, bitwidth)[0] for shard in q_shards]
        packed_by_src.append(packed_shards)
        scale_by_src.append([float(scale) for _ in range(world_size)])

    reduced_packed: list[torch.Tensor] = []
    reduced_scales: list[float] = []

    for dst_rank in range(world_size):
        local_sum = torch.zeros(shard_len, dtype=torch.float32)
        for src_rank in range(world_size):
            q_part = unpack_lowbit(
                packed_by_src[src_rank][dst_rank],
                bitwidth,
                shard_len,
            )
            fp_part = dequantize_tensor(
                q_part,
                scale_by_src[src_rank][dst_rank],
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            local_sum.add_(fp_part)

        q_reduced, reduced_scale = quantize_tensor(
            local_sum,
            bitwidth,
            stochastic_rounding=False,
        )
        packed_reduced, _ = pack_lowbit(q_reduced, bitwidth)
        reduced_packed.append(packed_reduced)
        reduced_scales.append(float(reduced_scale))

    out_shards = []
    for shard_rank in range(world_size):
        q_shard = unpack_lowbit(reduced_packed[shard_rank], bitwidth, shard_len)
        fp_shard = dequantize_tensor(
            q_shard,
            reduced_scales[shard_rank],
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        out_shards.append(fp_shard)

    restored = torch.cat(out_shards, dim=0)[:original_numel]
    return [restored.clone() for _ in range(world_size)]


def _make_case_input(rank: int, case: dict, device: torch.device) -> torch.Tensor:
    numel = int(case["numel"])
    pattern = str(case["pattern"])
    seed = int(case["seed"])

    if pattern == "linspace":
        base = torch.linspace(-2.5, 3.0, steps=numel, dtype=torch.float32)
        x = base + rank * 0.037
    elif pattern == "random":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + rank)
        x = torch.randn(numel, generator=gen, dtype=torch.float32) * 1.5 + rank * 0.031
    elif pattern == "alternating":
        idx = torch.arange(numel, dtype=torch.float32)
        signs = torch.where((idx.to(torch.int64) % 2) == 0, 1.0, -1.0)
        x = signs * (1.0 + 0.1 * rank)
    elif pattern == "small_range":
        idx = torch.arange(numel, dtype=torch.float32)
        x = 0.02 * torch.sin(idx / 7.0) + 0.015 * torch.cos(idx / 13.0) + rank * 0.002
    elif pattern == "sparse_spikes":
        x = torch.zeros(numel, dtype=torch.float32)
        spike_positions = torch.arange(rank, numel, 17, dtype=torch.int64)
        spike_values = torch.linspace(-3.0, 3.0, steps=spike_positions.numel(), dtype=torch.float32)
        x[spike_positions] = spike_values
    else:
        raise ValueError(f"unknown case pattern: {pattern}")

    return x.to(device)


def _worker(rank: int, world_size: int, init_file: str, case: dict, q):
    try:
        bitwidth = int(case["bitwidth"])
        bitscom.init(bitwidth=bitwidth)
        dist.init_process_group(
            backend="lowbit",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(rank)

        device = torch.device(f"cuda:{rank}")
        x = _make_case_input(rank, case, device)

        before = x.clone()
        gathered_before = [torch.empty_like(before) for _ in range(world_size)]
        dist.all_gather(gathered_before, before)

        group = LowBitGroup(bitwidth=bitwidth)
        actual = x.clone()
        group.all_reduce(actual)

        inputs_cpu = [t.cpu() for t in gathered_before]
        expected_all = _simulate_lowbit_allreduce_cpu(inputs_cpu, bitwidth=bitwidth)
        expected = expected_all[rank].to(actual.device)

        max_abs_err = (actual - expected).abs().max().to(torch.float32)
        mean_abs_err = (actual - expected).abs().mean().to(torch.float32)
        err = torch.stack([max_abs_err, mean_abs_err])
        dist.all_reduce(err, op=dist.ReduceOp.MAX)

        if rank == 0:
            q.put((True, {"case": case["name"], "errs": err.cpu().tolist()}))
    except Exception as exc:  # pragma: no cover - error path for spawned workers
        if rank == 0:
            q.put((False, f"case={case['name']}: {repr(exc)}"))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _worker_stochastic(rank: int, world_size: int, init_file: str, case: dict, q):
    try:
        bitwidth = int(case["bitwidth"])
        bitscom.init(bitwidth=bitwidth)
        dist.init_process_group(
            backend="lowbit",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(rank)

        device = torch.device(f"cuda:{rank}")
        x = _make_case_input(rank, case, device)

        gathered_before = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered_before, x)

        expected_all = _simulate_lowbit_allreduce_cpu(
            [t.cpu() for t in gathered_before],
            bitwidth=bitwidth,
        )
        expected = expected_all[rank].to(device)

        group = LowBitGroup(bitwidth=bitwidth, stochastic_rounding=True)

        torch.manual_seed(1000 + int(case["seed"]) + rank)
        torch.cuda.manual_seed(1000 + int(case["seed"]) + rank)
        out_a = x.clone()
        group.all_reduce(out_a)

        torch.manual_seed(2000 + int(case["seed"]) + rank)
        torch.cuda.manual_seed(2000 + int(case["seed"]) + rank)
        out_b = x.clone()
        group.all_reduce(out_b)

        stochastic_delta = (out_a - out_b).abs().max().to(torch.float32)
        err_a = (out_a - expected).abs().max().to(torch.float32)
        err_b = (out_b - expected).abs().max().to(torch.float32)

        gathered_a = [torch.empty_like(out_a) for _ in range(world_size)]
        gathered_b = [torch.empty_like(out_b) for _ in range(world_size)]
        dist.all_gather(gathered_a, out_a)
        dist.all_gather(gathered_b, out_b)
        cross_rank_diff_a = torch.stack(
            [(gathered_a[i] - gathered_a[0]).abs().max() for i in range(world_size)]
        ).max().to(torch.float32)
        cross_rank_diff_b = torch.stack(
            [(gathered_b[i] - gathered_b[0]).abs().max() for i in range(world_size)]
        ).max().to(torch.float32)

        finite_flag = torch.tensor(
            [
                float(torch.isfinite(out_a).all().item() and torch.isfinite(out_b).all().item())
            ],
            device=device,
            dtype=torch.float32,
        )

        stats = torch.stack([stochastic_delta, err_a, err_b, cross_rank_diff_a, cross_rank_diff_b])
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)

        if rank == 0:
            q.put(
                (
                    True,
                    {
                        "case": case["name"],
                        "stats": stats.cpu().tolist(),
                        "all_finite": float(finite_flag.item()),
                    },
                )
            )
    except Exception as exc:  # pragma: no cover - error path for spawned workers
        if rank == 0:
            q.put((False, f"case={case['name']}: {repr(exc)}"))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
def test_lowbit_allreduce_matches_cpu_reference_simulation(case):
    if os.getenv("BITSCOM_RUN_DIST", "0") != "1":
        pytest.skip("set BITSCOM_RUN_DIST=1 to run distributed lowbit allreduce reference test")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for lowbit allreduce reference test")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least 2 CUDA devices")
    if not dist.is_nccl_available():
        pytest.skip("NCCL backend is not available")

    world_size = 2
    mp_ctx = mp.get_context("spawn")
    q = mp_ctx.SimpleQueue()

    tmp = tempfile.NamedTemporaryFile(prefix="bitscom-lowbit-ref-", delete=False)
    tmp.close()
    init_file = str(Path(tmp.name).resolve())

    try:
        mp.spawn(_worker, args=(world_size, init_file, case, q), nprocs=world_size, join=True)
        ok, payload = q.get()
        assert ok, payload

        max_abs_err, mean_abs_err = payload["errs"]
        assert max_abs_err < 1e-4, f"{payload['case']} max_abs_err={max_abs_err}"
        assert mean_abs_err < 1e-5, f"{payload['case']} mean_abs_err={mean_abs_err}"
    finally:
        try:
            os.unlink(init_file)
        except OSError:
            pass


@pytest.mark.parametrize("case", STOCHASTIC_CASES, ids=[c["name"] for c in STOCHASTIC_CASES])
def test_lowbit_allreduce_stochastic_rounding(case):
    if os.getenv("BITSCOM_RUN_DIST", "0") != "1":
        pytest.skip("set BITSCOM_RUN_DIST=1 to run distributed lowbit allreduce reference test")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for lowbit allreduce reference test")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least 2 CUDA devices")
    if not dist.is_nccl_available():
        pytest.skip("NCCL backend is not available")

    world_size = 2
    mp_ctx = mp.get_context("spawn")
    q = mp_ctx.SimpleQueue()

    tmp = tempfile.NamedTemporaryFile(prefix="bitscom-lowbit-stochastic-ref-", delete=False)
    tmp.close()
    init_file = str(Path(tmp.name).resolve())

    try:
        mp.spawn(
            _worker_stochastic,
            args=(world_size, init_file, case, q),
            nprocs=world_size,
            join=True,
        )
        ok, payload = q.get()
        assert ok, payload

        stochastic_delta, err_a, err_b, cross_rank_diff_a, cross_rank_diff_b = payload["stats"]
        assert stochastic_delta > 0.0, f"{payload['case']} stochastic_delta={stochastic_delta}"
        assert err_a < 3.0, f"{payload['case']} err_a={err_a}"
        assert err_b < 3.0, f"{payload['case']} err_b={err_b}"
        assert cross_rank_diff_a < 1e-6, f"{payload['case']} cross_rank_diff_a={cross_rank_diff_a}"
        assert cross_rank_diff_b < 1e-6, f"{payload['case']} cross_rank_diff_b={cross_rank_diff_b}"
        assert payload["all_finite"] == 1.0, f"{payload['case']} has non-finite outputs"
    finally:
        try:
            os.unlink(init_file)
        except OSError:
            pass
