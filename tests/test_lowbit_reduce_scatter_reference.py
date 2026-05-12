import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import bitscom
from bitscom.quantization import dequantize_tensor, pack_lowbit, quantize_tensor, unpack_lowbit


pytestmark = pytest.mark.integration


TEST_CASES = [
    {
        "name": "reduce_scatter_bw4",
        "bitwidth": 4,
        "numel": 512,
        "seed": 17,
    },
    {
        "name": "reduce_scatter_bw2",
        "bitwidth": 2,
        "numel": 1024,
        "seed": 29,
    },
]


def _make_inputs(rank: int, world_size: int, case: dict, device: torch.device) -> list[torch.Tensor]:
    numel = int(case["numel"])
    seed = int(case["seed"])

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + rank * 97)
    base = torch.linspace(-1.2, 1.4, steps=numel, dtype=torch.float32)

    inputs = []
    for shard_idx in range(world_size):
        noise = torch.randn(numel, generator=gen, dtype=torch.float32) * 0.02
        shard = base + noise + rank * 0.05 + shard_idx * 0.01
        inputs.append(shard.to(device))
    return inputs


def _simulate_lowbit_reduce_scatter_cpu(
    inputs_by_rank: list[list[torch.Tensor]],
    bitwidth: int,
) -> list[torch.Tensor]:
    world_size = len(inputs_by_rank)
    if world_size == 0:
        return []

    shard_count = len(inputs_by_rank[0])
    if shard_count == 0:
        return [torch.empty(0) for _ in range(world_size)]

    numel = int(inputs_by_rank[0][0].numel())
    reduced_by_shard: list[torch.Tensor] = []

    for shard_idx in range(shard_count):
        local_sum = torch.zeros(numel, dtype=torch.float32)
        for src_rank in range(world_size):
            shard = inputs_by_rank[src_rank][shard_idx]
            q, scale = quantize_tensor(shard, bitwidth=bitwidth, stochastic_rounding=False)
            packed, _ = pack_lowbit(q, bitwidth)
            q_unpacked = unpack_lowbit(packed, bitwidth, numel)
            fp_part = dequantize_tensor(
                q_unpacked,
                scale,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            local_sum.add_(fp_part)
        reduced_by_shard.append(local_sum)

    outputs = [reduced_by_shard[rank_idx] for rank_idx in range(world_size)]
    return outputs


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
        inputs = _make_inputs(rank, world_size, case, device)

        output = torch.empty_like(inputs[0])
        dist.reduce_scatter(output, inputs, op=dist.ReduceOp.SUM)

        gathered_by_shard = []
        for shard_idx in range(world_size):
            gathered = [torch.empty_like(inputs[shard_idx]) for _ in range(world_size)]
            dist.all_gather(gathered, inputs[shard_idx])
            gathered_by_shard.append([t.cpu() for t in gathered])

        inputs_by_rank = [
            [gathered_by_shard[shard_idx][r] for shard_idx in range(world_size)]
            for r in range(world_size)
        ]

        expected_all = _simulate_lowbit_reduce_scatter_cpu(inputs_by_rank, bitwidth=bitwidth)
        expected = expected_all[rank].to(output.device)

        max_abs_err = (output - expected).abs().max().to(torch.float32)
        mean_abs_err = (output - expected).abs().mean().to(torch.float32)
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


def _run_case(case: dict) -> None:
    world_size = int(os.getenv("BITSCOM_DIST_WORLD_SIZE", "2"))
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()

    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = str(Path(tmpdir) / "init")
        mp.spawn(
            _worker,
            args=(world_size, init_file, case, queue),
            nprocs=world_size,
            join=True,
        )

    ok, payload = queue.get_nowait()
    assert ok, payload

    max_err, mean_err = payload["errs"]
    assert max_err < 0.6
    assert mean_err < 0.12


@pytest.mark.parametrize("case", TEST_CASES)
def test_lowbit_reduce_scatter_reference(case):
    _run_case(case)
