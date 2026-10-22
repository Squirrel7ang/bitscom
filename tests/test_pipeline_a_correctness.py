import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from bitscom.api import LowBitGroup


pytestmark = pytest.mark.integration


def _build_hierarchical_groups(rank: int):
    # Robust topology for cross-process correctness checks:
    # local_group has one rank (local phases become no-op but still exercised),
    # inter_group uses WORLD so all ranks participate in inter lowbit all-reduce.
    local_group = dist.new_group(ranks=[rank])
    inter_group = dist.group.WORLD
    return local_group, inter_group


def _pipeline_a_worker(rank: int, world_size: int, init_file: str, q):
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(rank)

        local_group, inter_group = _build_hierarchical_groups(rank)

        base = torch.linspace(-1.0, 1.0, steps=257, dtype=torch.float32, device=f"cuda:{rank}")
        x = base + rank * 0.01

        expected = x.clone()
        dist.all_reduce(expected, op=dist.ReduceOp.SUM)

        group = LowBitGroup(bitwidth=4, process_group=dist.group.WORLD)
        actual = x.clone()
        group.all_reduce(
            actual,
            op=dist.ReduceOp.SUM,
            local_group=local_group,
            inter_group=inter_group,
            chunk_size=63,
            async_op=False,
        )

        ref = [torch.empty_like(actual) for _ in range(world_size)]
        dist.all_gather(ref, actual)
        max_cross_rank_diff = torch.stack([(t - ref[0]).abs().max() for t in ref]).max()

        max_abs_err = (actual - expected).abs().max()
        max_val = expected.abs().max().clamp_min(1e-6)
        rel_err = max_abs_err / max_val

        # NCCL-backed process groups only accept CUDA tensors; the values were
        # computed from CUDA tensors above, so keep the reduce on the device.
        err_tensor = torch.tensor(
            [max_cross_rank_diff.item(), max_abs_err.item(), rel_err.item()],
            dtype=torch.float32,
            device=f"cuda:{rank}",
        )
        dist.all_reduce(err_tensor, op=dist.ReduceOp.MAX)

        if rank == 0:
            q.put((True, err_tensor.tolist()))
    except Exception as exc:  # pragma: no cover - error path for spawned workers
        if rank == 0:
            q.put((False, repr(exc)))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_pipeline_a_all_reduce_correctness():
    if os.getenv("BITSCOM_RUN_DIST", "0") != "1":
        pytest.skip("set BITSCOM_RUN_DIST=1 to run distributed pipeline correctness test")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for lowbit all_to_all correctness test")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least 2 CUDA devices")
    if not dist.is_nccl_available():
        pytest.skip("NCCL backend is not available")

    world_size = 2
    mp_ctx = mp.get_context("spawn")
    q = mp_ctx.SimpleQueue()

    tmp = tempfile.NamedTemporaryFile(prefix="bitscom-dist-", delete=False)
    tmp.close()
    init_file = str(Path(tmp.name).resolve())

    try:
        mp.spawn(
            _pipeline_a_worker,
            args=(world_size, init_file, q),
            nprocs=world_size,
            join=True,
        )
        ok, payload = q.get()
        assert ok, payload

        max_cross_rank_diff, max_abs_err, rel_err = payload
        assert max_cross_rank_diff < 1e-6
        assert max_abs_err < 1.5
        assert rel_err < 0.2
    finally:
        try:
            os.unlink(init_file)
        except OSError:
            pass
