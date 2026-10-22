"""Production-shaped lowbit all-reduce test (POLAR + backend_allreduce=True).

experiments/qwen14b/run_qwen14b_polar_dp_pp_tp.py does:
    lowbit_dp_group = dist.new_group(ranks=dp_ranks, backend="lowbit")
    lowbit_group = LowBitGroup(bitwidth=2, process_group=lowbit_dp_group,
                               backend_allreduce=True)
and POLAR's hook then calls
    lowbit_group.all_reduce(flat_pred, op=SUM, async_op=True)
which api.py routes to dist.all_reduce(..., group=lowbit_pg, async_op=True)
-> ProcessGroupLowBit::allreduce (C++ pack -> NCCL -> unpack).

No other test in this directory exercises that combination: the reference
test uses backend_allreduce=False (Python alltoall path) with the lowbit
backend as WORLD.

Run:
    BITSCOM_RUN_DIST=1 python -m pytest tests/test_backend_allreduce_polar.py -v -s
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import bitscom
from bitscom.api import LowBitGroup

pytestmark = pytest.mark.integration


CASES = [
    {"name": "polar_bucket_bw2", "bitwidth": 2, "numel": 64_000_000},
    {"name": "polar_bucket_bw4", "bitwidth": 4, "numel": 64_000_000},
    {"name": "small_div_bw2", "bitwidth": 2, "numel": 1024},
]


def _worker(rank: int, world_size: int, init_file: str, case: dict, q):
    try:
        # Corex: ncclCommInitRank inside the lowbit backend binds the *current*
        # device, so this must happen before any lowbit group is created.
        torch.cuda.set_device(rank)
        bitscom.init(bitwidth=int(case["bitwidth"]))
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        dev = torch.device(f"cuda:{rank}")
        dp_ranks = list(range(world_size))
        lowbit_pg = dist.new_group(ranks=dp_ranks, backend="lowbit")

        numel = int(case["numel"])
        gen = torch.Generator(device="cpu")
        gen.manual_seed(7 + rank)
        x = (
            torch.randn(numel, generator=gen, dtype=torch.float32) * 0.01
            + 0.001 * rank
        ).to(dev)

        # Dense reference on the NCCL world group (OP SUM), same as the hook.
        ref = x.clone()
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)

        group = LowBitGroup(
            bitwidth=int(case["bitwidth"]),
            process_group=lowbit_pg,
            backend_allreduce=True,
        )
        out = x.clone()
        work = group.all_reduce(out, op=dist.ReduceOp.SUM, async_op=True)
        assert work is not None, "async_op=True must return a Work"
        work.wait()

        # Oracle: the Python alltoall path on the same lowbit group. It is the
        # path covered by test_lowbit_allreduce_reference.py (verified there
        # against a CPU reference simulation), so agreement means the C++
        # backend allreduce implements the same quantized reduction.
        py_group = LowBitGroup(
            bitwidth=int(case["bitwidth"]),
            process_group=lowbit_pg,
            backend_allreduce=False,
        )
        out_py = x.clone()
        py_group.all_reduce(out_py, op=dist.ReduceOp.SUM, async_op=True).wait()

        gathered = [torch.empty_like(out) for _ in range(world_size)]
        dist.all_gather(gathered, out)
        cross = (
            torch.stack([(g - gathered[0]).abs().max() for g in gathered])
            .max()
            .to(torch.float32)
        )

        finite = torch.isfinite(out).all().to(torch.float32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)

        denom = ref.abs().mean().clamp_min(1e-12)
        rel = ((out - ref).abs().mean() / denom).to(torch.float32)
        be_vs_py = (out - out_py).abs().max().to(torch.float32)
        stats = torch.stack([cross, rel, be_vs_py])
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)

        if rank == 0:
            q.put(
                (
                    True,
                    {
                        "case": case["name"],
                        "cross_rank_max_diff": float(stats[0]),
                        "mean_rel_err_vs_dense": float(stats[1]),
                        "max_backend_vs_python_path": float(stats[2]),
                        "all_finite": float(finite),
                    },
                )
            )
    except Exception as exc:  # pragma: no cover - error path for spawned workers
        if rank == 0:
            q.put((False, f"case={case.get('name')}: {exc!r}"))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_backend_allreduce_polar_path(case):
    if os.getenv("BITSCOM_RUN_DIST", "0") != "1":
        pytest.skip("set BITSCOM_RUN_DIST=1 to run distributed lowbit tests")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least 2 CUDA devices")

    world_size = 2
    mp_ctx = mp.get_context("spawn")
    q = mp_ctx.SimpleQueue()
    tmp = tempfile.NamedTemporaryFile(prefix="bitscom-backend-", delete=False)
    tmp.close()
    init_file = str(Path(tmp.name).resolve())
    try:
        mp.spawn(
            _worker,
            args=(world_size, init_file, case, q),
            nprocs=world_size,
            join=True,
        )
        ok, payload = q.get()
        assert ok, payload
        assert payload["all_finite"] == 1.0, payload
        assert payload["cross_rank_max_diff"] < 1e-6, payload
        # 2-bit block quantization of zero-mean Gaussian noise is far from the
        # dense result by design (~1.1x mean abs error), so compare against the
        # Python path / CPU-reference algorithm instead of the dense result.
        assert payload["max_backend_vs_python_path"] < 1e-2, payload
        print(f"[backend_allreduce] {payload}")
    finally:
        try:
            os.unlink(init_file)
        except OSError:
            pass
