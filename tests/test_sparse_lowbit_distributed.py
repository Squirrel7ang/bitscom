"""
测试稀疏化 ARC-Top-K 量化 allreduce — 真分布式 NCCL 多卡版本。

运行方式
--------

## 单机多卡（推荐单机 2+ 卡时使用）

# 方式 A: torchrun（推荐，与多机命令一致）
torchrun --nproc_per_node=2 tests/test_sparse_lowbit_distributed.py

# 方式 B: mp.spawn（pytest 驱动，需设置环境变量）
BITSCOM_RUN_DIST=1 pytest tests/test_sparse_lowbit_distributed.py -v -s

# 只跑某个具体测试
BITSCOM_RUN_DIST=1 pytest tests/test_sparse_lowbit_distributed.py::test_sparse_allreduce_matches_reference -v -s
BITSCOM_RUN_DIST=1 pytest tests/test_sparse_lowbit_distributed.py::test_sparse_allreduce_mode_combinations -v -s

## 双机多卡（每机 2 卡为例）

# 节点 0（master）:
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=2 \
    --master_addr=<MASTER_IP> --master_port=29500 \
    tests/test_sparse_lowbit_distributed.py

# 节点 1:
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=2 \
    --master_addr=<MASTER_IP> --master_port=29500 \
    tests/test_sparse_lowbit_distributed.py

## 双机每机 8 卡

# torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
#     --master_addr=<MASTER_IP> --master_port=29500 \
#     tests/test_sparse_lowbit_distributed.py
# torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 \
#     --master_addr=<MASTER_IP> --master_port=29500 \
#     tests/test_sparse_lowbit_distributed.py

## 批量跑所有 mode 组合（bitscom 每次只能注册一次配置，需要逐一跑）
# for i in $(seq 0 8); do
#   echo "=== Mode combo index $i ==="
#   torchrun --nproc_per_node=2 tests/test_sparse_lowbit_distributed.py \
#       SPARSE_TEST_MODE=mode_combos SPARSE_MODE_COMBO_INDEX=$i
# done

## 批量跑所有 reference 测试用例
# for i in $(seq 0 3); do
#   echo "=== Test case index $i ==="
#   torchrun --nproc_per_node=2 tests/test_sparse_lowbit_distributed.py \
#       SPARSE_TEST_MODE=reference SPARSE_TEST_CASE_INDEX=$i
# done

前提条件
--------
- bitscom C++ extension 已编译安装: pip install -e .
- 至少 2 个 CUDA 设备 / 2 台带 GPU 的机器
- NCCL 可用（nccl-tests 验证过连通性）
"""

import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import bitscom
from bitscom.quantization import (
    DEFAULT_BLOCK_SIZE,
    quantize_pack_tensor_blockwise,
    unpack_dequantize_tensor_blockwise,
)


# ============================================================
# 稀疏化 allreduce 参考模拟（CPU 纯 Python，用于对比）
# ============================================================

def _cal_max_factor(size: int) -> int:
    factor = 1
    while size % factor == 0 and size // factor > factor:
        factor *= 2
    return factor


def _quantized_allreduce_cpu_ref(
    inputs: list[torch.Tensor],
    bitwidth: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> list[torch.Tensor]:
    """CPU 上模拟量化 allreduce: pack→alltoall→unpack/sum→repack→allgather→unpack"""
    world_size = len(inputs)
    flats = [t.contiguous().view(-1).to(torch.float32) for t in inputs]
    numel = int(flats[0].numel())
    if numel == 0:
        return [flat.clone() for flat in flats]

    pad = (world_size - (numel % world_size)) % world_size
    if pad:
        flats = [torch.cat([f, torch.zeros(pad, dtype=f.dtype)], dim=0) for f in flats]

    shard_len = flats[0].numel() // world_size

    send_packed: list[list[torch.Tensor]] = []
    send_scales: list[list[torch.Tensor]] = []
    for flat in flats:
        shards = list(flat.split(shard_len))
        pk = []
        sc = []
        for shard in shards:
            p, s, _ = quantize_pack_tensor_blockwise(
                shard, bitwidth, block_size=block_size, stochastic_rounding=False
            )
            pk.append(p)
            sc.append(s)
        send_packed.append(pk)
        send_scales.append(sc)

    reduced_packed = []
    reduced_scales = []
    for dst in range(world_size):
        local_sum = torch.zeros(shard_len, dtype=torch.float32)
        for src in range(world_size):
            fp = unpack_dequantize_tensor_blockwise(
                send_packed[src][dst],
                send_scales[src][dst],
                bitwidth,
                shard_len,
                block_size=block_size,
                dtype=torch.float32,
            )
            local_sum.add_(fp)
        p, s, _ = quantize_pack_tensor_blockwise(
            local_sum, bitwidth, block_size=block_size, stochastic_rounding=False
        )
        reduced_packed.append(p)
        reduced_scales.append(s)

    out_shards = []
    for r in range(world_size):
        fp = unpack_dequantize_tensor_blockwise(
            reduced_packed[r],
            reduced_scales[r],
            bitwidth,
            shard_len,
            block_size=block_size,
            dtype=torch.float32,
        )
        out_shards.append(fp)

    restored = torch.cat(out_shards, dim=0)[:numel]
    return [restored.clone() for _ in range(world_size)]


def _sparse_allreduce_cpu_ref(
    inputs: list[torch.Tensor],
    *,
    compression_ratio: float,
    projection_rank: int,
    priority_mode: int,          # 0=kFull, 1=kQuantize, 2=kDiscard
    priority_bitwidth: int,
    non_priority_mode: int,
    non_priority_bitwidth: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> list[torch.Tensor]:
    """CPU 上模拟 ARC-Top-K 稀疏化量化 allreduce"""
    world_size = len(inputs)
    flats = [t.contiguous().view(-1).to(torch.float32) for t in inputs]
    d = int(flats[0].numel())
    assert all(f.numel() == d for f in flats)
    if d == 0:
        return [f.clone() for f in flats]

    n = _cal_max_factor(d)
    m = d // n
    Gs = [f.view(n, m) for f in flats]

    # 随机投影（各 rank 相同的 V，模拟 allreduce 后取平均的 priority）
    r = projection_rank
    V = torch.randn(m, r, dtype=torch.float32)
    Ps = [torch.matmul(G, V) / math.sqrt(float(r)) for G in Gs]
    P_global = sum(Ps) / float(world_size)

    # Score + Top-K
    score = (P_global * P_global).sum(dim=1)
    K = max(1, int(n * compression_ratio))
    _, indices = torch.topk(score, K)

    # 互补索引
    comp_mask = torch.ones(n, dtype=torch.bool)
    comp_mask[indices] = False
    comp_indices = torch.arange(n)[comp_mask]

    results = []
    for rank_idx in range(world_size):
        pri_rows = Gs[rank_idx][indices, :]
        nonpri_rows = Gs[rank_idx][comp_indices, :]

        # priority 行
        if priority_mode == 0:  # kFull
            reduced_pri = torch.stack([Gs[r][indices, :] for r in range(world_size)]).sum(dim=0)
        elif priority_mode == 1:  # kQuantize
            r_flat = _quantized_allreduce_cpu_ref(
                [pri_rows.contiguous().view(-1) for _ in range(world_size)],
                bitwidth=priority_bitwidth,
                block_size=block_size,
            )[0]
            reduced_pri = r_flat.view(K, m)
        else:  # kDiscard
            reduced_pri = torch.zeros_like(pri_rows)

        # non-priority 行
        nonK = n - K
        if nonK > 0:
            if non_priority_mode == 0:
                reduced_nonpri = torch.stack(
                    [Gs[r][comp_indices, :] for r in range(world_size)]
                ).sum(dim=0)
            elif non_priority_mode == 1:
                r_flat = _quantized_allreduce_cpu_ref(
                    [nonpri_rows.contiguous().view(-1) for _ in range(world_size)],
                    bitwidth=non_priority_bitwidth,
                    block_size=block_size,
                )[0]
                reduced_nonpri = r_flat.view(nonK, m)
            else:
                reduced_nonpri = torch.zeros_like(nonpri_rows)
        else:
            reduced_nonpri = torch.zeros(0, m)

        result = torch.zeros(n, m)
        result[indices] = reduced_pri
        if nonK > 0:
            result[comp_indices] = reduced_nonpri

        results.append(result.view(-1))

    return results


# ============================================================
# 测试用例定义
# ============================================================

TEST_CONFIGS = [
    {
        "name": "rand_128_ratio50_priFull_nonpriQ4",
        "numel": 128,
        "projection_rank": 4,
        "compression_ratio": 0.5,
        "priority_mode": 0,         # kFull
        "priority_bitwidth": 4,
        "non_priority_mode": 1,     # kQuantize
        "non_priority_bitwidth": 4,
        "seed": 42,
    },
    {
        "name": "rand_256_ratio25_priFull_nonpriQ4",
        "numel": 256,
        "projection_rank": 8,
        "compression_ratio": 0.25,
        "priority_mode": 0,
        "priority_bitwidth": 4,
        "non_priority_mode": 1,
        "non_priority_bitwidth": 4,
        "seed": 57,
    },
    {
        "name": "rand_512_ratio50_priQ4_nonpriQ4",
        "numel": 512,
        "projection_rank": 8,
        "compression_ratio": 0.5,
        "priority_mode": 1,         # kQuantize
        "priority_bitwidth": 4,
        "non_priority_mode": 1,     # kQuantize
        "non_priority_bitwidth": 4,
        "seed": 99,
    },
    {
        "name": "rand_128_ratio50_priFull_nonpriDiscard",
        "numel": 128,
        "projection_rank": 4,
        "compression_ratio": 0.5,
        "priority_mode": 0,         # kFull
        "priority_bitwidth": 4,
        "non_priority_mode": 2,     # kDiscard
        "non_priority_bitwidth": 4,
        "seed": 123,
    },
]

MODE_COMBOS = [
    {"pri": 0, "nonpri": 0},   # full / full
    {"pri": 0, "nonpri": 1},   # full / quantize
    {"pri": 0, "nonpri": 2},   # full / discard
    {"pri": 1, "nonpri": 0},   # quantize / full
    {"pri": 1, "nonpri": 1},   # quantize / quantize
    {"pri": 1, "nonpri": 2},   # quantize / discard
    {"pri": 2, "nonpri": 0},   # discard / full
    {"pri": 2, "nonpri": 1},   # discard / quantize
    {"pri": 2, "nonpri": 2},   # discard / discard
]


# ============================================================
# Worker 函数（torchrun / mp.spawn 调用的单进程入口）
# ============================================================

def _sparse_worker(
    rank: int,
    world_size: int,
    init_method: str,
    config: dict,
    result_queue: Optional[mp.SimpleQueue],
):
    """
    分布式 worker：注册 lowbit 后端（稀疏化开启），执行 all_reduce，
    与 Python 参考模拟结果对比。
    """
    try:
        # ---- 初始化 ----
        bitscom.init(
            bitwidth=4,
            sparse_enabled=True,
            sparse_projection_rank=int(config["projection_rank"]),
            sparse_compression_ratio=float(config["compression_ratio"]),
            sparse_priority_mode=int(config["priority_mode"]),
            sparse_priority_quantize_bitwidth=int(config["priority_bitwidth"]),
            sparse_non_priority_mode=int(config["non_priority_mode"]),
            sparse_non_priority_quantize_bitwidth=int(config["non_priority_bitwidth"]),
        )
        dist.init_process_group(
            backend="lowbit",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

        # ---- 生成测试数据 ----
        numel = int(config["numel"])
        seed = int(config["seed"])
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + rank)
        x = torch.randn(numel, generator=gen, dtype=torch.float32, device=device)
        # 所有 rank 的不同随机种子给出不同本地数据

        # ---- 收集各 rank 原始数据用于参考计算 ----
        gathered_before = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered_before, x)

        inputs_cpu = [t.cpu() for t in gathered_before]
        expected_all = _sparse_allreduce_cpu_ref(
            inputs_cpu,
            compression_ratio=float(config["compression_ratio"]),
            projection_rank=int(config["projection_rank"]),
            priority_mode=int(config["priority_mode"]),
            priority_bitwidth=int(config["priority_bitwidth"]),
            non_priority_mode=int(config["non_priority_mode"]),
            non_priority_bitwidth=int(config["non_priority_bitwidth"]),
        )
        expected = expected_all[rank].to(device).view_as(x)

        # ---- 执行 C++ 稀疏化 allreduce ----
        actual = x.clone()
        work = dist.all_reduce(actual, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()

        # ---- 对比 ----
        abs_err = (actual - expected).abs()
        max_abs_err = abs_err.max().to(torch.float32)
        mean_abs_err = abs_err.mean().to(torch.float32)

        # 跨 rank 收集最大误差（确保所有 rank 的结果一致）
        err_stats = torch.stack([max_abs_err, mean_abs_err])
        dist.all_reduce(err_stats, op=dist.ReduceOp.MAX)

        is_finite = torch.tensor(
            [float(torch.isfinite(actual).all().item())],
            device=device,
            dtype=torch.float32,
        )
        dist.all_reduce(is_finite, op=dist.ReduceOp.MIN)

        if rank == 0:
            result_queue.put({
                "case": config["name"],
                "max_abs_err": float(err_stats[0].item()),
                "mean_abs_err": float(err_stats[1].item()),
                "all_finite": bool(is_finite.item()),
            })

    except Exception as exc:
        if rank == 0:
            result_queue.put({"error": f"{config['name']}: {repr(exc)}"})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _mode_combo_worker(
    rank: int,
    world_size: int,
    init_method: str,
    combo: dict,
    result_queue: Optional[mp.SimpleQueue],
):
    """Worker: 遍历所有 mode 组合，确保不崩溃且输出有限。"""
    try:
        bitscom.init(
            bitwidth=4,
            sparse_enabled=True,
            sparse_projection_rank=4,
            sparse_compression_ratio=0.5,
            sparse_priority_mode=int(combo["pri"]),
            sparse_priority_quantize_bitwidth=4,
            sparse_non_priority_mode=int(combo["nonpri"]),
            sparse_non_priority_quantize_bitwidth=4,
        )
        dist.init_process_group(
            backend="lowbit",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

        gen = torch.Generator(device="cpu")
        gen.manual_seed(42 + rank)
        x = torch.randn(128, generator=gen, dtype=torch.float32, device=device)

        work = dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()

        ok = bool(torch.isfinite(x).all().item())
        all_ok = torch.tensor([float(ok)], device=device, dtype=torch.float32)
        dist.all_reduce(all_ok, op=dist.ReduceOp.MIN)

        if rank == 0:
            result_queue.put({
                "combo": f"pri={combo['pri']},nonpri={combo['nonpri']}",
                "ok": bool(all_ok.item()),
            })

    except Exception as exc:
        if rank == 0:
            result_queue.put({"combo": f"pri={combo['pri']},nonpri={combo['nonpri']}", "error": repr(exc)})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ============================================================
# 测试入口（pytest 驱动 mp.spawn）
# ============================================================

def _run_dist_test(worker_fn, args, world_size: int, timeout_sec: int = 120):
    """用 mp.spawn 启动 world_size 个进程，运行 worker_fn(*args, q)。"""
    mp_ctx = mp.get_context("spawn")
    q = mp_ctx.SimpleQueue()

    tmp = tempfile.NamedTemporaryFile(prefix="bitscom-sparse-dist-", delete=False)
    tmp.close()
    init_file = str(Path(tmp.name).resolve())

    try:
        mp.spawn(
            worker_fn,
            args=(world_size, f"file://{init_file}", *args, q),
            nprocs=world_size,
            join=True,
        )
        results = []
        while not q.empty():
            results.append(q.get())
        return results
    finally:
        try:
            os.unlink(init_file)
        except OSError:
            pass


@pytest.mark.parametrize("config", TEST_CONFIGS, ids=[c["name"] for c in TEST_CONFIGS])
def test_sparse_allreduce_matches_reference(config):
    """C++ 稀疏化 allreduce 结果应与 Python 参考模拟一致。"""
    if os.getenv("BITSCOM_RUN_DIST", "0") != "1":
        pytest.skip("set BITSCOM_RUN_DIST=1 to run distributed sparse allreduce tests")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least 2 CUDA devices")
    if not dist.is_nccl_available():
        pytest.skip("NCCL backend is not available")

    world_size = min(2, torch.cuda.device_count())
    results = _run_dist_test(_sparse_worker, (config,), world_size)

    assert len(results) == 1, f"Expected 1 result, got {results}"
    r = results[0]
    assert "error" not in r, r.get("error", "")
    assert r["all_finite"], f"{r['case']}: non-finite values in output"
    # 全精度路径有浮点精度误差，量化路径有量化误差
    assert r["max_abs_err"] < 5.0, (
        f"{r['case']}: max_abs_err={r['max_abs_err']:.6f} too large"
    )
    print(f"\n  {r['case']}: max_abs_err={r['max_abs_err']:.6f}, "
          f"mean_abs_err={r['mean_abs_err']:.6f} ✓")


@pytest.mark.parametrize("combo", MODE_COMBOS, ids=[
    f"pri={c['pri']}_nonpri={c['nonpri']}" for c in MODE_COMBOS
])
def test_sparse_allreduce_mode_combinations(combo):
    """所有 3×3=9 种 mode 组合均不应崩溃且输出有限。"""
    if os.getenv("BITSCOM_RUN_DIST", "0") != "1":
        pytest.skip("set BITSCOM_RUN_DIST=1 to run distributed sparse allreduce tests")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least 2 CUDA devices")
    if not dist.is_nccl_available():
        pytest.skip("NCCL backend is not available")

    world_size = min(2, torch.cuda.device_count())
    results = _run_dist_test(_mode_combo_worker, (combo,), world_size)

    assert len(results) == 1
    r = results[0]
    assert "error" not in r, r.get("error", "")
    assert r["ok"], f"{r['combo']}: non-finite values"
    print(f"  {r['combo']}: ✓")


# ============================================================
# torchrun 入口
# ============================================================

if __name__ == "__main__":
    """
    torchrun 方式直接运行此文件时执行。

    环境变量:
        SPARSE_TEST_MODE: "reference" (默认) 或 "mode_combos"
        SPARSE_TEST_CASE_INDEX: 测试用例索引 (0, 1, 2, 3), 仅 reference 模式生效
        SPARSE_MODE_COMBO_INDEX: mode 组合索引 (0..8), 仅 mode_combos 模式生效
    """
    if "LOCAL_RANK" not in os.environ:
        print("This script should be run with torchrun. Examples:")
        print("  torchrun --nproc_per_node=2 tests/test_sparse_lowbit_distributed.py")
        print("  torchrun --nnodes=2 --node_rank=0 --nproc_per_node=2 \\")
        print("      --master_addr=<IP> --master_port=29500 \\")
        print("      tests/test_sparse_lowbit_distributed.py")
        sys.exit(0)

    # bitscom.init 必须在 dist.init_process_group 之前调用
    test_mode = os.environ.get("SPARSE_TEST_MODE", "reference")

    if test_mode == "mode_combos":
        combo_idx = int(os.environ.get("SPARSE_MODE_COMBO_INDEX", "0"))
        combo = MODE_COMBOS[combo_idx]
        bitscom.init(
            bitwidth=4,
            sparse_enabled=True,
            sparse_projection_rank=4,
            sparse_compression_ratio=0.5,
            sparse_priority_mode=int(combo["pri"]),
            sparse_priority_quantize_bitwidth=4,
            sparse_non_priority_mode=int(combo["nonpri"]),
            sparse_non_priority_quantize_bitwidth=4,
        )
    else:
        case_idx = int(os.environ.get("SPARSE_TEST_CASE_INDEX", "0"))
        config = TEST_CONFIGS[case_idx]
        bitscom.init(
            bitwidth=4,
            sparse_enabled=True,
            sparse_projection_rank=int(config["projection_rank"]),
            sparse_compression_ratio=float(config["compression_ratio"]),
            sparse_priority_mode=int(config["priority_mode"]),
            sparse_priority_quantize_bitwidth=int(config["priority_bitwidth"]),
            sparse_non_priority_mode=int(config["non_priority_mode"]),
            sparse_non_priority_quantize_bitwidth=int(config["non_priority_bitwidth"]),
        )

    # torchrun 自动设置环境变量
    dist.init_process_group(backend="lowbit")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    if test_mode == "mode_combos":
        combo_idx = int(os.environ.get("SPARSE_MODE_COMBO_INDEX", "0"))
        combo = MODE_COMBOS[combo_idx]

        gen = torch.Generator(device="cpu")
        gen.manual_seed(42 + rank)
        x = torch.randn(128, generator=gen, dtype=torch.float32, device=device)

        work = dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()

        ok = torch.isfinite(x).all()
        all_ok = torch.tensor([float(ok.item())], device=device, dtype=torch.float32)
        dist.all_reduce(all_ok, op=dist.ReduceOp.MIN)

        if rank == 0:
            label = f"pri={combo['pri']},nonpri={combo['nonpri']}"
            status = "✓" if bool(all_ok.item()) else "✗ FAIL"
            print(f"[{label}] {status}", flush=True)

    else:
        # reference 测试
        case_idx = int(os.environ.get("SPARSE_TEST_CASE_INDEX", "0"))
        config = TEST_CONFIGS[case_idx]

        numel = int(config["numel"])
        seed = int(config["seed"])
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + rank)
        x = torch.randn(numel, generator=gen, dtype=torch.float32, device=device)

        # 收集各 rank 原始数据用于参考计算
        gathered_before = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered_before, x)

        inputs_cpu = [t.cpu() for t in gathered_before]
        expected_all = _sparse_allreduce_cpu_ref(
            inputs_cpu,
            compression_ratio=float(config["compression_ratio"]),
            projection_rank=int(config["projection_rank"]),
            priority_mode=int(config["priority_mode"]),
            priority_bitwidth=int(config["priority_bitwidth"]),
            non_priority_mode=int(config["non_priority_mode"]),
            non_priority_bitwidth=int(config["non_priority_bitwidth"]),
        )
        expected = expected_all[rank].to(device).view_as(x)

        actual = x.clone()
        work = dist.all_reduce(actual, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()

        abs_err = (actual - expected).abs()
        max_err = abs_err.max().to(torch.float32)
        mean_err = abs_err.mean().to(torch.float32)

        err_stats = torch.stack([max_err, mean_err])
        dist.all_reduce(err_stats, op=dist.ReduceOp.MAX)

        if rank == 0:
            print(f"[{config['name']}] world_size={world_size}", flush=True)
            print(f"  max_abs_err = {err_stats[0].item():.6f}", flush=True)
            print(f"  mean_abs_err = {err_stats[1].item():.6f}", flush=True)
            ok = err_stats[0].item() < 5.0
            print(f"  {'✓ PASS' if ok else '✗ FAIL (error too large)'}", flush=True)

    dist.destroy_process_group()
