"""
bitscom/lowbit reduce_scatter 跨机测试（双机双卡，torchrun 启动）。

与 tests/test_nccl_reduce_scatter.py 一一对应：通信次数（iters）与通信量
（每个 rank 输入 world_size 个 n 长 chunk，输出一个 n 长分片）完全一致，
唯一区别是走 bitscom 的 lowbit backend，用于对比 bitscom 相对纯 NCCL 的
开销，并验证 reduce_scatter 是否死锁。

启动方式（先 master 后 worker，两节点跑同一个脚本）：
    # master 节点（node_rank=0）
    bash tests/run_bitscom.sh 0
    # worker 节点（node_rank=1）
    bash tests/run_bitscom.sh 1

排障环境变量：
    NCCL_SOCKET_IFNAME=eth1  直连网卡名（非默认路由网卡时必填，见 run_bitscom.sh）
    NCCL_DEBUG=INFO          打开 NCCL 详细日志，卡死时定位用
"""

import os
import socket
import time

import torch
import torch.distributed as dist

import bitscom
from bitscom.quantization import DEFAULT_BLOCK_SIZE


# 直连网卡名，务必按环境设置（跨机握手/collective 卡死时最可能的原因）
NCCL_SOCKET_IFNAME = os.environ.get("NCCL_SOCKET_IFNAME", "ens1f0")
if NCCL_SOCKET_IFNAME:
    os.environ["NCCL_SOCKET_IFNAME"] = NCCL_SOCKET_IFNAME


def log(*args):
    # 全部 flush：进程卡死时缓冲的 stdout 会丢，flush 保证最后一条日志可见。
    print(*args, flush=True)


def _local_ip() -> str:
    """本机到 master 方向的实际出口 IP，用于核对两节点是否在同一网段。"""
    master = os.environ.get("MASTER_ADDR", "10.31.10.62")
    port = int(os.environ.get("MASTER_PORT", "29500"))
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        s.connect((master, port))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


def dump_env(rank: int, world_size: int):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    host = socket.gethostname()
    ip = _local_ip()
    dev = torch.cuda.get_device_name(local_rank if local_rank >= 0 else 0)
    nccl_ver = torch.cuda.nccl.version()
    log(
        f"[init] rank={rank}/{world_size} local_rank={local_rank} "
        f"host={host} ip={ip}\n"
        f"        gpu={dev} torch={torch.__version__} cuda={torch.version.cuda} "
        f"nccl={nccl_ver} socket_ifname={NCCL_SOCKET_IFNAME or '(auto)'}"
    )


def check(rank: int, actual: torch.Tensor, expected: torch.Tensor, label: str):
    """bitscom 是近似通信，只硬性校验有限性（无 NaN/Inf）与形状，误差仅供参考。"""
    finite = torch.isfinite(actual).all().item()
    err = (actual - expected).abs().max().item()
    log(f"[{rank}] [CHECK] {label}: finite={finite} max_abs_err={err:.6f} "
        f"-> {'OK' if finite else 'FAIL'}")
    return finite


def bench_reduce_scatter(rank: int, world_size: int, n: int, iters: int):
    """list 形式 reduce_scatter（bitscom lowbit backend）。

    与 test_nccl_reduce_scatter.py 的 bench_reduce_scatter_list 一致：
    input_list = world_size 个 n 长 chunk（每个 rank 通信量 world_size*n），
    output = 一个 n 长分片。每个 chunk 填本 rank 值，期望全精度求和为
    sum(range(world_size))，bitscom 因量化/稀疏有误差。
    """
    expected_val = float(world_size * (world_size - 1) // 2)

    input_list = [torch.full((n,), float(rank), device="cuda") for _ in range(world_size)]
    output = torch.zeros(n, device="cuda")

    log(f"[{rank}] [RS_BITSCOM] warm-up barrier begin")
    dist.barrier()
    log(f"[{rank}] [RS_BITSCOM] warm-up barrier done, {iters} iters begin")

    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(iters):
        log(f"[{rank}] [RS_BITSCOM] iter {i} call")
        dist.reduce_scatter(output=output, input_list=input_list)
        log(f"[{rank}] [RS_BITSCOM] iter {i} returned")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    expected = torch.full((n,), expected_val, device="cuda")
    ok = check(rank, output, expected, "reduce_scatter(bitscom)")
    log(f"[{rank}] [RS_BITSCOM] {iters} iters: total {elapsed:.4f}s "
        f"avg {elapsed / iters * 1000:.2f}ms/iter")
    return ok


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    log(f"[{rank}] step0: cuda device set to {local_rank} "
        f"({torch.cuda.get_device_name(local_rank)})")

    # 与 test_perf_cross_node.py 保持同一份 bitscom 配置。
    bitscom.init(
        bitwidth=4,
        error_feedback=False,
        error_feedback_mode="none",
        block_size=DEFAULT_BLOCK_SIZE,
        sparse_enabled=True,
        sparse_projection_rank=4,
        sparse_compression_ratio=0.1,
        sparse_row_width=128,
        sparse_priority_mode=0,
        sparse_priority_quantize_bitwidth=4,
        sparse_non_priority_mode=2,
        sparse_non_priority_quantize_bitwidth=4,
    )
    log(f"[{rank}] step1: bitscom.init done (lowbit backend registered)")

    dist.init_process_group(backend="lowbit")
    log(f"[{rank}] step2: init_process_group(lowbit) done")

    dump_env(rank, world_size)

    # 与 test_nccl_reduce_scatter.py 保持一致：n=1024*1024，iters=4。
    n = 1024 * 1024
    iters = 4

    dist.barrier()
    if rank == 0:
        log(f"[{rank}] step3: all {world_size} ranks ready, start")

    ok = bench_reduce_scatter(rank, world_size, n, iters)

    dist.barrier()
    if rank == 0:
        log("=" * 60)
        log(f"RESULT reduce_scatter(bitscom) : {'PASS' if ok else 'FAIL'}")
        log("=" * 60)

    dist.destroy_process_group()
    log(f"[{rank}] done")


if __name__ == "__main__":
    main()
