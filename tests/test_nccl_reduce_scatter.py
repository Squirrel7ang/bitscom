"""
纯 NCCL reduce_scatter 跨机正确性测试（双机双卡，torchrun 启动）。

目的：隔离 bitscom/lowbit backend，直接验证标准 NCCL 的 reduce_scatter
（reduce_scatter_tensor + list 形式 reduce_scatter）在跨机环境下是否工作、
是否死锁。这是排查 test_perf_cross_node.py 里 standard 路径死锁的最小复现。

启动方式（先 master 后 worker，两节点跑同一个脚本）：
    # master 节点（node_rank=0）
    bash tests/run_nccl.sh 0
    # worker 节点（node_rank=1）
    bash tests/run_nccl.sh 1

排障环境变量：
    NCCL_SOCKET_IFNAME=eth1  直连网卡名（非默认路由网卡时必填，见 run_nccl.sh）
    NCCL_DEBUG=INFO          打开 NCCL 详细日志，卡死时定位用
"""

import os
import socket
import time

import torch
import torch.distributed as dist


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
    """校验张量是否与期望一致，用 max_abs_err 输出直观差异。"""
    err = (actual - expected).abs().max().item()
    ok = err < 1e-4
    log(f"[{rank}] [CHECK] {label}: max_abs_err={err:.6f} -> {'OK' if ok else 'FAIL'}")
    return ok


def bench_reduce_scatter_tensor(rank: int, world_size: int, n: int, iters: int):
    """单张量 reduce_scatter_tensor：input=[world_size*n]，output=[n]。

    每个 rank 的 input 全部填本 rank 值，则每个 rank 的 output 应为
    sum(range(world_size))（所有 rank 对该分片的贡献之和）。
    """
    expected_val = float(world_size * (world_size - 1) // 2)

    input_tensor = torch.full((world_size * n,), float(rank), device="cuda")
    output = torch.zeros(n, device="cuda")

    log(f"[{rank}] [RS_TENSOR] warm-up barrier begin")
    dist.barrier()
    log(f"[{rank}] [RS_TENSOR] warm-up barrier done, {iters} iters begin")

    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(iters):
        log(f"[{rank}] [RS_TENSOR] iter {i} call")
        dist.reduce_scatter_tensor(output, input_tensor)
        log(f"[{rank}] [RS_TENSOR] iter {i} returned")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    expected = torch.full((n,), expected_val, device="cuda")
    ok = check(rank, output, expected, "reduce_scatter_tensor")
    log(f"[{rank}] [RS_TENSOR] {iters} iters: total {elapsed:.4f}s "
        f"avg {elapsed / iters * 1000:.2f}ms/iter")
    return ok


def bench_reduce_scatter_list(rank: int, world_size: int, n: int, iters: int):
    """list 形式 reduce_scatter：input_list=[world_size 个 n 长 chunk]，output=[n]。

    每个 chunk 填本 rank 值，output 应为 sum(range(world_size))。
    """
    expected_val = float(world_size * (world_size - 1) // 2)

    input_list = [torch.full((n,), float(rank), device="cuda") for _ in range(world_size)]
    output = torch.zeros(n, device="cuda")

    log(f"[{rank}] [RS_LIST] warm-up barrier begin")
    dist.barrier()
    log(f"[{rank}] [RS_LIST] warm-up barrier done, {iters} iters begin")

    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(iters):
        log(f"[{rank}] [RS_LIST] iter {i} call")
        dist.reduce_scatter(output=output, input_list=input_list)
        log(f"[{rank}] [RS_LIST] iter {i} returned")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    expected = torch.full((n,), expected_val, device="cuda")
    ok = check(rank, output, expected, "reduce_scatter(list)")
    log(f"[{rank}] [RS_LIST] {iters} iters: total {elapsed:.4f}s "
        f"avg {elapsed / iters * 1000:.2f}ms/iter")
    return ok


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    log(f"[{rank}] step0: cuda device set to {local_rank} "
        f"({torch.cuda.get_device_name(local_rank)})")

    # 纯 NCCL，不引入 bitscom / lowbit backend。
    dist.init_process_group(backend="nccl")
    log(f"[{rank}] step1: init_process_group(nccl) done")

    dump_env(rank, world_size)

    n = 1024 * 1024        # 每个分片元素数
    iters = 4              # 迭代次数

    dist.barrier()
    if rank == 0:
        log(f"[{rank}] step2: all {world_size} ranks ready, start")

    # 先跑单张量版（test_perf_cross_node.py standard 死锁对应的那条路径）
    ok_tensor = bench_reduce_scatter_tensor(rank, world_size, n, iters)

    # 再跑 list 版（bitscom 路径用到的形式）
    ok_list = bench_reduce_scatter_list(rank, world_size, n, iters)

    dist.barrier()
    if rank == 0:
        log("=" * 60)
        log(f"RESULT reduce_scatter_tensor : {'PASS' if ok_tensor else 'FAIL'}")
        log(f"RESULT reduce_scatter(list)   : {'PASS' if ok_list else 'FAIL'}")
        log("=" * 60)

    dist.destroy_process_group()
    log(f"[{rank}] done")


if __name__ == "__main__":
    main()
