"""跨机量化 reduce_scatter 基准：16MB payload，32 次迭代。

用 bitwidth=4（<8）+ sparse_enabled=False，走 bitscom 的整张量量化
reduce_scatter 路径（pack → alltoall → unpack → sum → repack → allgather），
从而能对比 pack/unpack 的 C++ 老路径 vs 融合 CUDA kernel 两种实现的性能。

启动（两节点，先 master 后 worker）：
    bash tests/run_bench_quant.sh 0
    bash tests/run_bench_quant.sh 1
"""

import os
import socket
import time

import torch
import torch.distributed as dist

import bitscom

NCCL_SOCKET_IFNAME = os.environ.get("NCCL_SOCKET_IFNAME", "ens1f0")
if NCCL_SOCKET_IFNAME:
    os.environ["NCCL_SOCKET_IFNAME"] = NCCL_SOCKET_IFNAME


def log(*args):
    print(*args, flush=True)


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    bitscom.init(
        bitwidth=4,
        error_feedback=False,
        error_feedback_mode="none",
        block_size=256,
        sparse_enabled=False,
    )
    dist.init_process_group(backend="lowbit")

    # 每个 rank 的 input_list 总计 16MB FP32 = 4M 元素，切成 world_size 个 chunk。
    total_elems = int(os.environ.get("BENCH_ELEMS", 4 * 1024 * 1024))
    iters = int(os.environ.get("BENCH_ITERS", 32))
    n = total_elems // world_size

    input_list = [torch.randn(n, device="cuda") for _ in range(world_size)]
    output = torch.zeros(n, device="cuda")

    dist.barrier()
    torch.cuda.synchronize()

    # warm-up（不计时）
    dist.reduce_scatter(output=output, input_list=input_list)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        dist.reduce_scatter(output=output, input_list=input_list)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if rank == 0:
        mib = total_elems * 4 / 1024 / 1024
        log(f"[BENCH] payload={mib:.0f}MB/rank  iters={iters}  "
            f"total={elapsed:.4f}s  avg={elapsed / iters * 1000:.3f} ms/iter")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
