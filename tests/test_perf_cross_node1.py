"""
跨机分布式性能测试 —— 节点 1（worker）。

在 worker 节点 (10.31.10.210) 上运行：
    python tests/test_perf_cross_node_node1.py

本文件会 spawn 8 个本地进程（对应 8 张 GPU），占据全局 rank 8-15。
先确保 master 节点 (10.31.10.62) 已经运行 test_perf_cross_node_node0.py，
再运行本文件。

两个文件唯一的区别是 NODE_RANK（0 vs 1）。
"""

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import bitscom
from bitscom.quantization import DEFAULT_BLOCK_SIZE

from torch.profiler import ProfilerActivity, profile, record_function

# ================= 拓扑配置（两个文件唯一需要改的地方）=================
MASTER_ADDR = "10.31.10.62"     # master 节点 IP（在远端，本节点去连它）
MASTER_PORT = "29500"           # rendezvous 端口，需与 master 一致
NODE_RANK = 1                   # 本节点在全局中的编号：master=0, worker=1
N_GPUS = 8
N_NODES = 2
WORLD_SIZE = N_GPUS * N_NODES   # 16

COUNT = 4                       # 每个 backend 迭代次数
ELEMS = 1024 * 1024             # 每个 tensor 的元素个数（FP32 = 4MB）

# 直连网卡名。若系统默认网卡不是 10.31.x.x 的直连网卡，NCCL 可能选错 NIC。
# 用 `ip a` 找到 10.31.x.x 对应的网卡名（如 eth1 / ens8）填在这里，
# 或运行时用环境变量覆盖：NCCL_SOCKET_IFNAME=eth1 python ...
NCCL_SOCKET_IFNAME = os.environ.get("NCCL_SOCKET_IFNAME", "")

_bitscom_pg: dist.ProcessGroup = None
_standard_pg: dist.ProcessGroup = None


def init_worker(local_rank: int):
    global _bitscom_pg, _standard_pg
    global_rank = NODE_RANK * N_GPUS + local_rank

    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
    if NCCL_SOCKET_IFNAME:
        os.environ["NCCL_SOCKET_IFNAME"] = NCCL_SOCKET_IFNAME

    torch.cuda.set_device(local_rank)

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

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=global_rank,
        world_size=WORLD_SIZE,
    )

    _bitscom_pg = dist.new_group(ranks=list(range(WORLD_SIZE)), backend="lowbit")
    _standard_pg = dist.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

    torch.set_default_device(f"cuda:{torch.cuda.current_device()}")

    if global_rank == 0:
        print(f"[TOPOLOGY] world_size={WORLD_SIZE} master={MASTER_ADDR}:{MASTER_PORT}")


def _make_io(world_size: int):
    input_list = [torch.randn(ELEMS) for _ in range(world_size)]
    output = torch.zeros(ELEMS // world_size)
    return input_list, output


def bench_bitscom():
    world_size = dist.get_world_size()
    input_list, output = _make_io(world_size)
    dist.barrier()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(COUNT):
        dist.reduce_scatter(output=output, input_list=input_list, group=_bitscom_pg)
    torch.cuda.synchronize()
    end = time.time()
    if dist.get_rank() == 0:
        print(f"[BITSCOM] {COUNT} iters: {end - start:.4f}s, "
              f"avg {(end - start) / COUNT * 1000:.2f}ms")


def bench_standard():
    world_size = dist.get_world_size()
    input_list, output = _make_io(world_size)
    dist.barrier()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(COUNT):
        dist.reduce_scatter(output=output, input_list=input_list, group=_standard_pg)
    torch.cuda.synchronize()
    end = time.time()
    if dist.get_rank() == 0:
        print(f"[STANDARD] {COUNT} iters: {end - start:.4f}s, "
              f"avg {(end - start) / COUNT * 1000:.2f}ms")


def worker(local_rank: int):
    init_worker(local_rank)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function("bitscom"):
            bench_bitscom()
        with record_function("standard"):
            bench_standard()

    if dist.get_rank() == 0:
        prof.export_chrome_trace("./trace_node1.json")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, nprocs=N_GPUS, join=True)
