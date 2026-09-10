"""
跨机 bitscom 性能测试（torchrun 启动，单脚本，两节点共用）。

启动方式（先 master 后 worker）：

    # master 节点 10.31.10.62（node_rank=0）
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
        --master_addr=10.31.10.62 --master_port=29500 \
        tests/test_perf_cross_node.py

    # worker 节点 10.31.10.210（node_rank=1）
    torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 \
        --master_addr=10.31.10.62 --master_port=29500 \
        tests/test_perf_cross_node.py

node_rank / local_rank / rank 全部由 torchrun 注入环境变量，本文件不自行 spawn
进程，因此两节点跑的是同一个脚本，不存在两份文件不同步的问题。

排障开关（均为可选环境变量）：
    NCCL_SOCKET_IFNAME=eth1  直连网卡名。直连网卡不是默认路由网卡时，NCCL 会选错
                             接口导致握手/collective 卡死，必须显式指定。用 `ip a`
                             查 10.31.x.x 对应的网卡名。
    BITSCOM_NCCL_DEBUG=1     打开 NCCL 详细日志（NCCL_DEBUG=INFO），卡死时定位用。
"""

import os
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist

import bitscom
from bitscom.quantization import DEFAULT_BLOCK_SIZE

from torch.profiler import ProfilerActivity, profile, record_function


# ================= 可调参数 =================
COUNT = 4                      # 每个 backend 的迭代次数
ELEMS = 1024 * 1024            # 每个 tensor 元素数（FP32 => 4MB）

# 直连网卡名，务必按环境设置（见文件头说明）
NCCL_SOCKET_IFNAME = os.environ.get("NCCL_SOCKET_IFNAME", "ens1f0")
if NCCL_SOCKET_IFNAME:
    os.environ["NCCL_SOCKET_IFNAME"] = NCCL_SOCKET_IFNAME

# 卡死时打开 NCCL 详细日志
if os.environ.get("BITSCOM_NCCL_DEBUG", "0") == "1":
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,NET,COLL")


def log(*args):
    # 全部 flush：进程卡死时，缓冲的 stdout 会丢，flush 保证最后一条日志可见。
    print(*args, flush=True)


def _local_ip():
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
    """每个 rank 打印一行关键信息，用于核对跨机拓扑 / 版本一致性。"""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    host = socket.gethostname()
    ip = _local_ip()
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(local_rank if local_rank >= 0 else 0)
    else:
        dev = "NO-CUDA"
    nccl_ver = torch.cuda.nccl.version() if torch.cuda.is_available() else -1
    log(
        f"[init] rank={rank}/{world_size} local_rank={local_rank} "
        f"host={host} ip={ip}\n"
        f"        gpu={dev} torch={torch.__version__} cuda={torch.version.cuda} "
        f"nccl={nccl_ver} socket_ifname={NCCL_SOCKET_IFNAME or '(auto)'}"
    )


def main():
    # torchrun 已注入 RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    log(f"[{rank}] step0: cuda device set to {local_rank} "
        f"({torch.cuda.get_device_name(local_rank)})")

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

    # 显式 timeout，避免网络不通时无限挂起（torchrun 默认也有，这里兜底）
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=60),
    )
    log(f"[{rank}] step2: init_process_group done (nccl)")

    dump_env(rank, world_size)

    bitscom_pg = dist.new_group(ranks=list(range(world_size)), backend="lowbit")
    standard_pg = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    log(f"[{rank}] step3: new_group(lowbit/nccl) done")

    dist.barrier()
    if rank == 0:
        log(f"[{rank}] step4: all {world_size} ranks ready, start bench")

    torch.set_default_device(f"cuda:{torch.cuda.current_device()}")

    def bench_bitscom():
        # bitscom 的 lowbit backend 沿用 list 形式的 reduce_scatter
        # （input_list 长度为 world_size，每个是本 rank 贡献的满尺寸张量）。
        r = dist.get_rank()
        log(f"[{r}] [BITSCOM] 1/5 make_io begin")
        input_list = [torch.randn(ELEMS) for _ in range(world_size)]
        output = torch.zeros(ELEMS // world_size)
        log(f"[{r}] [BITSCOM] 1/5 make_io done ({len(input_list)} x {ELEMS} elem)")

        log(f"[{r}] [BITSCOM] 2/5 barrier begin")
        dist.barrier(group=bitscom_pg)
        log(f"[{r}] [BITSCOM] 2/5 barrier done")

        torch.cuda.synchronize()
        log(f"[{r}] [BITSCOM] 3/5 warm sync done, {COUNT} reduce_scatter begin")

        start = time.perf_counter()
        for i in range(COUNT):
            log(f"[{r}] [BITSCOM] 4/5 reduce_scatter iter {i} call")
            dist.reduce_scatter(output=output, input_list=input_list, group=bitscom_pg)
            log(f"[{r}] [BITSCOM] 4/5 reduce_scatter iter {i} returned")
        log(f"[{r}] [BITSCOM] 4/5 all iters dispatched, sync begin")
        torch.cuda.synchronize()
        log(f"[{r}] [BITSCOM] 5/5 sync done")

        end = time.perf_counter()
        elapsed = end - start
        avg_ms = elapsed / COUNT * 1000
        bytes_per_iter = world_size * ELEMS * 4
        bw_gbps = bytes_per_iter / (elapsed / COUNT) / 1e9
        log(f"[{r}] [BITSCOM] {COUNT} iters: total {elapsed:.4f}s, "
            f"avg {avg_ms:.2f}ms, ~{bw_gbps:.2f} GB/s")

    def bench_standard():
        # 标准 NCCL 用 reduce_scatter_tensor：单个输入张量，自动按 world_size 沿 dim0 切分。
        # input 形状 = [ELEMS * world_size]，output 形状 = [ELEMS]（本 rank 的分片）。
        r = dist.get_rank()
        log(f"[{r}] [STANDARD] 1/5 make_io begin")
        input_tensor = torch.randn(ELEMS * world_size)
        output = torch.zeros(ELEMS)
        log(f"[{r}] [STANDARD] 1/5 make_io done (input {ELEMS * world_size} elem)")

        log(f"[{r}] [STANDARD] 2/5 barrier begin")
        dist.barrier(group=standard_pg)
        log(f"[{r}] [STANDARD] 2/5 barrier done")

        torch.cuda.synchronize()
        log(f"[{r}] [STANDARD] 3/5 warm sync done, {COUNT} reduce_scatter_tensor begin")

        start = time.perf_counter()
        for i in range(COUNT):
            log(f"[{r}] [STANDARD] 4/5 reduce_scatter_tensor iter {i} call")
            dist.reduce_scatter_tensor(output, input_tensor, group=standard_pg)
            log(f"[{r}] [STANDARD] 4/5 reduce_scatter_tensor iter {i} returned")
        log(f"[{r}] [STANDARD] 4/5 all iters dispatched, sync begin")
        torch.cuda.synchronize()
        log(f"[{r}] [STANDARD] 5/5 sync done")

        end = time.perf_counter()
        elapsed = end - start
        avg_ms = elapsed / COUNT * 1000
        bytes_per_iter = world_size * ELEMS * 4
        bw_gbps = bytes_per_iter / (elapsed / COUNT) / 1e9
        log(f"[{r}] [STANDARD] {COUNT} iters: total {elapsed:.4f}s, "
            f"avg {avg_ms:.2f}ms, ~{bw_gbps:.2f} GB/s")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function("bitscom"):
            bench_bitscom()
        with record_function("standard"):
            bench_standard()

    if rank == 0:
        prof.export_chrome_trace("./trace.json")
        log("[0] trace exported to ./trace.json")

    dist.barrier()
    dist.destroy_process_group()
    log(f"[{rank}] done")


if __name__ == "__main__":
    main()
