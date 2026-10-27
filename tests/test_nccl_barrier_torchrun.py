"""只测 barrier 的最小自检（torchrun 启动，零参数）。

单独把 barrier 剥出来的原因：powerone2 上整套 NCCL 测试卡在第一步 barrier，
需要知道是"第一次 barrier 就挂"还是"跑几次之后挂"还是"子组上的 barrier 挂"。

每一步打一行 `[rank N] +<耗时>s >>> <步骤>`，卡住时最后打印的那行就是卡住的位置。

超时：默认组和 nccl 子组的进程组上都写死 timeout=timedelta(seconds=60)，
超时后 c10d watchdog 会抛异常结束进程，不要从外部发 SIGINT/SIGTERM/kill。

跑法（u13）：
    cd /data1/tangruijing/udtca/bitscom
    export LD_LIBRARY_PATH=/usr/local/corex/lib64:$LD_LIBRARY_PATH
    CUDA_VISIBLE_DEVICES=0,1 /root/miniconda3/envs/udtca/bin/python \
        -m torch.distributed.run --standalone --nproc_per_node=2 \
        --master_port=29515 tests/test_nccl_barrier_torchrun.py

跑法（powerone2 容器）：
    cd /data/tangruijing/udtca/bitscom
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29515 \
        tests/test_nccl_barrier_torchrun.py
"""

import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

T0 = time.monotonic()
RANK = int(os.environ.get("RANK", "0"))


def step(msg: str) -> None:
    """打点：卡住时最后一行就是卡住的步骤。"""
    print(f"[rank {RANK}] +{time.monotonic() - T0:7.3f}s >>> {msg}", flush=True)


def done(msg: str) -> None:
    print(f"[rank {RANK}] +{time.monotonic() - T0:7.3f}s OK  {msg}", flush=True)


def main() -> int:
    if "LOCAL_RANK" not in os.environ:
        print("ERROR: 必须用 torchrun 启动，见文件头部用法。", file=sys.stderr)
        return 2

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        print(f"ERROR: 至少 2 个进程，当前 WORLD_SIZE={world_size}", file=sys.stderr)
        return 2

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    try:
        dist.init_process_group(backend="nccl",
                                timeout=timedelta(seconds=60),
                                device_id=device)
    except TypeError:  # 老版本 torch 没有 device_id
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))

    if RANK == 0:
        print(
            f"=== barrier 专项自检 ===\n"
            f"world_size={world_size} torch={torch.__version__} timeout=60s\n"
            f"device={torch.cuda.get_device_name(local_rank)}",
            flush=True,
        )

    rc = 0
    current = "?"
    try:
        current = "cuda.synchronize"
        step(current)
        torch.cuda.synchronize()
        done(current)

        current = "barrier() 默认组 第1次"
        step(current)
        dist.barrier()
        done(current)

        current = "barrier() 默认组 连跑 5 次"
        step(current)
        for i in range(5):
            print(f"[rank {RANK}]     barrier #{i + 1}", flush=True)
            dist.barrier()
        done(current)

        current = "new_group(backend=nccl, timeout=60s)"
        step(current)
        try:
            pg = dist.new_group(ranks=list(range(world_size)),
                                backend="nccl",
                                timeout=timedelta(seconds=60),
                                device_id=device)
        except TypeError:
            pg = dist.new_group(ranks=list(range(world_size)),
                                backend="nccl",
                                timeout=timedelta(seconds=60))
        done(current)

        current = "子组上 barrier 3 次"
        step(current)
        for i in range(3):
            print(f"[rank {RANK}]     subgroup barrier #{i + 1}", flush=True)
            dist.barrier(group=pg)
        done(current)

        current = "先 all_reduce 再 barrier"
        step(current)
        x = torch.ones(1024, dtype=torch.float32, device=device)
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=pg)
        dist.barrier(group=pg)
        if float(x[0].item()) != float(world_size):
            raise AssertionError(f"all_reduce 结果 {float(x[0].item())} != {world_size}")
        done(current)

        current = "收尾 barrier"
        step(current)
        dist.barrier()
        done(current)

        if RANK == 0:
            print("=== ALL CHECKS PASSED ===", flush=True)
    except Exception as exc:  # noqa: BLE001
        rc = 1
        print(f"[rank {RANK}] FAILED at '{current}': {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
    finally:
        try:
            torch.cuda.synchronize()
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    return rc


if __name__ == "__main__":
    sys.exit(main())
