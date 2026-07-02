"""
测试：不注册 lowbit backend，直接使用 LowBitGroup 包装 NCCL group 进行量化通信。

单机（>=2 GPU）：
    torchrun --nproc_per_node=2 bitscom/tests/test_lowbit_without_backend.py

跨节点（2 节点 × 1 卡）：
    节点0: torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
              --master_addr=10.31.10.210 --master_port=29500 \
              bitscom/tests/test_lowbit_without_backend.py
    节点1: torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
              --master_addr=10.31.10.210 --master_port=29500 \
              bitscom/tests/test_lowbit_without_backend.py
"""

import torch
import torch.distributed as dist
from bitscom.api import LowBitGroup


def log(msg):
    print(f"[rank {dist.get_rank()}] {msg}", flush=True)


def test_step_by_step():
    """逐步验证 NCCL 和 bitscom 通信，每一步带 log 定位卡死位置。"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # ── Step 1: NCCL all_reduce 连通性 ──
    log("step 1/5: testing NCCL all_reduce...")
    t = torch.ones(128, dtype=torch.float32, device=device) * (rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    assert t[0].item() == expected, f"NCCL all_reduce failed: {t[0].item()} != {expected}"
    log("step 1/5: NCCL all_reduce OK")

    # ── Step 2: LowBitGroup 创建 ──
    log("step 2/5: creating LowBitGroup(bitwidth=4)...")
    lb = LowBitGroup(bitwidth=4, process_group=None)
    log(f"step 2/5: LowBitGroup created, pg world_size={lb.world_size}")

    # ── Step 3: bitscom 小 tensor (128 elem) ──
    log("step 3/5: bitscom all_reduce on 128 elements...")
    t = torch.ones(128, dtype=torch.float32, device=device) * (rank + 1)
    t_ref = t.clone()
    dist.all_reduce(t_ref, op=dist.ReduceOp.SUM)
    lb.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(t, t_ref, rtol=0.01, atol=0.5)
    log("step 3/5: bitscom small tensor OK")

    # ── Step 4: bitscom 中等 tensor (64K elem) ──
    log("step 4/5: bitscom all_reduce on 65536 elements...")
    t = torch.randn(65536, dtype=torch.float32, device=device)
    t_ref = t.clone()
    dist.all_reduce(t_ref, op=dist.ReduceOp.SUM)
    lb.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(t, t_ref, rtol=0.01, atol=0.5)
    log("step 4/5: bitscom 64K OK")

    # ── Step 5: bitscom 大 tensor (1M elem) ──
    log("step 5/5: bitscom all_reduce on 1M elements (~4MB)...")
    t = torch.randn(1_048_576, dtype=torch.float32, device=device)
    t_ref = t.clone()
    dist.all_reduce(t_ref, op=dist.ReduceOp.SUM)
    lb.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(t, t_ref, rtol=0.01, atol=0.5)
    log("step 5/5: bitscom 1M OK — correctness all PASSED")


def test_bandwidth():
    """不同规模下 NCCL vs bitscom(4bit) vs bitscom(2bit) 带宽对比。"""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    lb_4 = LowBitGroup(bitwidth=4, process_group=None)
    lb_2 = LowBitGroup(bitwidth=2, process_group=None)

    sizes = [
        ("10MB",   2_621_440),
        ("50MB",  13_107_200),
        ("100MB", 26_214_400),
        ("200MB", 52_428_800),
        ("400MB", 104_857_600),
    ]

    show_header = (rank == 0)
    for label, numel in sizes:
        t_nccl = torch.randn(numel, dtype=torch.float32, device=device)
        t_4b = t_nccl.clone()
        t_2b = t_nccl.clone()

        # warmup
        dist.all_reduce(t_nccl, op=dist.ReduceOp.SUM)
        lb_4.all_reduce(t_4b, op=dist.ReduceOp.SUM)
        lb_2.all_reduce(t_2b, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # bench NCCL
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        n_iters = 8
        nccl_ms = []
        for _ in range(n_iters):
            e0.record()
            dist.all_reduce(t_nccl, op=dist.ReduceOp.SUM)
            e1.record()
            torch.cuda.synchronize()
            nccl_ms.append(e0.elapsed_time(e1))
        nccl_avg = sum(nccl_ms) / len(nccl_ms)

        # bench 4-bit
        b4_ms = []
        for _ in range(n_iters):
            e0.record()
            lb_4.all_reduce(t_4b, op=dist.ReduceOp.SUM)
            e1.record()
            torch.cuda.synchronize()
            b4_ms.append(e0.elapsed_time(e1))
        b4_avg = sum(b4_ms) / len(b4_ms)

        # bench 2-bit
        b2_ms = []
        for _ in range(n_iters):
            e0.record()
            lb_2.all_reduce(t_2b, op=dist.ReduceOp.SUM)
            e1.record()
            torch.cuda.synchronize()
            b2_ms.append(e0.elapsed_time(e1))
        b2_avg = sum(b2_ms) / len(b2_ms)

        if show_header:
            print(f"\n  {'Size':>8}  {'NCCL':>10}  {'bitscom-4b':>12}  "
                  f"{'speedup':>8}  {'bitscom-2b':>12}  {'speedup':>8}")
            print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*12}  {'-'*8}")

        log(f"  {label:>8}  {nccl_avg:>8.1f}ms  {b4_avg:>10.1f}ms  "
            f"{nccl_avg/b4_avg:>6.2f}x  {b2_avg:>10.1f}ms  {nccl_avg/b2_avg:>6.2f}x")


def main():
    import os
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    log(f"world_size={dist.get_world_size()}")

    test_step_by_step()
    test_bandwidth()

    dist.destroy_process_group()
    log("done")


if __name__ == "__main__":
    main()
