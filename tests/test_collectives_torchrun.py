"""bitscom 集合通信自检脚本（torchrun 启动，不走 pytest，也不用 mp.spawn）。

覆盖四类通信：
  1. NCCL all-reduce（全精度基线，逐元素校验）
  2. NCCL 点对点 send/recv（阻塞 send/recv + 异步 isend/irecv，含 PP 激活形状 bf16）
  3. bitscom 低比特 all-reduce 的 C++ backend 路径
     （LowBitGroup(backend_allreduce=True) on dist.new_group(backend="lowbit")，
      这就是 run_qwen14b_polar_dp_pp_tp.py 里 DP 梯度同步走的那条路）
     校验：各 rank 结果一致 / 全 finite / 与 Python 路径(backend_allreduce=False)一致
  4. 同一路径的 bf16 版本（训练里的 grad 就是 bf16）

跑法（在容器里、仓库根目录的 bitscom 下）：

    cd /data/tangruijing/udtca/bitscom
    export CUDA_VISIBLE_DEVICES=0,1
    torchrun --nproc_per_node=2 --master_port=29511 tests/test_collectives_torchrun.py

    # 只测 NCCL（all-reduce + send/recv），跳过低比特部分
    torchrun --nproc_per_node=2 --master_port=29511 tests/test_collectives_torchrun.py --skip-lowbit

    # 按训练里那个 DP bucket 的量级压一压（6400 万元素 ≈ 256MB fp32 / rank）
    torchrun --nproc_per_node=2 --master_port=29511 tests/test_collectives_torchrun.py \
        --numel 64000000 --bitwidth 2

超时：两个进程组都在 init 时写死 `timeout=timedelta(seconds=60)`，任何集合通信
超过 60s 没完成，c10d watchdog 会自己判超时并把异常抛出来结束进程 ——
**不需要、也不应该从外部发 SIGINT/SIGTERM/kill**（Corex 4.5.0 下终止在途 NCCL 会触发
XID 24 MMU page fault，卡会永久挂死，见 .claude/memory/corex-mmu-wedge-sigkill.md）。
所有用例正常都是秒级，跑完会 barrier + 按顺序 destroy 进程组。

注意：超时是 c10d 在内部主动 abort 那个通信实现的，这条路和外部 kill 一样有可能碰
XID 24，它只是保证"不会无限期挂着、不用人去杀"，不等于绝对安全。
"""

import argparse
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

# bitscom 只在真正跑低比特用例时才 import：--skip-lowbit 时这个进程里
# 完全不出现 bitscom 的 C++ 扩展和它链的 NCCL，纯看 NCCL 自己。


# ---------------------------------------------------------------- 工具

def _barrier() -> None:
    torch.cuda.synchronize()
    dist.barrier()


def _cross_rank_max_diff(tensor: torch.Tensor) -> float:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    diff = torch.stack([(g - gathered[0]).abs().max() for g in gathered]).max()
    return float(diff.to(torch.float32).item())


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


# ---------------------------------------------------------------- 1. NCCL all-reduce

def check_nccl_all_reduce(device: torch.device) -> None:
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    numel = 4096
    x = torch.full((numel,), float(rank + 1), dtype=torch.float32, device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    expected = float(sum(range(1, world_size + 1)))
    got = float(x[0].item())
    err = abs(got - expected) / expected
    _assert(err < 1e-6, f"NCCL all_reduce 结果错: got={got} expected={expected}")
    _barrier()
    if rank == 0:
        print(f"[PASS] 1) NCCL all_reduce SUM  {got} (期望 {expected})", flush=True)


# ---------------------------------------------------------------- 2. NCCL send/recv

def check_p2p_blocking(device: torch.device) -> None:
    rank = dist.get_rank()
    numel = 4096

    # 0 -> 1
    if rank == 0:
        payload = torch.arange(numel, dtype=torch.float32, device=device)
        dist.send(payload, dst=1)
    elif rank == 1:
        buf = torch.empty(numel, dtype=torch.float32, device=device)
        dist.recv(buf, src=0)
        expected = torch.arange(numel, dtype=torch.float32, device=device)
        _assert(bool(torch.equal(buf, expected)), "0->1 阻塞 send/recv 内容不一致")
    _barrier()

    # 1 -> 0
    if rank == 1:
        payload = torch.arange(numel, dtype=torch.float32, device=device) * 2.0
        dist.send(payload, dst=0)
    elif rank == 0:
        buf = torch.empty(numel, dtype=torch.float32, device=device)
        dist.recv(buf, src=1)
        expected = torch.arange(numel, dtype=torch.float32, device=device) * 2.0
        _assert(bool(torch.equal(buf, expected)), "1->0 阻塞 send/recv 内容不一致")
    _barrier()

    if rank == 0:
        print("[PASS] 2) NCCL send/recv 阻塞双向", flush=True)


def check_p2p_async(device: torch.device, seq_len: int, hidden: int) -> None:
    """PP 形状的异步 p2p：前向激活 0->1，反向梯度 1->0。"""
    rank = dist.get_rank()
    shape = (1, seq_len, hidden)

    # 两端用同一个 seed，这样收发双方各自能造出同一份内容做比对。
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    fwd = torch.randn(shape, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(device)
    dgrad = (fwd.float() * 0.5).to(torch.bfloat16)

    # 前向：rank0 isend -> rank1 irecv
    works = []
    if rank == 0:
        works.append(dist.isend(fwd, dst=1))
    elif rank == 1:
        recv_buf = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        works.append(dist.irecv(recv_buf, src=0))
    for w in works:
        w.wait()
    if rank == 1:
        _assert(bool(torch.equal(recv_buf, fwd)), "PP 形状 0->1 异步 isend/irecv 内容不一致")
    _barrier()

    # 反向：rank1 isend -> rank0 irecv（形状同激活，模拟 dgrad）
    works = []
    if rank == 1:
        works.append(dist.isend(dgrad, dst=0))
    elif rank == 0:
        grad_buf = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        works.append(dist.irecv(grad_buf, src=1))
    for w in works:
        w.wait()
    if rank == 0:
        _assert(bool(torch.equal(grad_buf, dgrad)), "PP 形状 1->0 异步 isend/irecv 内容不一致")
    _barrier()

    if rank == 0:
        print(f"[PASS] 3) NCCL 异步 isend/irecv PP 形状 {tuple(shape)} bf16 双向", flush=True)


# ---------------------------------------------------------------- 3. 低比特 all-reduce

def _lowbit_check(
    device: torch.device,
    lowbit_pg,
    numel: int,
    bitwidth: int,
    dtype: torch.dtype,
    tag: str,
) -> None:
    from bitscom.api import LowBitGroup

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    gen = torch.Generator(device="cpu")
    gen.manual_seed(7 + rank)
    x32 = (torch.randn(numel, generator=gen, dtype=torch.float32) * 0.01 + 0.001 * rank).to(device)

    # 全精度参照（同一个 world，nccl backend，始终用 fp32 算）
    ref = x32.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)

    # 被测输入按用例 dtype 走（bf16 用例就是在压 C++ 侧的 bf16 分支）
    x = x32.to(dtype)

    # C++ backend 路径：pack -> NCCL -> unpack，训练里走的就是这条
    backend_group = LowBitGroup(
        bitwidth=bitwidth, process_group=lowbit_pg, backend_allreduce=True
    )
    out_backend = x.clone()
    work = backend_group.all_reduce(out_backend, op=dist.ReduceOp.SUM, async_op=True)
    _assert(work is not None, "async_op=True 必须返回 Work")
    work.wait()

    # Python alltoall 路径作为 oracle；它固定跑 fp32，bf16 用例只在 C++ 侧变 dtype。
    py_group = LowBitGroup(
        bitwidth=bitwidth, process_group=lowbit_pg, backend_allreduce=False
    )
    out_py = x.clone()
    py_group.all_reduce(out_py, op=dist.ReduceOp.SUM, async_op=True).wait()

    out_backend = out_backend.to(torch.float32).clone()
    out_py = out_py.to(torch.float32)
    ref32 = ref.to(torch.float32)

    finite = torch.isfinite(out_backend).all().to(torch.float32)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)

    scale = ref32.abs().max().clamp_min(1e-12)

    cross = _cross_rank_max_diff(out_backend)
    # backend 与 python 路径的相对偏差。bf16 用例本身带着 bf16 存储精度，阈值放宽。
    vs_py_rel = (out_backend - out_py).abs().max() / scale
    # 相对全精度 all-reduce 的误差，只报告不判定（低比特本来就有损）。
    rel_dense = (out_backend - ref32).abs().mean() / ref32.abs().mean().clamp_min(1e-12)

    stats = torch.stack([vs_py_rel, rel_dense]).to(torch.float32)
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    vs_py_rel_max = float(stats[0].item())
    rel_dense_max = float(stats[1].item())

    tol = 1e-2 if dtype == torch.float32 else 5e-2
    _assert(float(finite.item()) == 1.0, f"{tag}: 结果里有非有限值")
    _assert(cross < 1e-5, f"{tag}: 各 rank 结果不一致，cross_rank_max_diff={cross}")
    _assert(
        vs_py_rel_max < tol,
        f"{tag}: backend 路径与 Python 路径差太多 rel={vs_py_rel_max:.3e} (tol={tol})",
    )

    _barrier()
    if rank == 0:
        print(
            f"[PASS] {tag}  numel={numel} dtype={dtype} bw={bitwidth} "
            f"cross_rank_max_diff={cross:.3e} "
            f"backend_vs_python_rel={vs_py_rel_max:.3e} (tol={tol:.0e}) "
            f"mean_rel_err_vs_dense={rel_dense_max:.3f}",
            flush=True,
        )


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description="bitscom collectives self-check (torchrun)")
    parser.add_argument("--bitwidth", type=int, default=2)
    parser.add_argument("--numel", type=int, default=1 << 20,
                        help="低比特 all-reduce 的 fp32 buffer 元素数（默认 1M）")
    parser.add_argument("--bf16-numel", type=int, default=1 << 18,
                        help="bf16 用例的元素数（默认 256K）")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--skip-lowbit", action="store_true",
                        help="只测 NCCL all-reduce 与 send/recv")
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        print("ERROR: 这个脚本必须用 torchrun 启动，见文件头部的用法。", file=sys.stderr)
        return 2

    local_rank = int(os.environ["LOCAL_RANK"])
    # Corex: ncclCommInitRank 绑定的是“当前设备”，必须在建任何 group 之前设好。
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size < 2:
        print(f"ERROR: 至少需要 2 个进程，当前 WORLD_SIZE={world_size}", file=sys.stderr)
        return 2

    if not args.skip_lowbit:
        import bitscom
        # 注册 lowbit backend（要在自己建 lowbit group 之前）。
        bitscom.init(bitwidth=args.bitwidth)
    # timeout 挂在进程组上：任何集合通信超过 60s 没完成，c10d watchdog 会把它
    # 判超时并抛异常。这样测试自己会结束，不需要外部 SIGINT/SIGTERM。
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))

    if rank == 0:
        prop = torch.cuda.get_device_properties(local_rank)
        print(
            f"=== bitscom collectives self-check ===\n"
            f"world_size={world_size} bitwidth={args.bitwidth} "
            f"numel={args.numel} bf16_numel={args.bf16_numel} timeout=60s\n"
            f"torch={torch.__version__} device={prop.name}",
            flush=True,
        )

    try:
        check_nccl_all_reduce(device)
        # p2p 只用 rank0 <-> rank1，多进程时其余 rank 只参与 barrier。
        check_p2p_blocking(device)
        check_p2p_async(device, args.seq_len, args.hidden)

        if args.skip_lowbit:
            if rank == 0:
                print("[SKIP] 低比特 all-reduce（--skip-lowbit）", flush=True)
            _barrier()
        else:
            lowbit_pg = dist.new_group(
                ranks=list(range(world_size)),
                backend="lowbit",
                timeout=timedelta(seconds=60),
            )
            _lowbit_check(
                device, lowbit_pg, args.numel, args.bitwidth, torch.float32, "4) lowbit all_reduce fp32"
            )
            _lowbit_check(
                device, lowbit_pg, args.bf16_numel, args.bitwidth, torch.bfloat16,
                "5) lowbit all_reduce bf16",
            )

        _barrier()
        if rank == 0:
            print("=== ALL CHECKS PASSED ===", flush=True)

    finally:
        # 退出顺序：barrier -> destroy 子组 -> destroy world。
        try:
            torch.cuda.synchronize()
            dist.barrier()
        except Exception:
            pass
        try:
            if not args.skip_lowbit and "lowbit_pg" in locals():
                dist.destroy_process_group(lowbit_pg)
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
