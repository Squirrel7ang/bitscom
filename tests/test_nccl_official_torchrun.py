"""torch 官方 NCCL 后端的集合通信自检（torchrun 启动，零参数）。

只用 torch.distributed 的官方接口 + NCCL 后端，不 import bitscom。
所有 NCCL 通信都跑在一个显式创建、带超时的 nccl 子进程组上：

    nccl_pg = dist.new_group(ranks=list(range(world_size)),
                             backend="nccl",
                             timeout=timedelta(seconds=60))

torch 的超时是挂在进程组上的，不是单个调用上；组内任何一次集合通信超过 60s
没完成，c10d 的 watchdog 就会判超时并抛异常，进程自己退出 ——
不需要、也不要从外部发 SIGINT/SIGTERM/kill。

覆盖：
  1) barrier
  2) all_reduce  SUM, float32（逐元素校验）
  3) all_reduce  SUM, bfloat16（训练里 DP 梯度就是 bf16）
  4) broadcast  (root=0)
  5) all_gather（按 rank 序号校验）
  6) reduce_scatter SUM
  7) send / recv 阻塞，环形 +1 方向
  8) batch_isend_irecv 异步，BF16 PP 激活形状 (1, 256, 5120)，环形 +1 方向
     （Corex torch 对未批量 P2P 做串行化，裸 isend/irecv 双向环形会死锁）

跑法（容器里）：

    cd /data/tangruijing/udtca/bitscom
    torchrun --nproc_per_node=2 --master_port=29511 \\
        tests/test_nccl_official_torchrun.py

    # 想固定用哪两张卡
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29511 \\
        tests/test_nccl_official_torchrun.py
"""

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist


def _barrier(pg, label: str) -> None:
    try:
        torch.cuda.synchronize()
        dist.barrier(group=pg)
    except Exception as exc:  # noqa: BLE001 - 只为了把卡住的步骤打出来
        raise RuntimeError(f"[{label}] barrier 失败/超时") from exc


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _pass(rank: int, text: str) -> None:
    if rank == 0:
        print(f"[PASS] {text}", flush=True)


# ------------------------------------------------------------------ 用例

def check_barrier(pg, rank: int, device) -> None:
    torch.cuda.synchronize()
    dist.barrier(group=pg)
    _pass(rank, "1) barrier")


def check_all_reduce_f32(pg, rank: int, world_size: int, device) -> None:
    n = 4096
    x = torch.full((n,), float(rank + 1), dtype=torch.float32, device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=pg)
    expected = float(sum(range(1, world_size + 1)))
    got = float(x[0].item())
    _check(abs(got - expected) < 1e-6, f"all_reduce float32: got={got} expected={expected}")
    _barrier(pg, "all_reduce float32")
    _pass(rank, f"2) all_reduce float32 SUM = {got:.1f}")


def check_all_reduce_bf16(pg, rank: int, world_size: int, device) -> None:
    n = 1024  # bf16 尾数只有 8 位，元素少一点保证 SUM 精确
    x = torch.full((n,), float(rank + 1), dtype=torch.bfloat16, device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=pg)
    expected = float(sum(range(1, world_size + 1)))
    got = float(x[0].float().item())
    _check(abs(got - expected) < 1e-2, f"all_reduce bf16: got={got} expected={expected}")
    _barrier(pg, "all_reduce bf16")
    _pass(rank, f"3) all_reduce bf16 SUM = {got:.1f}")


def check_broadcast(pg, rank: int, device) -> None:
    n = 2048
    if rank == 0:
        t = torch.arange(n, dtype=torch.float32, device=device)
    else:
        t = torch.full((n,), -1.0, dtype=torch.float32, device=device)
    dist.broadcast(t, src=0, group=pg)
    expected = torch.arange(n, dtype=torch.float32, device=device)
    _check(bool(torch.equal(t, expected)), "broadcast: rank0 的数据没有正确广播")
    _barrier(pg, "broadcast")
    _pass(rank, "4) broadcast (root=0)")


def check_all_gather(pg, rank: int, world_size: int, device) -> None:
    n = 256
    mine = torch.full((n,), float(rank), dtype=torch.float32, device=device)
    out = [torch.full((n,), -1.0, dtype=torch.float32, device=device)
           for _ in range(world_size)]
    dist.all_gather(out, mine, group=pg)
    for r, buf in enumerate(out):
        _check(bool((buf == float(r)).all()), f"all_gather: 第 {r} 块内容不对")
    _barrier(pg, "all_gather")
    _pass(rank, f"5) all_gather ({world_size} 块)")


def check_reduce_scatter(pg, rank: int, world_size: int, device) -> None:
    # 每个 rank 准备 world_size 块，第 i 块的初值是 rank*100+i，reducescatter 后
    # 第 i 块应该等于所有 rank 的 rank*100+i 之和。
    chunk = 512
    inputs = [torch.full((chunk,), float(rank * 100 + i), dtype=torch.float32, device=device)
              for i in range(world_size)]
    out = torch.empty(chunk, dtype=torch.float32, device=device)
    dist.reduce_scatter(out, inputs, op=dist.ReduceOp.SUM, group=pg)

    expected = float(sum(r * 100 + rank for r in range(world_size)))
    got = float(out[0].item())
    _check(abs(got - expected) < 1e-3,
           f"reduce_scatter: 第 {rank} 块 got={got} expected={expected}")
    _barrier(pg, "reduce_scatter")
    _pass(rank, f"6) reduce_scatter SUM (第 {rank} 块 = {got:.0f})")


def check_send_recv(pg, rank: int, world_size: int, device) -> None:
    """阻塞 send/recv：相邻两个 rank 成对，先 even->odd，再 odd->even。

    不用环形，所有 rank 同时先 send 再 recv 会互相等死；成对、分两轮就没有环。
    """
    n = 4096
    pair = rank ^ 1                      # 0<->1, 2<->3, ...
    if pair >= world_size:               # 奇数个进程时最后一个自己一个
        # 保持 barrier 次数和其他 rank 一致，否则会等死。
        _barrier(pg, "send/recv 第一轮")
        _barrier(pg, "send/recv 第二轮")
        _pass(rank, f"7) send/recv 跳过（rank {rank} 没有配对）")
        return

    is_even = (rank % 2 == 0)
    payload = torch.full((n,), float(rank), dtype=torch.float32, device=device)
    buf = torch.full((n,), -1.0, dtype=torch.float32, device=device)

    # 第一轮：偶数 -> 奇数
    if is_even:
        dist.send(payload, dst=pair, group=pg)
    else:
        dist.recv(buf, src=pair, group=pg)
        _check(bool((buf == float(pair)).all()), f"send/recv: 从 {pair} 收到的内容不对")
    _barrier(pg, "send/recv 第一轮")

    # 第二轮：奇数 -> 偶数
    buf.fill_(-1.0)
    if is_even:
        dist.recv(buf, src=pair, group=pg)
        _check(bool((buf == float(pair)).all()), f"send/recv: 从 {pair} 收到的内容不对")
    else:
        dist.send(payload, dst=pair, group=pg)
    _barrier(pg, "send/recv 第二轮")

    _pass(rank, f"7) send/recv 阻塞双向（{rank} <-> {pair}）")


def check_isend_irecv_pp(pg, rank: int, world_size: int, device,
                         seq_len: int = 256, hidden: int = 5120) -> None:
    """PP 激活形状的异步收发：bf16 (1, seq_len, hidden)，环形 +1。

    每个 rank 用同一个 seed 造同一份 base，再加上自己的 rank；
    所以从 src 收到的应当正好等于本地算出来的 base + src。
    """
    shape = (1, seq_len, hidden)
    dst = (rank + 1) % world_size
    src = (rank - 1 + world_size) % world_size

    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    base = torch.randn(shape, generator=gen, dtype=torch.float32)

    local = (base + float(rank)).to(torch.bfloat16).to(device)
    expected = (base + float(src)).to(torch.bfloat16).to(device)

    buf = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    # Corex torch 会按提交顺序串行化未批量的 P2P（运行时打印
    # "[WARNING unbatched P2P serialized]"）：单独 isend / irecv 时，
    # 两个 rank 都先卡在自己的 send 上等对端 recv，而 recv 还没提交，
    # 形成环等死锁。batch_isend_irecv 把 send+recv 放进同一个 group
    # 一次性提交，两半同时进 NCCL，配对才成立。
    p2p_ops = [
        dist.P2POp(dist.isend, local, dst, group=pg),
        dist.P2POp(dist.irecv, buf, src, group=pg),
    ]
    for work in dist.batch_isend_irecv(p2p_ops):
        work.wait()

    _check(bool(torch.equal(buf, expected)),
           f"isend/irecv PP 形状: 从 {src} 收到的内容不对")
    _barrier(pg, "isend/irecv PP")
    _pass(rank, f"8) isend/irecv bf16 {tuple(shape)} 环形 +1（{rank}->{dst}, {src}->{rank}）")


# ------------------------------------------------------------------ main

def main() -> int:
    if "LOCAL_RANK" not in os.environ:
        print("ERROR: 必须用 torchrun 启动，见文件头部用法。", file=sys.stderr)
        return 2

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size < 2:
        print(f"ERROR: 至少 2 个进程，当前 WORLD_SIZE={world_size}", file=sys.stderr)
        return 2

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 默认组也带上超时，避免 init 阶段无限挂。
    # device_id 显式钉住本进程用的卡：不传的话 torch 会警告 rank->GPU 映射可能错，
    # 在 Corex 上这种错配正是挂死的一种来源。
    try:
        dist.init_process_group(backend="nccl",
                                timeout=timedelta(seconds=60),
                                device_id=device)
    except TypeError:  # 老版本 torch 没有 device_id
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))

    # 被测的 NCCL 进程组：显式 backend + 显式超时。
    try:
        nccl_pg = dist.new_group(
            ranks=list(range(world_size)),
            backend="nccl",
            timeout=timedelta(seconds=60),
            device_id=device,
        )
    except TypeError:
        nccl_pg = dist.new_group(
            ranks=list(range(world_size)),
            backend="nccl",
            timeout=timedelta(seconds=60),
        )

    if rank == 0:
        try:
            nccl_ver = torch.cuda.nccl.version()
        except Exception:
            nccl_ver = "unknown"
        print(
            f"=== torch NCCL 官方接口自检 ===\n"
            f"world_size={world_size} torch={torch.__version__} "
            f"nccl={nccl_ver} timeout=60s\n"
            f"device={torch.cuda.get_device_name(local_rank)}",
            flush=True,
        )

    steps = [
        ("barrier", check_barrier, ()),
        ("all_reduce_f32", check_all_reduce_f32, (world_size,)),
        ("all_reduce_bf16", check_all_reduce_bf16, (world_size,)),
        ("broadcast", check_broadcast, ()),
        ("all_gather", check_all_gather, (world_size,)),
        ("reduce_scatter", check_reduce_scatter, (world_size,)),
        ("send_recv", check_send_recv, (world_size,)),
        ("isend_irecv_pp", check_isend_irecv_pp, (world_size,)),
    ]

    rc = 0
    current = "?"
    try:
        for name, fn, extra in steps:
            current = name
            print(f"[rank {rank}] >>> {name}", flush=True)
            fn(nccl_pg, rank, *extra, device)
        dist.barrier(group=nccl_pg)
        if rank == 0:
            print("=== ALL CHECKS PASSED ===", flush=True)
    except Exception as exc:  # noqa: BLE001
        rc = 1
        print(f"[rank {rank}] FAILED at step '{current}': {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
    finally:
        try:
            torch.cuda.synchronize()
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group(nccl_pg)
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    return rc


if __name__ == "__main__":
    sys.exit(main())
