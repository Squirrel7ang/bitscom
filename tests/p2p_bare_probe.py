#!/usr/bin/env python3
"""裸 isend/irecv 双向收发探针 —— 判定“未批量 P2P 会死锁”是不是 powerone2 独有。

背景：本镜像 torch 的 Schedule1F1B 在 warmup / cooldown 阶段用的是裸
``dist.isend`` / ``dist.irecv``（``pipelining/schedules.py`` 的 ``_batch_p2p``
只把“混合方向”的批交给 ``batch_isend_irecv``），稳定态才把 send+recv 合批。
所以两台机器上都会走到裸 P2P；这里只回答一个问题：**这台机器上裸 P2P 是否可用**。

用例（与训练无关，不 import bitscom）：

  batched_mixed     对照组：send+recv 放进同一个 batch_isend_irecv
  warmup_bare       1F1B warmup 的形状：裸 irecv 等到，再裸 isend
  warmup_batch1     同上，但每个单独操作也包成 batch_isend_irecv（候选修法）
  bare_simultaneous 疑点用例：裸 isend 与裸 irecv 同时挂起（powerone2 上会死锁）
                    —— 默认不跑，必须显式 --cases 指定

数据按 PP 激活形状 bf16 (1, 256, 5120) 环状 +1 传递，逐元素精确校验，
每次迭代的 payload 不同，所以“收到了上一轮/垃圾数据”也会被抓到
（挂了会打成 constant，例如 2.36936e-38 那种 min==max 的整块常量）。

防挂死：主线程只在每个用例结束后 touch 心跳；超过 --hang-timeout 没有推进，
CommWatchdog 会调 ncclCommRevoke 撤掉本 rank 的全部 NCCL comm，等本机各 rank
都静默后再 os._exit(124)。全程不发任何信号、不 kill 进程 —— 也请不要给这个
脚本套 timeout 或从外面 kill，那正是会打出 XID 24 让 GPU 永久挂死的路径。

跑法（容器里，先确认没有别人的进程，机器状态正常）：

    cd /data/tangruijing/udtca/bitscom
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29531 \\
        tests/p2p_bare_probe.py 2>&1 | tee /data1/tangruijing/logs/p2p_probe_$(hostname).log

    # 只有想复现双向死锁时才加（会等到 --hang-timeout 后 revoke 退出）
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29531 \\
        tests/p2p_bare_probe.py --cases bare_simultaneous --iters 1 \\
        2>&1 | tee /data1/tangruijing/logs/p2p_probe_bare_$(hostname).log

判定：safe 用例全 PASS 且 bare_simultaneous 也 PASS/正常退出 ⇒ 这台机器的
裸 P2P 没问题；safe 用例通过、bare_simultaneous 卡住 ⇒ 与 powerone2 同病。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

SHAPE = (1, 256, 5120)
DTYPE = torch.bfloat16
# 进程组自身超时必须远大于 hang-timeout：先让 revoke 路径收尾，
# 不要让 torch 的 NCCL watchdog 去 abort 在途通信（Corex 上那会打 XID 24）。
PG_TIMEOUT_S = 900


def _log(rank: int, msg: str) -> None:
    print(f"[rank {rank}] {msg}", flush=True)


def _base() -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    return torch.randn(SHAPE, generator=gen, dtype=torch.float32)


def _payload(base: torch.Tensor, value: float, device) -> torch.Tensor:
    return (base + value).to(DTYPE).to(device)


def _describe(buf: torch.Tensor) -> str:
    mn = float(buf.min())
    mx = float(buf.max())
    flat = [float(v) for v in buf.reshape(-1)[:8]]
    kind = "constant" if mn == mx else "varying"
    return f"min={mn:.6g} max={mx:.6g} first8={flat} ({kind})"


def _verify(buf, base, src, it, device, name) -> None:
    expected = _payload(base, float(src) + 0.25 * it, device)
    if not torch.equal(buf, expected):
        raise AssertionError(
            f"{name} iter {it}: 从 {src} 收到的内容不对 | {_describe(buf)}"
        )


# ------------------------------------------------------------------ 用例


def case_batched_mixed(pg, rank, world, device, base, it) -> None:
    dst = (rank + 1) % world
    src = (rank - 1) % world
    local = _payload(base, float(rank) + 0.25 * it, device)
    buf = torch.full(SHAPE, float("nan"), dtype=DTYPE, device=device)
    ops = [
        dist.P2POp(dist.isend, local, dst, group=pg),
        dist.P2POp(dist.irecv, buf, src, group=pg),
    ]
    for work in dist.batch_isend_irecv(ops):
        work.wait()
    _verify(buf, base, src, it, device, "batched_mixed")


def case_bare_simultaneous(pg, rank, world, device, base, it) -> None:
    dst = (rank + 1) % world
    src = (rank - 1) % world
    local = _payload(base, float(rank) + 0.25 * it, device)
    buf = torch.full(SHAPE, float("nan"), dtype=DTYPE, device=device)
    send_work = dist.isend(local, dst, group=pg)
    recv_work = dist.irecv(buf, src, group=pg)
    send_work.wait()
    recv_work.wait()
    _verify(buf, base, src, it, device, "bare_simultaneous")


def case_warmup_bare(pg, rank, world, device, base, it) -> None:
    dst = (rank + 1) % world
    src = (rank - 1) % world
    local = _payload(base, float(rank) + 0.25 * it, device)
    buf = torch.full(SHAPE, float("nan"), dtype=DTYPE, device=device)
    dist.irecv(buf, src, group=pg).wait()
    dist.isend(local, dst, group=pg).wait()
    _verify(buf, base, src, it, device, "warmup_bare")


def case_warmup_batch1(pg, rank, world, device, base, it) -> None:
    dst = (rank + 1) % world
    src = (rank - 1) % world
    local = _payload(base, float(rank) + 0.25 * it, device)
    buf = torch.full(SHAPE, float("nan"), dtype=DTYPE, device=device)
    for work in dist.batch_isend_irecv([dist.P2POp(dist.irecv, buf, src, group=pg)]):
        work.wait()
    for work in dist.batch_isend_irecv([dist.P2POp(dist.isend, local, dst, group=pg)]):
        work.wait()
    _verify(buf, base, src, it, device, "warmup_batch1")


CASES = {
    "batched_mixed": case_batched_mixed,
    "warmup_bare": case_warmup_bare,
    "warmup_batch1": case_warmup_batch1,
    "bare_simultaneous": case_bare_simultaneous,
}
SAFE_CASES = ("batched_mixed", "warmup_bare", "warmup_batch1")


# ------------------------------------------------------------------ 工具


def _load_revoke():
    """加载 experiments/qwen14b/nccl_revoke.py；不可用时返回 None。"""
    revoke_dir = os.environ.get("UDTCA_REVOKE_DIR", "/data/tangruijing/udtca/experiments/qwen14b")
    if revoke_dir not in sys.path:
        sys.path.insert(0, revoke_dir)
    try:
        import nccl_revoke  # noqa: PLC0415 - 延迟到运行时再找
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: 加载 nccl_revoke 失败（{exc!r}），只能跑安全用例", flush=True)
        return None
    return nccl_revoke


def _new_group(world_size: int, device) -> "dist.ProcessGroup":
    kw = dict(ranks=list(range(world_size)), backend="nccl",
              timeout=timedelta(seconds=PG_TIMEOUT_S))
    try:
        return dist.new_group(device_id=device, **kw)
    except TypeError:  # 老版本 torch 没有 device_id
        return dist.new_group(**kw)


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default=",".join(SAFE_CASES),
                    help="逗号分隔，可选：%s" % ",".join(CASES))
    ap.add_argument("--iters", type=int, default=10, help="每个用例迭代次数")
    ap.add_argument("--hang-timeout", type=float, default=120.0,
                    help="多久没有推进就 revoke 退出（秒）")
    args = ap.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in cases if c not in CASES]
    if unknown:
        print(f"ERROR: 未知用例 {unknown}，可选 {list(CASES)}", file=sys.stderr)
        return 2

    if "LOCAL_RANK" not in os.environ:
        print("ERROR: 必须用 torchrun 启动，见文件头部用法。", file=sys.stderr)
        return 2
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size < 2:
        print(f"ERROR: 至少 2 个进程，当前 WORLD_SIZE={world_size}", file=sys.stderr)
        return 2

    revoke = _load_revoke()
    if "bare_simultaneous" in cases and revoke is None:
        print("ERROR: 没有 nccl_revoke，拒绝跑 bare_simultaneous（卡住后无法安全退出）",
              file=sys.stderr)
        return 2

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    try:
        dist.init_process_group(backend="nccl",
                                timeout=timedelta(seconds=PG_TIMEOUT_S),
                                device_id=device)
    except TypeError:
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=PG_TIMEOUT_S))
    pg = _new_group(world_size, device)

    if rank == 0:
        try:
            nccl_ver = torch.cuda.nccl.version()
        except Exception:  # noqa: BLE001
            nccl_ver = "unknown"
        print(
            f"=== 裸 P2P 探针 ===\n"
            f"host={os.uname().nodename} world_size={world_size} "
            f"torch={torch.__version__} nccl={nccl_ver}\n"
            f"device={torch.cuda.get_device_name(local_rank)} "
            f"shape={SHAPE} dtype={DTYPE} iters={args.iters}\n"
            f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '-')} "
            f"cases={cases}",
            flush=True,
        )
    if os.environ.get("NCCL_P2P_DISABLE", "0") not in ("0", ""):
        _log(rank, "WARN: NCCL_P2P_DISABLE 已设置，测到的是非直连路径，结论无效")

    watchdog = None
    if revoke is not None:
        watchdog = revoke.CommWatchdog(timeout_s=args.hang_timeout, exit_code=124)
        watchdog.capture(force=True)
        watchdog.touch()
        watchdog.start()

    base = _base()
    results = []
    rc = 0
    try:
        for name in cases:
            if watchdog is not None:
                watchdog.capture(force=True)
            _log(rank, f">>> {name} x{args.iters}")
            t_case = time.perf_counter()
            try:
                for it in range(args.iters):
                    CASES[name](pg, rank, world_size, device, base, it)
                    torch.cuda.synchronize()
                    dist.barrier(group=pg)
                    if watchdog is not None:
                        watchdog.touch()
                dt = time.perf_counter() - t_case
                results.append(f"{name}=PASS({dt:.1f}s)")
                _log(rank, f"[PASS] {name} x{args.iters} ({dt:.1f}s)")
            except Exception as exc:  # noqa: BLE001
                rc = 1
                results.append(f"{name}=FAIL({type(exc).__name__})")
                _log(rank, f"[FAIL] {name}: {type(exc).__name__}: {exc}")
                if watchdog is not None:
                    watchdog.touch()  # 失败也要继续，别让看门狗误判
        _log(rank, "SUMMARY " + " ".join(results))
    finally:
        if watchdog is not None:
            watchdog.stop()
        try:
            torch.cuda.synchronize()
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        for target in (pg, None):
            try:
                dist.destroy_process_group(target)
            except Exception:  # noqa: BLE001
                pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
