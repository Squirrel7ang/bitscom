#!/usr/bin/env python3
"""成对连通性探针：逐对建 NCCL 组并收发，找出哪几对 GPU 有问题。

背景：2026-10-27 11:35 的 16 卡测试里，world comm（16 rank）建成功，但
`init_device_mesh` 第一次 `ncclCommSplit` 时，组 {0,8} 和 {1,9} 没建完，
其余 6 个 dp 组（{2,10}…{7,15}）都是 0.05 s 完成。需要判断这是链路问题
还是 `ncclCommSplit` 实现问题。

本探针**不走 split**：对每一对 rank 用单测同款的
`dist.new_group(ranks=[a,b], device_id=device)` 单独建组（NCCL comm 在
建组时立刻创建），然后组内做 batch_isend_irecv 双向收发，bf16 形状
(1, 256, 5120)，逐元素精确校验，并报告建组耗时和收发耗时。

判定：
  {0,8} / {1,9} 建组或收发失败/超时，其它对正常  -> 这两对之间的链路有问题
  所有对都正常                                     -> 链路没问题，问题在
                                                     ncclCommSplit（split 路径）

注意：这个探针不测“裸 isend/irecv 串行化”那个问题（那用
p2p_bare_probe.py）。这里所有收发都用 batch_isend_irecv 提交。

防挂死：主线程每完成一对 touch 心跳；超过 --hang-timeout 没有推进，
CommWatchdog 会调 ncclCommRevoke 撤掉本 rank 的 NCCL comm，等本机各 rank
静默后 os._exit(124)。全程不发信号、不 kill —— 也不要给这个脚本套 timeout
或从外面 kill，那正是会打出 XID 24 让 GPU 永久挂死的路径。

跑法（容器里，16 卡，和失败时的规模一致）：

    cd /data/tangruijing/udtca/bitscom
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 15) torchrun --nproc_per_node=16 \
        --master_port=29535 tests/pair_connectivity_probe.py \
        2>&1 | tee /data1/tangruijing/logs/pair_probe_$(hostname).log

    # 只测可疑的两对（其余 rank 仍要参与 new_group 的调用顺序）
    ... tests/pair_connectivity_probe.py --pairs 0-8,1-9

    # 对照：测所有 120 对（耗时长，一般不需要）
    ... tests/pair_connectivity_probe.py --pairs all

输出里每对一行，例如（rank a 打印）：
    [rank 0] pair 0-8  group=0.52s  n=10  avg=3.10ms  bw=1.66GB/s  PASS
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

SHAPE = (1, 256, 5120)
DTYPE = torch.bfloat16
BYTES = SHAPE[0] * SHAPE[1] * SHAPE[2] * 2  # bf16 = 2 bytes
# 进程组超时远大于 hang-timeout：先让 revoke 收尾，不让 torch 自己 abort
# 在途通信（Corex 上那是 XID 24 的路径）。
PG_TIMEOUT_S = 900

DEFAULT_PAIRS = "0-8,1-9,2-10,3-11,4-12,5-13,6-14,7-15"


def _log(rank: int, msg: str) -> None:
    print(f"[rank {rank}] {msg}", flush=True)


def _parse_pairs(spec: str, world: int) -> list[tuple[int, int]]:
    if spec.strip().lower() == "all":
        return list(itertools.combinations(range(world), 2))
    pairs: list[tuple[int, int]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        a, b = (int(v) for v in tok.split("-"))
        if not (0 <= a < world and 0 <= b < world) or a == b:
            raise ValueError(f"非法 pair {tok}（world_size={world}）")
        pairs.append((a, b))
    if not pairs:
        raise ValueError("--pairs 为空")
    return pairs


def _load_revoke():
    revoke_dir = os.environ.get(
        "UDTCA_REVOKE_DIR", "/data/tangruijing/udtca/experiments/qwen14b"
    )
    if revoke_dir not in sys.path:
        sys.path.insert(0, revoke_dir)
    try:
        import nccl_revoke  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: 加载 nccl_revoke 失败（{exc!r}），卡住时不会自动收尾", flush=True)
        return None
    return nccl_revoke


def _base() -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    return torch.randn(SHAPE, generator=gen, dtype=torch.float32)


def _payload(base: torch.Tensor, rank: int, it: int, device) -> torch.Tensor:
    return (base + float(rank) + 0.25 * it).to(DTYPE).to(device)


def _describe(buf: torch.Tensor) -> str:
    mn = float(buf.min())
    mx = float(buf.max())
    flat = [float(v) for v in buf.reshape(-1)[:8]]
    kind = "constant" if mn == mx else "varying"
    return f"min={mn:.6g} max={mx:.6g} first8={flat} ({kind})"


def _new_pair_group(a: int, b: int, device) -> "dist.ProcessGroup":
    kw = dict(ranks=[a, b], backend="nccl",
              timeout=timedelta(seconds=PG_TIMEOUT_S))
    try:
        return dist.new_group(device_id=device, **kw)
    except TypeError:  # 老版本 torch 没有 device_id
        return dist.new_group(**kw)


def _transfer(pg, rank: int, peer: int, base, it: int, device) -> float:
    local = _payload(base, rank, it, device)
    buf = torch.full(SHAPE, float("nan"), dtype=DTYPE, device=device)
    t0 = time.perf_counter()
    ops = [
        dist.P2POp(dist.isend, local, peer, group=pg),
        dist.P2POp(dist.irecv, buf, peer, group=pg),
    ]
    for work in dist.batch_isend_irecv(ops):
        work.wait()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    expected = _payload(base, peer, it, device)
    if not torch.equal(buf, expected):
        raise AssertionError(
            f"iter {it}: 从 rank {peer} 收到的内容不对 | {_describe(buf)}"
        )
    return dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", default=DEFAULT_PAIRS,
                    help=f"逗号分隔的 a-b 对，或 all；默认 {DEFAULT_PAIRS}")
    ap.add_argument("--iters", type=int, default=10, help="每对收发迭代次数")
    ap.add_argument("--hang-timeout", type=float, default=120.0,
                    help="多久没有推进就 revoke 退出（秒）")
    args = ap.parse_args()

    # 裸 torchrun 跑时没有启动器那套环境，这里补齐跑通和留证据所必需的
    os.environ.setdefault("NCCL_SYM_KERNEL", "0")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    log_dir = os.environ.get("UDTCA_LOG_DIR", "/data1/tangruijing/logs")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "ALL")
    os.environ.setdefault(
        "NCCL_DEBUG_FILE", os.path.join(log_dir, "nccl_pairprobe.%h.%p.log")
    )
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError:
        pass

    if "LOCAL_RANK" not in os.environ:
        print("ERROR: 必须用 torchrun 启动，见文件头部用法。", file=sys.stderr)
        return 2
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        print(f"ERROR: 至少 2 个进程，当前 WORLD_SIZE={world}", file=sys.stderr)
        return 2
    pairs = _parse_pairs(args.pairs, world)

    revoke = _load_revoke()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    try:
        dist.init_process_group(backend="nccl", init_method="env://",
                                timeout=timedelta(seconds=PG_TIMEOUT_S),
                                device_id=device)
    except TypeError:
        dist.init_process_group(backend="nccl", init_method="env://",
                                timeout=timedelta(seconds=PG_TIMEOUT_S))

    if rank == 0:
        print(
            f"=== 成对连通性探针 ===\n"
            f"host={os.uname().nodename} world={world} "
            f"torch={torch.__version__} "
            f"nccl={'.'.join(str(v) for v in torch.cuda.nccl.version())}\n"
            f"pairs={pairs} iters={args.iters} shape={SHAPE} dtype={DTYPE}\n"
            f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '-')} "
            f"NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', '-')} "
            f"NCCL_DEBUG_FILE={os.environ.get('NCCL_DEBUG_FILE', '-')}",
            flush=True,
        )

    watchdog = None
    if revoke is not None:
        watchdog = revoke.CommWatchdog(timeout_s=args.hang_timeout, exit_code=124)
        watchdog.capture(force=True)
        watchdog.touch()
        watchdog.start()

    base = _base()
    created: list[tuple[tuple[int, int], "dist.ProcessGroup"]] = []
    results: list[str] = []
    rc = 0
    try:
        for a, b in pairs:
            member = rank in (a, b)
            t0 = time.perf_counter()
            try:
                pg = _new_pair_group(a, b, device)
            except Exception as exc:  # noqa: BLE001
                rc = 1
                results.append(f"{a}-{b}=GROUP_FAIL({type(exc).__name__})")
                _log(rank, f"pair {a}-{b} 建组失败: {type(exc).__name__}: {exc}")
                if watchdog is not None:
                    watchdog.touch()
                continue
            dt_group = time.perf_counter() - t0
            created.append(((a, b), pg))
            if watchdog is not None:
                watchdog.touch()

            # 非成员也要走完全部的 new_group 调用顺序，但不能进组内收发；
            # 也不能跳过下面的 barrier —— barrier 是 world 集合操作，
            # 少一个 rank 整组就会挂住。
            if member:
                peer = b if rank == a else a
                try:
                    times = []
                    for it in range(args.iters):
                        times.append(_transfer(pg, rank, peer, base, it, device))
                        if watchdog is not None:
                            watchdog.touch()
                    avg = sum(times) / len(times)
                    bw = BYTES / avg / 1e9 if avg > 0 else float("inf")
                    results.append(f"{a}-{b}=PASS")
                    _log(rank, f"pair {a}-{b}  group={dt_group:.3f}s  n={args.iters}  "
                               f"avg={avg * 1e3:.2f}ms  bw={bw:.2f}GB/s  PASS")
                except Exception as exc:  # noqa: BLE001
                    rc = 1
                    results.append(f"{a}-{b}=FAIL({type(exc).__name__})")
                    _log(rank, f"pair {a}-{b}  group={dt_group:.3f}s  FAIL: "
                               f"{type(exc).__name__}: {exc}")
                if watchdog is not None:
                    watchdog.touch()  # 失败也要继续测下一对

            # 每对之间同步一次，保证所有 rank（含非成员）进度一致
            try:
                dist.barrier()
            except Exception:  # noqa: BLE001
                pass

        _log(rank, "SUMMARY " + " ".join(results))
    finally:
        if watchdog is not None:
            watchdog.stop()
        # 不再补一次 barrier：主循环每对之间已经同步过，某个 rank 异常时
        # 补 barrier 只会让其它 rank 二次挂住。
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        for (a, b), pg in reversed(created):
            if rank in (a, b):
                try:
                    dist.destroy_process_group(pg)
                except Exception:  # noqa: BLE001
                    pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
