#!/usr/bin/env python3
"""PP warm-up 投票卡死：最小复现阶梯（给天数定位用）。

现象（2026-10-27/28，powerone1，16 卡 DP2×PP8，容器 corex torch 2.13.0）：
  torch.distributed.pipelining 的 PP warm-up 投票，PP 组 nranks=8、int32
  count=1、阻塞 dist.send/dist.recv：
      rank i: recv(src=i-1) -> send(dst=i+1)        # _warmup_forward_vote
      rank i: recv(src=i+1) -> send(dst=i-1)        # _warmup_backward_result
  实际日志（两组同一次启动，同一份代码）：
      A 组(0-7):  0->1 完成, 1->2 完成, 3->2 完成, 2->1 永久不返回
                  2[2] -> 1[1] via P2P/indirect/3[3]/read
      B 组(8-15): 全部完成，5 轮 6.8s
                  2[10] -> 1[9] via P2P/indirect/0[8]/read

本脚本默认跑**阶梯**，从最简到最完整，一跑到底，看在哪一级先卡：

  pair     只有 2->1 这一跳（无模型、无其他 rank 参与），20 轮
  chain3   0..3 级的完整投票，3 当末端（无模型），20 轮
  chain7   0..7 完整 8 级投票（无模型），20 轮
  warmup   加载 14B 模型 + PolarParallel（含 73 次 DP 参数广播），然后调用
           torch 内部原函数 schedule._warmup_p2p([stage], True, False) 5 轮

每一级通过就打 `PHASE <name> PASS`，卡住就停在那里。**第一个卡住的级别就是
可以交给天数的最小复现脚本**；如果 pair 就卡，整件事只需要 torch + NCCL，
和训练代码、模型大小、DP 广播都无关。

也可以单独跑某一级（--phases pair / --phases chain3,warmup ...）。

内核日志（vmid / XID / mmu page fault / IO_PAGE_FAULT / segfault / GPU link / BUG 等）：
  每个节点由 local rank 0 自动收集，全部落在文件里，跑完直接在 stdout 看统计：
    <log_dir>/dmesg_events_<host>.log          后台每 10 s 抓一次的新事件（卡死时也有）
    <log_dir>/dmesg_<host>_start_<时间>.txt     开跑前基线（full + _filtered.txt）
    <log_dir>/dmesg_<host>_end_<phase>_rc<N>_<时间>.txt   收尾前（rc=0 通过，124 被 revoke）
    <log_dir>/dmesg_<host>_post_<phase>_rc<N>_<时间>.txt  拆组之后
  统计行形如：`[hop dmesg rank=0 ...] powerone: vmid=0 xid=1 page_fault=1 events=3 -> ...`
  开跑前基线和结束各一份，直接 diff 就知道这次运行期间有没有新增 vmid 泄漏 / page fault。

防挂死：CommWatchdog 默认 300 s 无心跳就 ncclCommRevoke 收尾 os._exit(124)。
**不要**从外面套 timeout、不要 kill、不要 Ctrl-C（XID 24 会让卡永久挂死）。

跑法（容器里，16 卡）：

  cd /data/tangruijing/udtca/bitscom && torchrun --nproc_per_node=16 --master_port=29641 tests/pp_vote_hop_repro.py 2>&1 | tee /data1/tangruijing/logs/pp_hop_ladder_$(date +%H%M%S).log

只想起最小复现（不加载模型）：

  cd /data/tangruijing/udtca/bitscom && torchrun --nproc_per_node=16 --master_port=29641 tests/pp_vote_hop_repro.py --phases pair 2>&1 | tee /data1/tangruijing/logs/pp_hop_pair_$(date +%H%M%S).log
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

# 内核日志里真正与本次卡死相关的几类事件。必须用精确短语，不能用裸子串：
#   * 裸 "fault" 会命中 "default"（大量无关日志）
#   * 裸 "mmu"  会命中 "communication"（"coMMUnication"，同样是无关日志）
#   * 裸 "reset" 会命中 megaraid 的 "Online Controller Reset"
# 已知的真实日志形态（2026-10-27/28 powerone1）：
#   iluvatar 0000:07:00.0: No enough vmid !
#   iluvatar 0000:67:00.0 XID: 24 mmu page fault!
#   pt_nccl_heartbt[1821643]: segfault at ... ip ... in libcuda.so.1[...]
DMESG_SIGNATURES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("vmid", re.compile(r"\bvmid\b", re.IGNORECASE)),
    ("xid", re.compile(r"\bXID\s*[:=]", re.IGNORECASE)),
    ("mmu_page_fault", re.compile(r"\bmmu\s+page\s+fault\b", re.IGNORECASE)),
    ("page_fault", re.compile(r"\bpage\s+fault\b", re.IGNORECASE)),
    ("segfault", re.compile(r"\bsegfault\b|\bsegmentation\s+fault\b",
                            re.IGNORECASE)),
    ("gpu_reset", re.compile(r"\biluvatar\b[^:\n]*\breset\b|"
                             r"\breset\b[^:\n]*\biluvatar\b", re.IGNORECASE)),
    # powerone2 的 P2P 缺陷在内核里是 IOMMU 报出来的（AMD-Vi IO_PAGE_FAULT），
    # 和 powerone1 的 iluvatar XID 24 是同一类页错误
    ("io_page_fault", re.compile(r"\bIO_PAGE_FAULT\b|\bAMD-Vi\b.*\b(?:fault|error|event)\b|"
                                 r"\bIOMMU\b.*\b(?:fault|error)\b", re.IGNORECASE)),
    ("gpu_link", re.compile(r"\biluvatar\b.*\blink\s+(?:down|up|state|training)\b|"
                            r"\blink\s+(?:down|up)\b.*\biluvatar\b", re.IGNORECASE)),
    ("fatal", re.compile(r"\bBUG\b|\bOops\b|kernel\s+panic|\bNMI\b|"
                         r"hung\s+task|soft\s+lockup|call\s+trace|"
                         r"blocked\s+for\s+more\s+than|"
                         r"\bgpu\s+(?:hang|stuck|timeout)\b",
                         re.IGNORECASE)),
)
# 明确无关、容易误报的行：存储控制器 reset、SATA/mlx5 网卡的 link down
DMESG_EXCLUDE = re.compile(
    r"online controller reset|megaraid|\bata\d*:\s|\bmlx5_core\b|\bscsi\b",
    re.IGNORECASE,
)

DEFAULT_ENTRY = (
    "/data/tangruijing/udtca/experiments/qwen14b/run_qwen14b_polar_dp_pp_tp.py"
)
PG_TIMEOUT_S = 21600


def _log(tag: str, msg: str) -> None:
    try:
        rank = dist.get_rank()
    except Exception:  # noqa: BLE001
        rank = -1
    print(f"[hop {tag} rank={rank} pid={os.getpid()} "
          f"t={time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_entry(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"找不到真实入口脚本: {path}")
    spec = importlib.util.spec_from_file_location("qwen14b_entry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["qwen14b_entry"] = module
    spec.loader.exec_module(module)
    return module


def _load_revoke():
    revoke_dir = os.environ.get(
        "UDTCA_REVOKE_DIR", "/data/tangruijing/udtca/experiments/qwen14b"
    )
    if revoke_dir not in sys.path:
        sys.path.insert(0, revoke_dir)
    import nccl_revoke  # noqa: PLC0415

    return nccl_revoke


def _read_dmesg() -> str | None:
    """读整份内核日志；容器里通常 dmesg 可用，不行再退到 /dev/kmsg。"""
    for cmd in (["dmesg", "-T"], ["dmesg"], ["cat", "/dev/kmsg"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except Exception:  # noqa: BLE001
            continue
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
    return None


def _classify_dmesg_line(line: str) -> list[str]:
    if DMESG_EXCLUDE.search(line):
        return []
    return [name for name, rx in DMESG_SIGNATURES if rx.search(line)]


def _dmesg_events(text: str) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        cats = _classify_dmesg_line(line)
        if cats:
            out.append((line, cats))
    return out


class DmesgTail(threading.Thread):
    """后台按 interval 轮询 dmesg，把新出现的 vmid/XID/page fault 行实时落盘。

    卡死时 CommWatchdog 会从另一个线程 os._exit，来不及做收尾统计；这份
    tail 文件保证那时候证据已经在盘上。
    """

    def __init__(self, path: str, interval_s: float = 10.0) -> None:
        super().__init__(name="udtca-dmesg-tail", daemon=True)
        self.path = path
        self.interval_s = interval_s
        self._seen: set[str] = set()
        self._stop = threading.Event()

    def _append_new(self) -> int:
        text = _read_dmesg()
        if text is None:
            return 0
        fresh = [(ln, cats) for ln, cats in _dmesg_events(text)
                 if ln not in self._seen]
        if not fresh:
            return 0
        self._seen.update(ln for ln, _ in fresh)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                for ln, cats in fresh:
                    f.write(f"[{time.strftime('%H:%M:%S')}] "
                            f"[{','.join(cats)}] {ln}\n")
                f.flush()
        except OSError:
            pass
        return len(fresh)

    def run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                n = self._append_new()
            except Exception:  # noqa: BLE001
                n = 0
            if n:
                _log("dmesg", f"{n} new event(s) -> {self.path}")

    def stop(self) -> None:
        self._stop.set()

    def final(self) -> None:
        self._stop.set()
        try:
            n = self._append_new()
        except Exception:  # noqa: BLE001
            n = 0
        if n:
            _log("dmesg", f"{n} new event(s) -> {self.path}")


def dump_dmesg_snapshot(log_dir: str, tag: str) -> None:
    """运行结束时把整份 dmesg 和过滤结果存盘，并在 stdout 打印匹配行。

    只在该 node 的 local rank 0 调用（每个节点一份，不重复写）。
    """
    host = socket.gethostname()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(log_dir, f"dmesg_{host}_{tag}_{stamp}")
    text = _read_dmesg()
    if text is None:
        _log("dmesg", "WARN 读不到 dmesg（容器无权限？），本次没有内核日志")
        return
    events = _dmesg_events(text)
    try:
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(text)
        with open(base + "_filtered.txt", "w", encoding="utf-8") as f:
            for ln, cats in events:
                f.write(f"[{','.join(cats)}] {ln}\n")
    except OSError as exc:
        _log("dmesg", f"WARN 写 dmesg 失败: {exc!r}")
        return

    counts = {name: sum(1 for _, cats in events if name in cats)
              for name, _ in DMESG_SIGNATURES}
    stats = " ".join(f"{k}={v}" for k, v in counts.items() if v or
                     k in ("vmid", "xid", "mmu_page_fault"))
    _log("dmesg", f"{host}: events={len(events)} {stats} "
                  f"-> {base}_filtered.txt")
    for ln, cats in events[-20:]:
        _log("dmesg", f"  [{','.join(cats)}] {ln}")


def _pick_socket_ifname() -> str | None:
    """选 bootstrap 网卡（容器是 host network，看到的就是主机网卡）。

    优先级：powerone1 训练约定的 ens120f0 → 默认路由所在网卡（powerone2 是
    enx00e04c680003，与它的训练脚本一致）→ 不设，交给 NCCL 自己选。
    """
    try:
        names = {name for _, name in socket.if_nameindex()}
    except OSError:
        names = set()
    if "ens120f0" in names:
        return "ens120f0"
    for cmd in (["ip", "route", "get", "1.1.1.1"],
                ["ip", "route", "show", "default"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except Exception:  # noqa: BLE001
            continue
        m = re.search(r"\bdev\s+(\S+)", r.stdout)
        if m and m.group(1) in names:
            return m.group(1)
    return None


def _zero(device) -> torch.Tensor:
    # stage.py 里就是 torch.zeros(1, dtype=torch.int32, device=self.device)
    return torch.zeros(1, dtype=torch.int32, device=device)


def _one(device) -> torch.Tensor:
    return torch.tensor([1], dtype=torch.int32, device=device)


def _recv(t, src_global, group) -> None:
    dist.recv(t, src=src_global, group=group)


def _send(t, dst_global, group) -> None:
    dist.send(t, dst=dst_global, group=group)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phases", default="pair,chain3,chain7,warmup",
                    help="逗号分隔，可选 pair / chain3 / chain7 / warmup")
    ap.add_argument("--entry", default=DEFAULT_ENTRY)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--tokenizer-name", default=None)
    ap.add_argument("--pp-size", type=int, default=8)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--micro-batches", type=int, default=16)
    ap.add_argument("--per-device-batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rounds", type=int, default=20, help="pair/chain 每级的轮数")
    ap.add_argument("--rounds-warmup", type=int, default=5)
    ap.add_argument("--src", type=int, default=2)
    ap.add_argument("--dst", type=int, default=1)
    ap.add_argument("--pre-forward", action=argparse.BooleanOptionalAction,
                    default=True, help="pair 先跑一次 dst->src 建链")
    ap.add_argument("--use-flash-attn", action=argparse.BooleanOptionalAction,
                    default=False)
    ap.add_argument("--init-from-pretrained", action="store_true", default=False)
    ap.add_argument("--hang-timeout", type=float,
                    default=float(os.environ.get("UDTCA_COMM_WATCHDOG_S", "300")))
    ap.add_argument("--no-watchdog", action="store_true")
    ap.add_argument("--no-dmesg", action="store_true",
                    help="关掉 dmesg 收集（不推荐）")
    ap.add_argument("--dmesg-interval", type=float, default=10.0,
                    help="后台轮询 dmesg 的间隔秒数，0=只在结束时抓")
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"[hop] 忽略未使用参数: {unknown}", flush=True)

    os.environ.setdefault("NCCL_SYM_KERNEL", "0")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    # 容器是 host network，直接看主机网卡；调用方显式导出了就尊重调用方
    if "NCCL_SOCKET_IFNAME" not in os.environ:
        iface = _pick_socket_ifname()
        if iface:
            os.environ["NCCL_SOCKET_IFNAME"] = iface
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    log_dir = os.environ.get("UDTCA_LOG_DIR", "/data1/tangruijing/logs")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "ALL")
    os.environ.setdefault(
        "NCCL_DEBUG_FILE", os.path.join(log_dir, "nccl_ppvoterepro.%h.%p.log")
    )
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError:
        pass

    print(f"[hop env] host={socket.gethostname()} "
          f"NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', 'auto')} "
          f"NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE', '-')} "
          f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '-')}", flush=True)

    if "LOCAL_RANK" not in os.environ:
        print("ERROR: 必须用 torchrun 启动，见文件头部用法。", file=sys.stderr)
        return 2

    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{local_rank}")
    model_parallel = args.pp_size * args.tp_size
    if world % model_parallel != 0:
        print(f"ERROR: world_size={world} 不能被 pp*tp={model_parallel} 整除",
              file=sys.stderr)
        return 2
    dp_size = world // model_parallel
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    from torch.distributed.device_mesh import init_device_mesh

    revoke = None if args.no_watchdog else _load_revoke()
    watchdog = None
    if revoke is not None:
        watchdog = revoke.CommWatchdog(timeout_s=args.hang_timeout, exit_code=124)
        watchdog.capture(force=True)
        watchdog.touch()
        watchdog.start()

    # 每个节点只由 local rank 0 收内核日志（vmid 泄漏 / XID / page fault）
    dmesg_tail = None
    dmesg_events_path = os.path.join(
        log_dir, f"dmesg_events_{socket.gethostname()}.log"
    )
    if not args.no_dmesg and local_rank == 0:
        dmesg_tail = DmesgTail(dmesg_events_path, args.dmesg_interval)
        if args.dmesg_interval > 0:
            dmesg_tail.start()
        _log("dmesg", f"tail file: {dmesg_events_path}")

    def beat(tag: str = "") -> None:
        if watchdog is not None:
            watchdog.touch()

    entry = None      # 只有 warmup 阶段才加载真实训练入口
    device_mesh = pp_group = None
    pp_rank = pp_size = 0
    trainer = None

    def build_mesh() -> None:
        nonlocal device_mesh, pp_group, pp_rank, pp_size
        if args.tp_size > 1:
            device_mesh = init_device_mesh(
                "cuda", (dp_size, args.pp_size, args.tp_size),
                mesh_dim_names=("dp", "pp", "tp"),
            )
        else:
            device_mesh = init_device_mesh(
                "cuda", (dp_size, args.pp_size), mesh_dim_names=("dp", "pp")
            )
        pp_group = device_mesh["pp"].get_group()
        pp_rank = device_mesh["pp"].get_local_rank()
        pp_size = dist.get_world_size(group=pp_group)

    def pp_global(r: int) -> int:
        return dist.get_global_rank(pp_group, r)

    def build_trainer() -> None:
        """与真实训练完全一致的初始化（模型 + PolarParallel 的 DP 广播）。"""
        nonlocal entry, trainer
        entry = _load_entry(args.entry)
        cfg = entry.TrainConfig(
            model_name=args.model_name, tokenizer_name=args.tokenizer_name,
            dataset_name_or_path="HuggingFaceFW/fineweb", dataset_config=None,
            text_field="text", seq_len=args.seq_len,
            per_device_batch_size=args.per_device_batch_size,
            grad_accum_steps=1, lr=args.lr, warmup_ratio=0.02, max_tokens=0,
            max_steps=1, weight_decay=0.1, beta1=0.9, beta2=0.95, clip_norm=1.0,
            log_interval=1, save_interval=1000, save_dir="/tmp/pp_hop_repro_ckpt",
            num_workers=0, use_flash_attn=args.use_flash_attn, bf16=True,
            fp16=False, activation_checkpointing=False,
            init_from_pretrained=args.init_from_pretrained,
            dataset_cache_path=None,
        )
        tokenizer = entry.build_tokenizer(cfg, local_files_only=True)
        entry._set_seed(args.seed)
        model = entry.build_qwen_model(cfg, local_files_only=True)
        stage_model = entry.partition_qwen_model(model, pp_rank, args.pp_size,
                                                 debug_nan_steps=0)
        _log("init", f"partitioned stage={pp_rank}")
        beat("partition")

        def loss_fn(output, target):
            return entry.vocab_parallel_lm_loss(
                output, target, ignore_index=tokenizer.pad_token_id,
                tp_mesh=device_mesh["tp"] if args.tp_size > 1 else None,
            )

        wrapper_args = SimpleNamespace(
            using_polar=False, method="none", polar_hook="none", dataset_config=None,
            init_from_pretrained=args.init_from_pretrained, lr=args.lr,
            train_mode="pp_hop_repro", local_sgd_mode="blocking",
            seq_len=args.seq_len,
        )
        _log("init", "PolarParallel init start（含 DP 参数广播）")
        trainer = entry.PolarParallel(
            args=wrapper_args, device_mesh=device_mesh,
            micro_batches=args.micro_batches, loss_fn=loss_fn,
            stage_model=stage_model, dataloader=None, comm_timing=-1,
            use_local_sgd=False, local_sgd_steps=1, baseline_mode="manual",
        )
        _log("init", "PolarParallel init done")
        beat("wrapper")

    def run_pair(n: int) -> None:
        src, dst = args.src, args.dst
        for r in range(n):
            if pp_rank == dst and args.pre_forward:
                t0 = time.perf_counter()
                _log("pair", f"round {r} pre-forward send {dst} -> {src}")
                _send(_one(device), pp_global(src), pp_group)
                _log("pair", f"round {r} pre-forward send done "
                             f"{time.perf_counter() - t0:.3f}s")
                beat(f"r{r}-pre")
            if pp_rank == src and args.pre_forward:
                t0 = time.perf_counter()
                _log("pair", f"round {r} pre-forward recv {src} <- {dst}")
                _recv(_zero(device), pp_global(dst), pp_group)
                _log("pair", f"round {r} pre-forward recv done "
                             f"{time.perf_counter() - t0:.3f}s")
                beat(f"r{r}-pre")
            if pp_rank == src:
                t0 = time.perf_counter()
                _log("pair", f"round {r} send {src} -> {dst}")
                _send(_one(device), pp_global(dst), pp_group)
                _log("pair", f"round {r} send done {time.perf_counter() - t0:.3f}s")
                beat(f"r{r}-pair")
            elif pp_rank == dst:
                t0 = time.perf_counter()
                _log("pair", f"round {r} recv {dst} <- {src}")
                _recv(_zero(device), pp_global(src), pp_group)
                _log("pair", f"round {r} recv done {time.perf_counter() - t0:.3f}s")
                beat(f"r{r}-pair")
            else:
                beat(f"r{r}-idle")

    def run_chain(last: int, n: int) -> None:
        last = min(last, pp_size - 1)
        for r in range(n):
            if pp_rank > last:
                beat(f"r{r}-idle")
                continue
            t_round = time.perf_counter()
            vote = _one(device)
            if pp_rank == 0:
                acc = vote
            else:
                acc = _zero(device)
                _log("chain", f"round {r} recv fwd <- {pp_rank - 1}")
                _recv(acc, pp_global(pp_rank - 1), pp_group)
                _log("chain", f"round {r} recv fwd done")
                acc = acc * vote
            beat(f"r{r}-fwd-recv")
            if pp_rank < last:
                _log("chain", f"round {r} send fwd -> {pp_rank + 1}")
                _send(acc, pp_global(pp_rank + 1), pp_group)
                _log("chain", f"round {r} send fwd done")
            beat(f"r{r}-fwd-send")
            if pp_rank == last:
                result = acc
            else:
                result = _zero(device)
                _log("chain", f"round {r} recv bwd <- {pp_rank + 1}")
                _recv(result, pp_global(pp_rank + 1), pp_group)
                _log("chain", f"round {r} recv bwd done")
            beat(f"r{r}-bwd-recv")
            if pp_rank > 0:
                _log("chain", f"round {r} send bwd -> {pp_rank - 1}")
                _send(result, pp_global(pp_rank - 1), pp_group)
                _log("chain", f"round {r} send bwd done")
            beat(f"r{r}-bwd-send")
            _log("chain", f"round {r} done {time.perf_counter() - t_round:.3f}s")

    def run_warmup(n: int) -> None:
        sched = trainer.schedule
        for r in range(n):
            t0 = time.perf_counter()
            _log("warmup", f"round {r} _warmup_p2p enter")
            sched._warmup_p2p([trainer.stage], True, False)  # schedules.py:431
            _log("warmup", f"round {r} _warmup_p2p done "
                           f"{time.perf_counter() - t0:.3f}s")
            beat(f"r{r}-warmup")

    rc = 0
    current = "init"
    t_start = time.perf_counter()
    try:
        torch.cuda.set_device(device)
        if not args.no_dmesg and local_rank == 0:
            dump_dmesg_snapshot(log_dir, "start")   # 开跑前的基线
        dist.init_process_group(
            backend="nccl", init_method="env://",
            timeout=timedelta(seconds=PG_TIMEOUT_S), device_id=device,
        )
        build_mesh()
        _log("init", f"pp_rank={pp_rank}/{pp_size} world={world} dp={dp_size} "
                     f"phases={phases}")
        beat("mesh")

        for ph in phases:
            current = ph
            _log("phase", f"{ph} START")
            t0 = time.perf_counter()
            if ph == "pair":
                run_pair(args.rounds)
            elif ph == "chain3":
                run_chain(3, args.rounds)
            elif ph == "chain7":
                run_chain(pp_size - 1, args.rounds)
            elif ph == "warmup":
                if trainer is None:
                    build_trainer()
                run_warmup(args.rounds_warmup)
            else:
                raise SystemExit(f"未知 phase: {ph}")
            _log("phase", f"{ph} PASS {time.perf_counter() - t0:.3f}s")
            beat(f"phase-{ph}")

        _log("phase", f"SUMMARY PASS phases={phases} "
                      f"total={time.perf_counter() - t_start:.1f}s")
    except Exception as exc:  # noqa: BLE001
        _log("phase", f"SUMMARY FAIL phase={current} "
                      f"{type(exc).__name__}: {exc}")
        rc = 1
    finally:
        if watchdog is not None:
            watchdog.stop()
        # 先收内核日志，再拆组：vmid/page fault 往往就出现在拆卸或卡死之后
        if dmesg_tail is not None:
            dmesg_tail.final()
        if not args.no_dmesg and local_rank == 0:
            dump_dmesg_snapshot(log_dir, f"end_{current}_rc{rc}")
        try:
            if entry is not None:
                entry.shutdown_distributed()
            else:
                dist.barrier()
                dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
        if dmesg_tail is not None:
            dmesg_tail.final()
            if not args.no_dmesg:
                try:
                    dump_dmesg_snapshot(log_dir, f"post_{current}_rc{rc}")
                except Exception:  # noqa: BLE001
                    pass
        try:
            if revoke is not None and hasattr(revoke, "exit_now"):
                revoke.exit_now(rc)
            else:
                os._exit(rc)
        except Exception:  # noqa: BLE001
            os._exit(rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
