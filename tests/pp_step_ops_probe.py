#!/usr/bin/env python3
"""把 Schedule1F1B.step() 里的通信按接口拆开，逐项复现（风格同 pair_connectivity_probe）.

torch：容器内 2.13.0（/usr/local/corex/lib64/python3/dist-packages/torch）。
源码位置（本镜像）：
  pipelining/schedules.py:_step_microbatches  (Schedule1F1B, line 1026)
  pipelining/schedules.py:_warmup_p2p         (line 341)
  pipelining/schedules.py:_batch_p2p          (line 623)
  pipelining/stage.py:_warmup_forward_vote    (line 1750)
  pipelining/stage.py:_warmup_backward_result (line 1791)
  pipelining/stage.py:_recv_meta/_send_meta   (line 1720/1736)
  pipelining/stage.py:get_fwd_recv/send_ops, get_bwd_recv/send_ops

schedule.step() 实际用到的通信，按出现顺序：

  # 阶段                接口                                    数据
  1 warm-up 投票        dist.recv / dist.send（阻塞）            int32 × 1
  2 元数据({0}→{N-1})   dist.recv_object_list / send_object_list  pickle 对象
    元数据({N-1}→{0})   （use_batch=True）
  3 前向激活            dist.irecv / dist.isend（异步 Work）      bf16 激活
     → _batch_p2p 对“全是 isend / 全是 irecv”的批次**不走**
       batch_isend_irecv，而是逐个立即提交（schedules.py:623）
  4 反向梯度            dist.irecv / dist.isend（同上）            bf16 梯度
  5 1B1F 融合           dist.batch_isend_irecv([isend→i+1,       激活+梯度
                         irecv←i+1])   ← 只有“混合批次”才真正 batch
  6 收尾                Work.wait()

本脚本按这 6 个接口各做一个 phase，再给一个 `loop` 把 1F1B 的**调用顺序**
完整走一遍（不做任何计算、不建模型），每步打时间戳：

  vote  meta  act  grad  mixed  loop

判定：哪个 phase 卡住，就是那个接口/那条路径在 Corex 上有问题；
`vote` 卡但 `act/grad/mixed` 不卡 → 问题在阻塞式小包 P2P；
`loop` 卡但各自都过 → 问题在 op 的**组合/顺序**（流/信令配对）。

相对 torch 源码的唯一差别：`_batch_p2p` 的同构快路径里用的是
`group_dst/group_src`（组内 rank），这里用等价的 `dst/src`（全局 rank）
调用同一个 `dist.isend/irecv`。

防挂死：CommWatchdog 默认 300 s 无心跳就 ncclCommRevoke 收尾 os._exit(124)；
内核日志（vmid/XID/page fault）自动落盘，规则与 pp_vote_hop_repro.py 相同。
**不要**从外面套 timeout、不要 kill、不要 Ctrl-C。

跑法（容器里，16 卡；单机即可，--pp-size 8 时 dp=2）：

  cd /data/tangruijing/udtca/bitscom && torchrun --nproc_per_node=16 --master_port=29661 tests/pp_step_ops_probe.py 2>&1 | tee /data1/tangruijing/logs/pp_ops_$(date +%H%M%S).log

  # 只测某一项
  ... tests/pp_step_ops_probe.py --phases vote
  # 加 --sync 可对照“每步 synchronize”的行为（torch 本身没有）
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import timedelta

import torch
import torch.distributed as dist

PG_TIMEOUT_S = 21600

# ---------------- 内核日志（自包含，不依赖项目里其它脚本） ----------------
# 必须用精确短语：裸 "fault" 会命中 "default"，裸 "mmu" 会命中 "communication"
DMESG_SIGNATURES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("vmid", re.compile(r"\bvmid\b", re.IGNORECASE)),
    ("xid", re.compile(r"\bXID\s*[:=]", re.IGNORECASE)),
    ("mmu_page_fault", re.compile(r"\bmmu\s+page\s+fault\b", re.IGNORECASE)),
    ("page_fault", re.compile(r"\bpage\s+fault\b", re.IGNORECASE)),
    ("segfault", re.compile(r"\bsegfault\b|\bsegmentation\s+fault\b",
                            re.IGNORECASE)),
    # powerone2 的 P2P 缺陷是 IOMMU 报的
    ("io_page_fault", re.compile(r"\bIO_PAGE_FAULT\b|"
                                 r"\bAMD-Vi\b.*\b(?:fault|error|event)\b|"
                                 r"\bIOMMU\b.*\b(?:fault|error)\b",
                                 re.IGNORECASE)),
    ("gpu_reset", re.compile(r"\biluvatar\b[^:\n]*\breset\b|"
                             r"\breset\b[^:\n]*\biluvatar\b", re.IGNORECASE)),
    ("gpu_link", re.compile(r"\biluvatar\b.*\blink\s+(?:down|up|state|training)\b|"
                            r"\blink\s+(?:down|up)\b.*\biluvatar\b",
                            re.IGNORECASE)),
    ("fatal", re.compile(r"\bBUG\b|\bOops\b|kernel\s+panic|\bNMI\b|"
                         r"hung\s+task|soft\s+lockup|call\s+trace|"
                         r"blocked\s+for\s+more\s+than|"
                         r"\bgpu\s+(?:hang|stuck|timeout)\b", re.IGNORECASE)),
)
DMESG_EXCLUDE = re.compile(
    r"online controller reset|megaraid|\bata\d*:\s|\bmlx5_core\b|\bscsi\b",
    re.IGNORECASE,
)


def _read_dmesg() -> str | None:
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
    """后台轮询 dmesg，把新的 vmid/XID/page fault 行实时落盘（卡死时也有证据）。"""

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
                _out("dmesg", f"{n} new event(s) -> {self.path}", -1)

    def final(self) -> None:
        self._stop.set()
        try:
            self._append_new()
        except Exception:  # noqa: BLE001
            pass


def dump_dmesg_snapshot(log_dir: str, tag: str) -> None:
    host = socket.gethostname()
    base = os.path.join(log_dir,
                        f"dmesg_{host}_{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    text = _read_dmesg()
    if text is None:
        _out("dmesg", "WARN 读不到 dmesg（容器无权限？）", -1)
        return
    events = _dmesg_events(text)
    try:
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(text)
        with open(base + "_filtered.txt", "w", encoding="utf-8") as f:
            for ln, cats in events:
                f.write(f"[{','.join(cats)}] {ln}\n")
    except OSError as exc:
        _out("dmesg", f"WARN 写 dmesg 失败: {exc!r}", -1)
        return
    counts = {name: sum(1 for _, cats in events if name in cats)
              for name, _ in DMESG_SIGNATURES}
    stats = " ".join(f"{k}={v}" for k, v in counts.items()
                     if v or k in ("vmid", "xid", "mmu_page_fault"))
    _out("dmesg", f"{host}: events={len(events)} {stats} "
                  f"-> {base}_filtered.txt", -1)
    for ln, cats in events[-20:]:
        _out("dmesg", f"  [{','.join(cats)}] {ln}", -1)


def _pick_socket_ifname() -> str | None:
    """powerone1 约定 ens120f0 → 否则默认路由网卡（powerone2 = enx00e04c680003）。"""
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


def _load_revoke():
    """唯一的可选外部依赖：防挂死用的 nccl_revoke（没有就只警告，不影响测试）。"""
    revoke_dir = os.environ.get(
        "UDTCA_REVOKE_DIR", "/data/tangruijing/udtca/experiments/qwen14b")
    if revoke_dir not in sys.path:
        sys.path.insert(0, revoke_dir)
    try:
        import nccl_revoke  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: 加载 nccl_revoke 失败（{exc!r}），卡住时不会自动收尾",
              flush=True)
        return None
    return nccl_revoke


def _out(tag: str, msg: str, pp_rank: int) -> None:
    try:
        rank = dist.get_rank()
    except Exception:  # noqa: BLE001
        rank = -1
    print(f"[ops {tag} rank={rank} pp={pp_rank} t={time.strftime('%H:%M:%S')}] "
          f"{msg}", flush=True)


class Probe:
    def __init__(self, args, device, group, pp_rank, pp_size, beat) -> None:
        self.args = args
        self.device = device
        self.group = group
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.last = pp_size - 1
        self.beat = beat
        # 激活/梯度的真实形状：per_device_batch × seq_len × hidden，bf16
        n = args.batch * args.seq_len * args.hidden
        self.fwd_in = torch.zeros(n, dtype=torch.bfloat16, device=device)
        self.fwd_out = torch.full((n,), float(pp_rank), dtype=torch.bfloat16,
                                  device=device)
        self.bwd_in = torch.zeros(n, dtype=torch.bfloat16, device=device)
        self.bwd_out = torch.full((n,), float(pp_rank) + 0.5,
                                  dtype=torch.bfloat16, device=device)

    # ---- 邻居：组内 → 全局（等价 stage._resolve_peer_global_rank）----
    def _g(self, stage_idx: int) -> int:
        return dist.get_global_rank(self.group, stage_idx)

    def _sync(self) -> None:
        if self.args.sync:
            torch.cuda.synchronize(self.device)

    def _wait(self, work) -> None:
        for w in work:
            w.wait()
        self._sync()

    @staticmethod
    def _fire(p2p_ops: list) -> list:
        """等价 schedules.py:623 `_batch_p2p`。"""
        if not p2p_ops:
            return []
        kinds = {p.op for p in p2p_ops}
        if kinds == {dist.isend}:
            return [p.op(p.tensor, dst=p.peer, group=p.group, tag=p.tag)
                    for p in p2p_ops]
        if kinds == {dist.irecv}:
            return [p.op(p.tensor, src=p.peer, group=p.group, tag=p.tag)
                    for p in p2p_ops]
        return dist.batch_isend_irecv(p2p_ops)

    def _fwd_recv_ops(self) -> list:
        if self.pp_rank == 0:
            return []
        return [dist.P2POp(dist.irecv, self.fwd_in, self._g(self.pp_rank - 1),
                           self.group)]

    def _fwd_send_ops(self) -> list:
        if self.pp_rank >= self.last:
            return []
        return [dist.P2POp(dist.isend, self.fwd_out, self._g(self.pp_rank + 1),
                           self.group)]

    def _bwd_recv_ops(self) -> list:
        if self.pp_rank >= self.last:
            return []
        return [dist.P2POp(dist.irecv, self.bwd_in, self._g(self.pp_rank + 1),
                           self.group)]

    def _bwd_send_ops(self) -> list:
        if self.pp_rank == 0:
            return []
        return [dist.P2POp(dist.isend, self.bwd_out, self._g(self.pp_rank - 1),
                           self.group)]

    # ================= phase 1: warm-up 投票（阻塞 send/recv）=================
    def vote(self, rounds: int) -> None:
        vote = torch.ones(1, dtype=torch.int32, device=self.device)
        for r in range(rounds):
            t0 = time.perf_counter()
            if self.pp_rank > 0:
                acc = torch.zeros(1, dtype=torch.int32, device=self.device)
                _out("vote", f"round {r} recv fwd <- {self.pp_rank - 1}", self.pp_rank)
                dist.recv(acc, src=self._g(self.pp_rank - 1), group=self.group)
                acc = acc * vote
            else:
                acc = vote
            self.beat("vote-fwd")
            if self.pp_rank < self.last:
                _out("vote", f"round {r} send fwd -> {self.pp_rank + 1}", self.pp_rank)
                dist.send(acc, dst=self._g(self.pp_rank + 1), group=self.group)
            self.beat("vote-fwd-send")

            if self.pp_rank < self.last:
                result = torch.zeros(1, dtype=torch.int32, device=self.device)
                _out("vote", f"round {r} recv bwd <- {self.pp_rank + 1}", self.pp_rank)
                dist.recv(result, src=self._g(self.pp_rank + 1), group=self.group)
            else:
                result = acc
            self.beat("vote-bwd")
            if self.pp_rank > 0:
                _out("vote", f"round {r} send bwd -> {self.pp_rank - 1}", self.pp_rank)
                dist.send(result, dst=self._g(self.pp_rank - 1), group=self.group)
            self.beat("vote-bwd-send")
            _out("vote", f"round {r} done {time.perf_counter() - t0:.3f}s",
                 self.pp_rank)

    # ================= phase 2: 元数据（object list）=================
    def meta(self, rounds: int) -> None:
        fwd = {"stage": self.pp_rank, "kind": "fwd",
               "shape": [self.args.batch, self.args.seq_len, self.args.hidden],
               "dtype": "bfloat16"}
        bwd = {"stage": self.pp_rank, "kind": "bwd"}
        for r in range(rounds):
            t0 = time.perf_counter()
            if self.pp_rank > 0:
                objs: list = [None]
                _out("meta", f"round {r} recv_object_list fwd <- {self.pp_rank - 1}",
                     self.pp_rank)
                dist.recv_object_list(objs, src=self._g(self.pp_rank - 1),
                                      group=self.group, device=self.device,
                                      use_batch=True)
                self.beat("meta-fwd")
            if self.pp_rank < self.last:
                _out("meta", f"round {r} send_object_list fwd -> {self.pp_rank + 1}",
                     self.pp_rank)
                dist.send_object_list([fwd], dst=self._g(self.pp_rank + 1),
                                      group=self.group, device=self.device,
                                      use_batch=True)
                self.beat("meta-fwd-send")

            if self.pp_rank < self.last:
                objs = [None]
                _out("meta", f"round {r} recv_object_list bwd <- {self.pp_rank + 1}",
                     self.pp_rank)
                dist.recv_object_list(objs, src=self._g(self.pp_rank + 1),
                                      group=self.group, device=self.device,
                                      use_batch=True)
                self.beat("meta-bwd")
            if self.pp_rank > 0:
                _out("meta", f"round {r} send_object_list bwd -> {self.pp_rank - 1}",
                     self.pp_rank)
                dist.send_object_list([bwd], dst=self._g(self.pp_rank - 1),
                                      group=self.group, device=self.device,
                                      use_batch=True)
                self.beat("meta-bwd-send")
            _out("meta", f"round {r} done {time.perf_counter() - t0:.3f}s",
                 self.pp_rank)

    # ================= phase 3: 前向激活（irecv/isend）=================
    def act(self, mbs: int) -> None:
        # 顺序同 Schedule1F1B 的 warm-up 段：recv → wait → send(不 wait) → 下轮先 wait
        pending: list = []
        for m in range(mbs):
            t0 = time.perf_counter()
            _out("act", f"mb {m} recv(if any)", self.pp_rank)
            self._wait(self._fire(self._fwd_recv_ops()))
            self.beat("act-recv")
            self._wait(pending)
            pending = self._fire(self._fwd_send_ops())
            self.beat("act-send")
            _out("act", f"mb {m} done {time.perf_counter() - t0:.3f}s", self.pp_rank)
        self._wait(pending)

    # ================= phase 4: 反向梯度（irecv/isend）=================
    def grad(self, mbs: int) -> None:
        pending: list = []
        for m in range(mbs):
            t0 = time.perf_counter()
            _out("grad", f"mb {m} recv(if any)", self.pp_rank)
            self._wait(self._fire(self._bwd_recv_ops()))
            self.beat("grad-recv")
            self._wait(pending)
            pending = self._fire(self._bwd_send_ops())
            self.beat("grad-send")
            _out("grad", f"mb {m} done {time.perf_counter() - t0:.3f}s", self.pp_rank)
        self._wait(pending)

    # ================= phase 5: 1B1F 融合（batch_isend_irecv）=================
    def mixed(self, rounds: int) -> None:
        # 1F1B 稳态里唯一真正走 batch 的路径：fwd_send → i+1 与 bwd_recv ← i+1
        if not (self._fwd_send_ops() and self._bwd_recv_ops()):
            _out("mixed", "边界 stage，无混合 op", self.pp_rank)
            return
        for r in range(rounds):
            t0 = time.perf_counter()
            ops = self._fwd_send_ops() + self._bwd_recv_ops()
            _out("mixed", f"round {r} batch_isend_irecv(isend,irecv) <-> "
                          f"{self.pp_rank + 1}", self.pp_rank)
            self._wait(self._fire(ops))
            self.beat("mixed")
            _out("mixed", f"round {r} done {time.perf_counter() - t0:.3f}s",
                 self.pp_rank)

    # ================= phase 6: 完整 1F1B op 顺序（无计算、无模型）=================
    def loop(self, mbs: int) -> None:
        warmup = min(mbs, self.pp_size - self.pp_rank)
        fwd_idx = bwd_idx = 0
        send_work: list = []
        fwd_sends: list = []
        t_loop = time.perf_counter()

        for _ in range(warmup):
            t0 = time.perf_counter()
            self._wait(self._fire(self._fwd_recv_ops()))
            self.beat("loop-w-recv")
            self._wait(send_work)
            send_work = []
            fwd_sends = self._fwd_send_ops()
            if fwd_idx != warmup - 1:
                send_work = self._fire(fwd_sends)
                fwd_sends = []
            fwd_idx += 1
            self.beat("loop-w-send")
            _out("loop", f"warmup fwd {fwd_idx - 1} "
                         f"{time.perf_counter() - t0:.3f}s", self.pp_rank)

        while True:
            t0 = time.perf_counter()
            self._wait(self._fire(fwd_sends + self._bwd_recv_ops()))
            fwd_sends = []
            self.beat("loop-1b1f-recv")
            bwd_sends = self._bwd_send_ops()
            bwd_idx += 1
            if fwd_idx == mbs:
                break
            self._wait(self._fire(bwd_sends + self._fwd_recv_ops()))
            self.beat("loop-1b1f-fwd")
            fwd_sends = self._fwd_send_ops()
            fwd_idx += 1
            _out("loop", f"1b1f f={fwd_idx - 1} b={bwd_idx - 1} "
                         f"{time.perf_counter() - t0:.3f}s", self.pp_rank)

        send_work = self._fire(bwd_sends)
        while bwd_idx < mbs:
            t0 = time.perf_counter()
            self._wait(self._fire(self._bwd_recv_ops()))
            self._wait(send_work)
            send_work = self._fire(self._bwd_send_ops())
            bwd_idx += 1
            self.beat("loop-cool")
            _out("loop", f"cooldown b {bwd_idx - 1} "
                         f"{time.perf_counter() - t0:.3f}s", self.pp_rank)
        self._wait(send_work)
        _out("loop", f"done {time.perf_counter() - t_loop:.3f}s", self.pp_rank)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phases", default="vote,meta,act,grad,mixed,loop")
    ap.add_argument("--pp-size", type=int, default=8)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=5, help="vote/meta/mixed 轮数")
    ap.add_argument("--micro-batches", type=int, default=16,
                    help="act/grad/loop 的 micro-batch 数")
    ap.add_argument("--batch", type=int, default=16, help="per-device batch")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=5120, help="Qwen2.5-14B hidden")
    ap.add_argument("--sync", action="store_true",
                    help="每个 op 后 torch.cuda.synchronize（torch 本身没有）")
    ap.add_argument("--hang-timeout", type=float,
                    default=float(os.environ.get("UDTCA_COMM_WATCHDOG_S", "300")))
    ap.add_argument("--no-watchdog", action="store_true")
    ap.add_argument("--no-dmesg", action="store_true")
    ap.add_argument("--dmesg-interval", type=float, default=10.0)
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"[ops] 忽略未使用参数: {unknown}", flush=True)

    os.environ.setdefault("NCCL_SYM_KERNEL", "0")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    if "NCCL_SOCKET_IFNAME" not in os.environ:
        iface = _pick_socket_ifname()
        if iface:
            os.environ["NCCL_SOCKET_IFNAME"] = iface
    log_dir = os.environ.get("UDTCA_LOG_DIR", "/data1/tangruijing/logs")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "ALL")
    os.environ.setdefault("NCCL_DEBUG_FILE",
                          os.path.join(log_dir, "nccl_ppops.%h.%p.log"))
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError:
        pass

    if "LOCAL_RANK" not in os.environ:
        print("ERROR: 必须用 torchrun 启动，见文件头部用法。", file=sys.stderr)
        return 2
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world % args.pp_size != 0:
        print(f"ERROR: world={world} 不能被 pp={args.pp_size} 整除",
              file=sys.stderr)
        return 2
    dp_size = world // args.pp_size
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    device = torch.device(f"cuda:{local_rank}")

    from torch.distributed.device_mesh import init_device_mesh

    print(f"[ops env] host={socket.gethostname()} "
          f"NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', 'auto')} "
          f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '-')} "
          f"sync={args.sync} phases={phases}", flush=True)

    revoke = None if args.no_watchdog else _load_revoke()
    watchdog = None
    if revoke is not None:
        watchdog = revoke.CommWatchdog(timeout_s=args.hang_timeout, exit_code=124)
        watchdog.capture(force=True)
        watchdog.touch()
        watchdog.start()

    dmesg_tail = None
    if not args.no_dmesg and local_rank == 0:
        dmesg_tail = DmesgTail(os.path.join(
            log_dir, f"dmesg_events_{socket.gethostname()}.log"),
            args.dmesg_interval)
        if args.dmesg_interval > 0:
            dmesg_tail.start()

    def beat(tag: str = "") -> None:
        if watchdog is not None:
            watchdog.touch()

    rc = 0
    current = "init"
    pp_rank = -1
    t_start = time.perf_counter()
    try:
        torch.cuda.set_device(device)
        if not args.no_dmesg and local_rank == 0:
            dump_dmesg_snapshot(log_dir, "start")
        dist.init_process_group(backend="nccl", init_method="env://",
                                timeout=timedelta(seconds=PG_TIMEOUT_S),
                                device_id=device)
        if args.tp_size > 1:
            mesh = init_device_mesh("cuda", (dp_size, args.pp_size, args.tp_size),
                                    mesh_dim_names=("dp", "pp", "tp"))
        else:
            mesh = init_device_mesh("cuda", (dp_size, args.pp_size),
                                    mesh_dim_names=("dp", "pp"))
        pp_group = mesh["pp"].get_group()
        pp_rank = mesh["pp"].get_local_rank()
        pp_size = dist.get_world_size(group=pp_group)
        _out("init", f"pp_rank={pp_rank}/{pp_size} world={world} dp={dp_size}",
             pp_rank)
        beat("mesh")

        probe = Probe(args, device, pp_group, pp_rank, pp_size, beat)
        for ph in phases:
            current = ph
            t0 = time.perf_counter()
            _out("phase", f"{ph} START", pp_rank)
            if ph == "vote":
                probe.vote(args.rounds)
            elif ph == "meta":
                probe.meta(args.rounds)
            elif ph == "act":
                probe.act(args.micro_batches)
            elif ph == "grad":
                probe.grad(args.micro_batches)
            elif ph == "mixed":
                probe.mixed(args.rounds)
            elif ph == "loop":
                probe.loop(args.micro_batches)
            else:
                raise SystemExit(f"未知 phase: {ph}")
            _out("phase", f"{ph} PASS {time.perf_counter() - t0:.3f}s", pp_rank)
            beat(f"phase-{ph}")

        _out("phase", f"SUMMARY PASS phases={phases} "
                      f"total={time.perf_counter() - t_start:.1f}s", pp_rank)
    except Exception as exc:  # noqa: BLE001
        _out("phase", f"SUMMARY FAIL phase={current} "
                      f"{type(exc).__name__}: {exc}", pp_rank)
        rc = 1
    finally:
        # 注意：watchdog 一直留到最后；拆组本身也可能挂
        if dmesg_tail is not None:
            dmesg_tail.final()
        if not args.no_dmesg and local_rank == 0:
            try:
                dump_dmesg_snapshot(log_dir, f"end_{current}_rc{rc}")
            except Exception:  # noqa: BLE001
                pass
        try:
            if rc == 0:
                dist.barrier()
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
        if dmesg_tail is not None:
            dmesg_tail.final()
            if not args.no_dmesg and local_rank == 0:
                try:
                    dump_dmesg_snapshot(log_dir, f"post_{current}_rc{rc}")
                except Exception:  # noqa: BLE001
                    pass
        if watchdog is not None:
            watchdog.stop()
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
