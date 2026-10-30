#!/usr/bin/env python3
"""按 qwen14b 训练入口的顺序，一步步做初始化，每一步打一个检查点。

对应 experiments/qwen14b/run_qwen14b_polar_dp_pp_tp.py 里的真实步骤：

    1) bitscom.init(bitwidth)                      入口 main() ~1377
    2) torch.cuda.set_device(local_rank)           入口 ~1389
    3) dist.init_process_group(nccl, device_id)    入口 ~1391
    4) init_device_mesh("cuda", (dp, pp[, tp]))    入口 ~1421
    5) create_lowbit_dp_group(dp_mesh.get_group()) 入口 ~1440（--method bitscom）
    6) import 训练入口模块（含 datasets 占位兜底）    --with-data / --with-model
    7) token 缓存 + tokenizer + dataloader          入口 ~1447 / ~1474 / ~1523（--with-data）
    8) build_qwen_model + partition_qwen_model      入口 ~1479 / ~1482（--with-model）
    9) LowBitGroup 构造                             入口 ~1497（--method bitscom）
   10) allow_tf32 + revoke-probe                    入口 ~1445 / ~1558
   11) PolarParallel 构造 + train 循环入口           入口 ~1536 / ~1567（--with-trainer）

第 11 步的 train() 完整跑完第一个 schedule.step（含前向+反向+PP 通信）
并打印前后快照；之后的 DP 梯度同步、optimizer.step 照常进行。

每步在**每个 rank** 上打印：阶段名、距启动耗时、显存占用，然后
cuda.synchronize() + dist.barrier()，因此：
  - 所有 rank 都停在第 N 步之后 = 第 N+1 步是卡住的地方；
  - 只有部分 rank 停住 = 在对应 rank 的日志里直接看。

默认不加载模型、不碰数据集：4 卡（DP=2 × PP=2）就能跑完整个分布式初始化。
模型 14B 需要 --with-model 才构建/切分；PolarParallel 需要再加 --with-trainer。
第 11 步的数据是合成的（不依赖 datasets/tokenizer）。

跑法（容器里，仓库根目录的 bitscom 下；4 卡先用 2,3,4,5）：

    cd /data/tangruijing/udtca/bitscom

    # 4 卡，纯分布式初始化（DP2 × PP2），不建 lowbit 组
    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 \\
        --master_port=29551 tests/train_init_steps.py --dp 2 --pp 2

    # 4 卡，带上训练里的 lowbit DP 组（DP 组 rank 对是 {0,1} 和 {2,3}）
    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 \\
        --master_port=29552 tests/train_init_steps.py --dp 2 --pp 2 \\
        --method bitscom --bitwidth 2

    # 8 卡：初始化 + 模型 + PolarParallel，停在 schedule.step 之前
    torchrun --nproc_per_node=8 --master_port=29553 \\
        tests/train_init_steps.py --dp 2 --pp 4 --method bitscom \\
        --with-model --with-trainer

    # 4 卡版本（模型能切到 4 卡时）
    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 \\
        --master_port=29553 tests/train_init_steps.py --dp 2 --pp 2 \\
        --method bitscom --with-model --with-trainer

    # 数据链路（需要能 import datasets 的解释器，镜像自带的 numpy 1.26 不行）
    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 \\
        --master_port=29554 tests/train_init_steps.py --dp 2 --pp 2 \\
        --method bitscom --with-data --max-steps 1

注意：
  - NCCL 全量日志默认写到 $UDTCA_LOG_DIR（默认 /data1/tangruijing/logs）下的
    nccl_initsteps.<host>.<pid>.log；目录里没有新文件 = 进程从来没进过 NCCL。
  - 默认挂 nvcc/revoke 看门狗（--hang-timeout，默认 180s，0 关闭）：卡住时
    只做 ncclCommRevoke 然后自退，不发信号、不 kill。**不要**从外面套 timeout。
  - 退出走训练入口同款顺序：barrier -> 各子组 -> world。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "polar-sgd" / "src", REPO_ROOT / "bitscom" / "python"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_T0 = time.monotonic()
# set_device 之前不碰 CUDA：torch.cuda.memory_allocated() 会初始化当前设备
# （那时是 0 号卡），把进程的 CUDA 上下文钉到 0 号卡上。
_CUDA_READY = False


# ---------------------------------------------------------------- 检查点

def _mem_gib():
    if not _CUDA_READY:
        return (float("nan"), float("nan"))
    try:
        return (
            torch.cuda.memory_allocated() / 2**30,
            torch.cuda.memory_reserved() / 2**30,
        )
    except Exception:  # noqa: BLE001
        return (float("nan"), float("nan"))


def stage(rank: int, name: str, detail: str = "") -> None:
    alloc, reserved = _mem_gib()
    print(
        f"[rank {rank}] [stage +{time.monotonic() - _T0:7.2f}s] {name}"
        f"  mem_alloc={alloc:.2f}GiB mem_reserved={reserved:.2f}GiB"
        + (f"  {detail}" if detail else ""),
        flush=True,
    )


def checkpoint(rank: int, name: str, detail: str = "") -> None:
    """打印检查点，然后 synchronize + barrier：所有 rank 都过这一步才继续。"""
    stage(rank, name, detail)
    torch.cuda.synchronize()
    dist.barrier()


# ---------------------------------------------------------------- 入口同款工具

def create_lowbit_dp_group(dp_group):
    """与入口 create_lowbit_dp_group 相同：所有 rank 按同样的顺序建 lowbit 组。"""
    my_ranks = tuple(int(r) for r in dist.get_process_group_ranks(dp_group))
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, my_ranks)
    dp_rank_groups = sorted({tuple(int(r) for r in ranks) for ranks in gathered})

    _watchdog_s = float(os.environ.get("UDTCA_COMM_WATCHDOG_S", "600"))
    _pg_timeout = timedelta(
        seconds=_watchdog_s + float(os.environ.get("UDTCA_PG_TIMEOUT_MARGIN_S", "21600"))
    )
    selected = None
    print(f"[rank {dist.get_rank()}] lowbit DP 组划分: {dp_rank_groups}", flush=True)
    for ranks in dp_rank_groups:
        group = dist.new_group(ranks=list(ranks), backend="lowbit", timeout=_pg_timeout)
        print(f"[rank {dist.get_rank()}] new_group(lowbit) ranks={ranks} done", flush=True)
        if ranks == my_ranks:
            selected = group
    if selected is None:
        raise RuntimeError(f"failed to create lowbit DP group for ranks={my_ranks}")
    return selected


def _build_cfg(entry, args, cache_path=None):
    """与入口 main() 里构造 TrainConfig 的字段保持一致（只填初始化需要的）。"""
    return entry.TrainConfig(
        model_name=args.model_name,
        tokenizer_name=None,
        dataset_name_or_path=args.dataset_name_or_path,
        dataset_config=args.dataset_config,
        seq_len=args.seq_len,
        per_device_batch_size=args.per_device_batch_size,
        max_steps=args.max_steps,
        num_workers=0,
        use_flash_attn=False,
        dataset_cache_path=str(cache_path) if cache_path else None,
    )


def _make_cache_args(args):
    """入口 _prepare_node_local_hf_cache(args) 需要的字段子集。"""
    from argparse import Namespace

    return Namespace(
        dataset_cache_dir=args.dataset_cache_dir,
        dataset_cache_samples=max(
            args.max_steps * args.per_device_batch_size,
            args.per_device_batch_size,
        ),
        dataset_name_or_path=args.dataset_name_or_path,
        dataset_config=args.dataset_config,
        data_files=None,
        text_field="text",
        seq_len=args.seq_len,
        model_name=args.model_name,
        tokenizer_name=None,
        hf_local_files_only=False,
        refresh_dataset_cache=False,
        max_steps=args.max_steps,
        per_device_batch_size=args.per_device_batch_size,
    )


def _trainer_args(args, *, method: str):
    """PolarParallel 读的属性子集（默认值与入口 launcher 的 METHOD_ARGS 对齐）。

    入口的 argparse 写在 main() 里，没法单独复用；这里把 PolarParallel /
    hooks / profiler 真正读到的字段补齐，其余行为全部走 wrapper 里的默认值。
    """
    from argparse import Namespace

    return Namespace(
        using_polar=True,
        polar_hook=args.polar_hook,
        polar_beta=0.9,
        method=method,
        bitwidth=args.bitwidth,
        polar_bucket_numel=64_000_000,
        polar_max_inflight_buckets=1,
        comm_timing=args.comm_timing,
        micro_batches=args.micro_batches,
        max_steps=args.max_steps,
        grad_clip_norm=1.0,
        debug_nan_steps=0,
        init_from_pretrained=False,
        train_mode="init-steps",
        run_label="train_init_steps",
        dataset_config=None,
        dataset_name_or_path=args.dataset_name_or_path,
        step_log_dir=os.path.join(
            os.environ.get("UDTCA_LOG_DIR", "/data1/tangruijing/logs"),
            "init_steps_step_csv",
        ),
        lr=2.0e-4,
        local_sgd_mode="blocking",
        log_interval=1,
        # profiler 在 Corex 上额外扰动 CUDA，初始化阶段直接关掉
        disable_profiler=True,
        profiler_wait_steps=1,
        profiler_warmup_steps=1,
        profiler_active_steps=1,
        profiler_repeat=1,
        profiler_memory=False,
        profiler_shapes=False,
        profiler_stack=False,
        profiler_flops=False,
        profiler_acc_events=False,
    )


class _SyntheticTokens(Dataset):
    """不依赖 datasets/tokenizer 的假 batch，只为走通 trainer 的取样路径。

    schedule.step 之前就退出，所以这里的数据内容不影响任何训练结果；只要
    key/shape/dtype 和真实 dataloader 一致即可。
    """

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, seed: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        ids = torch.randint(0, self.vocab_size, (self.seq_len,), generator=gen)
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
        }


def _make_synthetic_dataloader(args, vocab_size: int):
    dataset = _SyntheticTokens(
        num_samples=max(4, args.per_device_batch_size * 4),
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        seed=args.seed,
    )
    return DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )


def _ensure_datasets_importable() -> None:
    """入口模块顶层有 `from datasets import load_dataset`，但模型阶段用不到它。

    镜像自带解释器的 numpy 是 1.26.4，而 pyarrow 要求 >=2.0，所以 `import
    datasets` 直接抛 ImportError，把整个入口模块的 import 带崩。这里只在
    datasets 确实不可用时注入一个占位模块（真调用 load_dataset 才报错）；
    需要真实 datasets 的 --with-data 阶段会在启动时单独检查并提示。
    """
    try:
        import datasets  # noqa: F401
        return
    except Exception as exc:  # noqa: BLE001
        reason = exc
    import types

    print(
        f"WARN: import datasets 失败（{reason!r}）；注入占位模块，"
        "模型阶段不依赖 datasets（--with-data 需要能 import datasets 的环境）",
        flush=True,
    )
    stub = types.ModuleType("datasets")

    def _load_dataset(*_args, **_kwargs):
        raise RuntimeError(
            "datasets 在这个解释器里不可用（pyarrow 需要 numpy>=2.0）"
        )

    stub.load_dataset = _load_dataset
    sys.modules["datasets"] = stub


def _load_entry_module():
    """只到模型阶段才 import 训练入口（它会拉起 psgd / transformers）。"""
    import importlib.util

    _ensure_datasets_importable()
    path = REPO_ROOT / "experiments" / "qwen14b" / "run_qwen14b_polar_dp_pp_tp.py"
    spec = importlib.util.spec_from_file_location("qwen14b_train_entry", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen14b_train_entry"] = mod
    spec.loader.exec_module(mod)
    return mod


def _detect_nic() -> str:
    """与启动器相同的网卡探测：默认路由的出口，其次第一个全局地址。"""
    import subprocess

    try:
        out = subprocess.run(
            ["ip", "-o", "-4", "route", "show", "default"],
            capture_output=True, text=True, timeout=5,
        ).stdout.split()
        if len(out) >= 5:
            return out[4]
        out = subprocess.run(
            ["ip", "-o", "-4", "addr", "show", "scope", "global"],
            capture_output=True, text=True, timeout=5,
        ).stdout.splitlines()
        for line in out:
            parts = line.split()
            if len(parts) > 1 and parts[1] != "lo":
                return parts[1].split("@")[0]
    except Exception:  # noqa: BLE001
        pass
    return ""


def _load_revoke():
    revoke_dir = os.environ.get(
        "UDTCA_REVOKE_DIR", str(REPO_ROOT / "experiments" / "qwen14b")
    )
    if revoke_dir not in sys.path:
        sys.path.insert(0, revoke_dir)
    try:
        import nccl_revoke  # noqa: PLC0415

        return nccl_revoke
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: 加载 nccl_revoke 失败（{exc!r}），卡住时不会自动收尾", flush=True)
        return None


def _shutdown(rank: int) -> None:
    """训练入口同款的退出顺序：barrier -> 各子组 -> world。"""
    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass
    try:
        dist.barrier()
        print(f"[shutdown][rank={rank}] barrier ok", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[shutdown][rank={rank}] barrier failed: {exc!r}", flush=True)

    world = getattr(getattr(dist, "group", None), "WORLD", None)
    groups = []
    try:
        from torch.distributed import distributed_c10d as _dc

        pg_map = list(_dc._world.pg_map.keys())
    except Exception:  # noqa: BLE001
        pg_map = []
    for g in pg_map:
        if g is world:
            continue
        try:
            ranks = tuple(dist.get_process_group_ranks(g))
        except Exception:  # noqa: BLE001
            ranks = ()
        try:
            tag = _dc._get_group_tag(g)
        except Exception:  # noqa: BLE001
            tag = ""
        groups.append((ranks, tag, g))
    groups.sort(key=lambda item: (item[0], item[1]))

    for ranks, tag, g in groups:
        try:
            dist.destroy_process_group(g)
            print(f"[shutdown][rank={rank}] destroyed ranks={ranks} tag={tag}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[shutdown][rank={rank}] destroy ranks={ranks} failed: {exc!r}",
                  flush=True)
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------- main

def main() -> int:
    global _CUDA_READY
    ap = argparse.ArgumentParser(description="qwen14b 训练初始化分步检查")
    ap.add_argument("--dp", type=int, default=2)
    ap.add_argument("--pp", type=int, default=2)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--method", choices=["none", "bitscom"], default="none",
                    help="bitscom = 和训练一样建 lowbit DP 组")
    ap.add_argument("--bitwidth", type=int, default=2)
    ap.add_argument("--with-model", action="store_true",
                    help="构建并切分 Qwen 模型（默认跳过，4 卡装不下）")
    ap.add_argument("--with-data", action="store_true",
                    help="建 token 缓存 + tokenizer + dataloader（需要真实 datasets）")
    ap.add_argument("--with-trainer", action="store_true",
                    help="构造 PolarParallel 并进入 train 循环，在 schedule.step "
                         "之前退出（需要 --with-model；数据用合成 batch）")
    ap.add_argument("--micro-batches", type=int, default=1,
                    help="PolarParallel 的 micro_batches（--with-trainer 时生效）")
    ap.add_argument("--comm-timing", type=int, default=0,
                    help="POLAR 通信时机（--with-trainer 时生效）")
    ap.add_argument("--polar-hook", type=str, default="ef_lowmem",
                    help="POLAR hook（默认和 launcher 一致：ef_lowmem）")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--dataset-name-or-path", type=str, default="HuggingFaceFW/fineweb")
    ap.add_argument("--dataset-config", type=str, default=None)
    ap.add_argument("--dataset-cache-dir", type=str,
                    default="experiments/qwen14b/cache")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=1,
                    help="只用于推 dataset_cache_samples，不训练")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--hang-timeout", type=float, default=180.0,
                    help="看门狗超时秒数，0 = 关闭")
    args = ap.parse_args()

    if "LOCAL_RANK" not in os.environ:
        print("ERROR: 必须用 torchrun 启动，见文件头部用法。", file=sys.stderr)
        return 2
    if args.with_data:
        # 数据阶段要真 datasets/pyarrow；镜像自带解释器的 numpy 1.26 不行。
        try:
            import datasets  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            print(
                f"ERROR: --with-data 需要能 import datasets，当前失败：{exc!r}\n"
                "       请用训练同款解释器（conda activate udtca），"
                "或在 4 卡上先不跑 --with-data。",
                file=sys.stderr,
            )
            return 2
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != args.dp * args.pp * args.tp:
        print(
            f"ERROR: world_size={world} 必须等于 dp*pp*tp="
            f"{args.dp * args.pp * args.tp}",
            file=sys.stderr,
        )
        return 2
    if args.with_trainer and not args.with_model:
        print("ERROR: --with-trainer 需要 --with-model（要先有切分好的 stage_model）",
              file=sys.stderr)
        return 2
    if args.with_trainer:
        os.environ.setdefault("POLAR_PRE_STEP_DEBUG", "1")

    # 环境与 run_docker_single_node.sh 的训练环境对齐。容器镜像自带
    # COREX_PATH/PATH/LD_LIBRARY_PATH/PYTHONPATH，这里只补会改变 NCCL
    # 初始化行为的那些；每一项都取自启动器同名导出（setdefault 可覆盖）。
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_RETRY_CNT", "600")
    os.environ.setdefault("NCCL_SOCKET_RETRY_SLEEP_MSEC", "1000")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("UDTCA_COMM_WATCHDOG_S", "600")
    # Corex 2.28+ 对称内存小消息 all_reduce 在 TG 卡上缺 device code，
    # bf16 all_reduce 会报 cudaErrorInvalidDeviceFunction(98)（假失败）。
    os.environ.setdefault("NCCL_SYM_KERNEL", "0")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    # 网卡：启动器按默认路由探测；NCCL 自己选也行，但训练里是显式钉住的。
    if not os.environ.get("NCCL_SOCKET_IFNAME"):
        _nic = _detect_nic()
        if _nic:
            os.environ["NCCL_SOCKET_IFNAME"] = _nic

    # 留证据：NCCL 全量日志落文件。
    log_dir = os.environ.get("UDTCA_LOG_DIR", "/data1/tangruijing/logs")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT")
    os.environ.setdefault(
        "NCCL_DEBUG_FILE", os.path.join(log_dir, "nccl_initsteps.%h.%p.log")
    )
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError:
        pass

    device = torch.device(f"cuda:{local_rank}")
    revoke = _load_revoke()
    watchdog = None

    rc = 0
    try:
        # ---- 1) bitscom.init（入口 main() 里在 init_process_group 之前）
        bitscom_module = None
        if args.method == "bitscom":
            import bitscom

            stage(rank, "1) import bitscom ok", f"path={bitscom.__file__}")
            bitscom.init(bitwidth=args.bitwidth)
            bitscom_module = bitscom
            stage(rank, "1) bitscom.init done", f"bitwidth={args.bitwidth}")
        else:
            stage(rank, "1) bitscom.init skipped", "--method none")

        # ---- 2) set_device（入口先设卡，再带 device_id 建 PG）
        torch.cuda.set_device(device)
        _CUDA_READY = True
        stage(rank, "2) set_device done", f"device={device} idx={torch.cuda.current_device()}")
        if rank == 0:
            print(
                f"=== qwen14b 初始化分步检查 ===\n"
                f"host={os.uname().nodename} world={world} "
                f"dp={args.dp} pp={args.pp} tp={args.tp} method={args.method} "
                f"with_model={args.with_model}\n"
                f"torch={torch.__version__} "
                f"nccl={'.'.join(str(v) for v in torch.cuda.nccl.version())} "
                f"device={torch.cuda.get_device_name(local_rank)}\n"
                f"NCCL_DEBUG_FILE={os.environ.get('NCCL_DEBUG_FILE', '-')} "
                f"NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', '-')} "
                f"NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE', '-')} "
                f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '-')} "
                f"CUDA_DEVICE_MAX_CONNECTIONS="
                f"{os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', '-')} "
                f"UDTCA_COMM_WATCHDOG_S={os.environ.get('UDTCA_COMM_WATCHDOG_S', '-')}",
                flush=True,
            )

        # 看门狗要在 set_device 之后创建（CommWatchdog 会读 current_device）
        if revoke is not None and args.hang_timeout > 0:
            watchdog = revoke.CommWatchdog(timeout_s=args.hang_timeout, exit_code=124)
            watchdog.capture(force=True)
            watchdog.touch()
            watchdog.start()
            stage(rank, "watchdog started", f"timeout={args.hang_timeout}s")
        if watchdog is not None:
            watchdog.touch()

        # ---- 3) init_process_group（入口 ~1391，eager 初始化）
        _wd = float(os.environ.get("UDTCA_COMM_WATCHDOG_S", "600"))
        _pg_timeout = timedelta(
            seconds=_wd + float(os.environ.get("UDTCA_PG_TIMEOUT_MARGIN_S", "21600"))
        )
        stage(rank, "3) init_process_group start", f"timeout={_pg_timeout}")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=_pg_timeout,
            device_id=device,
        )
        stage(
            rank, "3) init_process_group done",
            f"backend={dist.get_backend()} rank={dist.get_rank()}/{dist.get_world_size()}",
        )
        if watchdog is not None:
            watchdog.touch()
        torch.cuda.synchronize()
        dist.barrier()
        stage(rank, "3) first barrier ok")

        # ---- 4) init_device_mesh（入口 ~1421）
        dims = (args.dp, args.pp) if args.tp == 1 else (args.dp, args.pp, args.tp)
        names = ("dp", "pp") if args.tp == 1 else ("dp", "pp", "tp")
        stage(rank, "4) init_device_mesh start", f"shape={dims} names={names}")
        device_mesh = __import__(
            "torch.distributed.device_mesh", fromlist=["init_device_mesh"]
        ).init_device_mesh("cuda", dims, mesh_dim_names=names)
        for name in names:
            sub = device_mesh[name]
            stage(
                rank, f"4) mesh[{name}] ready",
                f"my_rank_in_group={sub.get_local_rank()} "
                f"ranks={list(dist.get_process_group_ranks(sub.get_group()))}",
            )
        dp_mesh = device_mesh["dp"]
        pp_mesh = device_mesh["pp"]
        if watchdog is not None:
            watchdog.touch()
        torch.cuda.synchronize()
        dist.barrier()
        stage(rank, "4) mesh checkpoint barrier ok")

        # ---- 5) lowbit DP 组（入口 ~1440，只有 bitscom）
        lowbit_dp_group = None
        if args.method == "bitscom":
            stage(rank, "5) create_lowbit_dp_group start")
            lowbit_dp_group = create_lowbit_dp_group(dp_mesh.get_group())
            stage(rank, "5) create_lowbit_dp_group done")
            if watchdog is not None:
                watchdog.touch()
            torch.cuda.synchronize()
            dist.barrier()
            stage(rank, "5) lowbit group checkpoint barrier ok")
        else:
            stage(rank, "5) create_lowbit_dp_group skipped", "--method none")

        # ---- 6) import 训练入口（数据/模型阶段都要用它；默认不 import）
        entry = None
        cfg = None
        if args.with_data or args.with_model:
            stage(rank, "6) load training entry module start")
            entry = _load_entry_module()
            stage(rank, "6) load training entry module done")
            if hasattr(entry, "_set_seed"):
                entry._set_seed(args.seed)
            else:
                torch.manual_seed(args.seed)
            cfg = _build_cfg(entry, args)
        else:
            stage(rank, "6) training entry module skipped",
                  "加 --with-data / --with-model 才 import")

        # ---- 7) 数据集缓存 + tokenizer + dataloader（入口 ~1447 / ~1474 / ~1523）
        if args.with_data:
            ns = _make_cache_args(args)
            stage(rank, "7) dataset cache start", f"dir={ns.dataset_cache_dir}")
            cache_path = entry._prepare_node_local_hf_cache(ns)
            stage(rank, "7) dataset cache done", f"path={cache_path}")

            cfg = _build_cfg(entry, args, cache_path)
            stage(rank, "7) build_tokenizer start")
            tokenizer = entry.build_tokenizer(cfg, local_files_only=True)
            stage(rank, "7) build_tokenizer done",
                  f"class={type(tokenizer).__name__} vocab={len(tokenizer)}")

            stage(rank, "7) get_dataloader start", f"pp={args.pp}")
            dataloader = entry.get_dataloader(cfg, tokenizer, args.pp)
            try:
                dl_len = len(dataloader)
            except Exception:  # noqa: BLE001
                dl_len = "?"
            stage(rank, "7) get_dataloader done", f"len={dl_len}")
            torch.cuda.synchronize()
            dist.barrier()
            stage(rank, "7) data checkpoint barrier ok")
        else:
            stage(rank, "7) dataset/tokenizer/dataloader skipped", "--with-data")

        # ---- 8) 模型构建 + PP 切分（入口 ~1479 / ~1482，默认跳过）
        if args.with_model:
            stage(rank, "8) build_qwen_model start", f"model={args.model_name}")
            model = entry.build_qwen_model(cfg, local_files_only=True)
            alloc, reserved = _mem_gib()
            stage(rank, "8) build_qwen_model done",
                  f"params={sum(p.numel() for p in model.parameters()):,}")

            stage_idx = pp_mesh.get_local_rank()
            stage(rank, "8) partition_qwen_model start", f"stage_idx={stage_idx}/{args.pp}")
            stage_model = entry.partition_qwen_model(
                model, stage_idx, args.pp, debug_nan_steps=0,
            )
            del model
            alloc, reserved = _mem_gib()
            stage(rank, "8) partition_qwen_model done",
                  f"stage_params={sum(p.numel() for p in stage_model.parameters()):,}")
            if watchdog is not None:
                watchdog.touch()
            torch.cuda.synchronize()
            dist.barrier()
            stage(rank, "8) model checkpoint barrier ok")
        else:
            stage(rank, "8) model build skipped", "加 --with-model 才构建")

        # ---- 9) LowBitGroup 构造（入口 ~1497；只构造，不发集合通信）
        lowbit_group = None
        if args.method == "bitscom":
            stage(rank, "9) LowBitGroup construct start")
            lowbit_group = bitscom_module.LowBitGroup(
                bitwidth=args.bitwidth,
                process_group=lowbit_dp_group,
                backend_allreduce=True,
            )
            stage(rank, "9) LowBitGroup construct done")
        else:
            stage(rank, "9) LowBitGroup construct skipped")

        # ---- 10) 入口在 trainer 之前做的两件小事
        torch.backends.cuda.matmul.allow_tf32 = True
        stage(rank, "10) allow_tf32=True")

        # 入口启动时枚举每个 PG 是否可取消（只枚举，不 revoke）。
        if os.environ.get("UDTCA_REVOKE_PROBE", "1") != "0":
            try:
                from nccl_revoke import _nccl_comms, revoke_summary

                probe = _nccl_comms(int(os.environ.get("LOCAL_RANK", "0")))
                stage(
                    rank, "10) revoke-probe done",
                    f"groups={revoke_summary().get('groups', 0)} "
                    f"cancelable={len(probe)}",
                )
            except Exception as exc:  # noqa: BLE001
                stage(rank, "10) revoke-probe unavailable", repr(exc))

        # ---- 11) PolarParallel 构造 + 进入 train 循环（在 schedule.step 前退出）
        if args.with_trainer:
            from psgd.parallelism.polar.wrapper import PolarParallel

            ns = _trainer_args(args, method=args.method)
            vocab_size = int(
                getattr(getattr(stage_model, "config", None), "vocab_size", 152064)
            )
            stage(
                rank, "11) PolarParallel construct start",
                f"hook={args.polar_hook} micro_batches={args.micro_batches} "
                f"comm_timing={args.comm_timing} vocab={vocab_size}",
            )
            dataloader = _make_synthetic_dataloader(args, vocab_size)

            def loss_fn(output, target):
                return entry.vocab_parallel_lm_loss(
                    output,
                    target,
                    ignore_index=0,
                    tp_mesh=device_mesh["tp"] if args.tp > 1 else None,
                    debug=False,
                )

            trainer = PolarParallel(
                args=ns,
                device_mesh=device_mesh,
                micro_batches=args.micro_batches,
                loss_fn=loss_fn,
                stage_model=stage_model,
                dataloader=dataloader,
                comm_timing=args.comm_timing,
                use_local_sgd=False,
                local_sgd_steps=1,
                baseline_mode="manual",
            )
            trainer.lowbit_group = lowbit_group
            if watchdog is not None:
                watchdog.touch()
            checkpoint(rank, "11) PolarParallel ready")

            stage(
                rank, "11) trainer.train() start",
                f"按 max_steps={args.max_steps} 正常训练",
            )
            trainer.train()
            stage(
                rank, "11) trainer.train() returned",
                "训练循环正常结束（含 schedule.step + DP 同步 + optimizer.step）",
            )
            del trainer
        else:
            stage(rank, "11) PolarParallel/trainer skipped", "加 --with-trainer 才构建")

        stage(rank, "SUMMARY PASS", f"total={time.monotonic() - _T0:.2f}s")

    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        stage(rank, f"SUMMARY FAIL {type(exc).__name__}", str(exc))
        rc = 1
    finally:
        if watchdog is not None:
            watchdog.stop()
        if dist.is_initialized():
            _shutdown(rank)
    return rc


if __name__ == "__main__":
    sys.exit(main())
