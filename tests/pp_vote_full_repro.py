#!/usr/bin/env python3
"""完整初始化 + 单次 schedule.step()：复现 PP warm-up 投票卡死（2→1）。

和 tests/vote_repro.py 的区别
-----------------------------
vote_repro.py 只重放“投票”本身（裸 dist.send/recv），结果是 PASS。
本脚本把真实训练入口 run_qwen14b_polar_dp_pp_tp.py 的初始化**整条**走一遍：

  1. torch.cuda.set_device(local_rank)
  2. dist.init_process_group(backend="nccl", device_id=...)   （立刻建 comm）
  3. init_device_mesh("cuda", (dp, pp))                        （ncclCommSplit）
  4. AutoConfig / AutoTokenizer  ← 同一个 HF 缓存
  5. _set_seed(seed) + build_qwen_model()                      （meta 设备）
  6. partition_qwen_model(stage_idx, pp_size)                  （真实切分）
  7. PolarParallel.__init__：
       - stage_model.to_empty(cuda) + _reset_module_parameters
       - _broadcast_stage_parameters_from_dp_root()          ← 真实 run 里
         投票前那 73 次 DP 广播就在这一步
       - PipelineStage + Schedule1F1B
  8. 只调用一次 schedule.step(...)                             ← 卡死点在这里

其中 model / partition / loss / wrapper 全部**直接 import 真实入口模块**，
不是复刻，所以参数数量、显存占用、PG 数量、comm split 顺序都和真实 run 一致。

判定
----
  step 正常返回  -> 这套初始化 + 单步调度在 Corex 上能跑通（问题不在这条路径上）
  卡住           -> 与真实 run 同样的窗口收到窗口栈；看 SUMMARY 与 py-spy

防挂死（和真实 run 一致）
------------------------
CommWatchdog 每 UDTCA_COMM_WATCHDOG_S（默认 300 s）没有心跳就调 ncclCommRevoke，
等本机 16 个 rank 静默后 os._exit(124)。**不要**从外面套 timeout、不要 kill、
不要 Ctrl-C —— 那会打出 XID 24 让卡永久挂死。

跑法（容器里，16 卡；命令行和真实 run 只差脚本路径）
----------------------------------------------------
cd /data/tangruijing/udtca/bitscom && torchrun --nproc_per_node=16 --master_port=29631 tests/pp_vote_full_repro.py --pp-size 8 --tp-size 1 --micro-batches 16 --per-device-batch-size 16 --seq-len 256 --seed 1234 --steps 1 2>&1 | tee /data1/tangruijing/logs/pp_vote_full_$(date +%H%M%S).log

想抓卡住时的 Python 栈（真实入口 import 时就注册好了 SIGUSR1）：
  pgrep -f 'python -u /data/tangruijing/udtca/bitscom/tests/pp_vote_full_repro.py' | xargs -r kill -USR1
只对这个脚本的 worker 发；不要发给 torchrun。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

DEFAULT_ENTRY = (
    "/data/tangruijing/udtca/experiments/qwen14b/run_qwen14b_polar_dp_pp_tp.py"
)

# 进程组超时远大于 hang-timeout：先让 revoke 收尾，不让 torch 自己 abort
PG_TIMEOUT_S = 21600


def _log(msg: str) -> None:
    try:
        rank = dist.get_rank()
    except Exception:
        rank = -1
    print(f"[pp-repro rank={rank} pid={os.getpid()} t={time.strftime('%H:%M:%S')}] {msg}",
          flush=True)


def _load_entry(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"找不到真实入口脚本: {path}")
    spec = importlib.util.spec_from_file_location("qwen14b_entry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["qwen14b_entry"] = module
    spec.loader.exec_module(module)  # 这里会注册 SIGUSR1 faulthandler
    return module


def _load_revoke():
    revoke_dir = os.environ.get(
        "UDTCA_REVOKE_DIR", "/data/tangruijing/udtca/experiments/qwen14b"
    )
    if revoke_dir not in sys.path:
        sys.path.insert(0, revoke_dir)
    import nccl_revoke  # noqa: PLC0415

    return nccl_revoke


def _synthetic_batch(args, tokenizer, device):
    """和 get_dataloader 输出同形状的 batch：16 x 256，纯随机 token。"""
    vocab = int(getattr(tokenizer, "vocab_size", 152064))
    gen = torch.Generator(device=device).manual_seed(args.seed + dist.get_rank())
    input_ids = torch.randint(1, vocab - 1, (args.per_device_batch_size, args.seq_len),
                              generator=gen, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "labels": input_ids.clone()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--entry", default=DEFAULT_ENTRY, help="真实训练入口路径")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--tokenizer-name", default=None)
    ap.add_argument("--pp-size", type=int, default=8)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--micro-batches", type=int, default=16)
    ap.add_argument("--per-device-batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=1, help="调用几次 schedule.step")
    ap.add_argument("--use-flash-attn", action=argparse.BooleanOptionalAction,
                    default=False)
    ap.add_argument("--init-from-pretrained", action="store_true", default=False)
    ap.add_argument("--debug-nan-steps", type=int, default=0)
    ap.add_argument("--hang-timeout", type=float,
                    default=float(os.environ.get("UDTCA_COMM_WATCHDOG_S", "300")))
    ap.add_argument("--no-watchdog", action="store_true")
    ap.add_argument("--dump-stacks-on-hang", action="store_true", default=True,
                    help="保留参数，栈由 SIGUSR1 触发")
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"[pp-repro] 忽略未使用参数: {unknown}", flush=True)

    # --- 和真实 run 相同的 NCCL 环境兜底（启动器里也是这些）---
    os.environ.setdefault("NCCL_SYM_KERNEL", "0")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "ens120f0")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
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

    if "LOCAL_RANK" not in os.environ:
        print("ERROR: 必须用 torchrun 启动，见文件头部用法。", file=sys.stderr)
        return 2

    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{local_rank}")

    entry = _load_entry(args.entry)
    from torch.distributed.device_mesh import init_device_mesh

    model_parallel = args.pp_size * args.tp_size
    if world % model_parallel != 0:
        print(f"ERROR: world_size={world} 不能被 pp*tp={model_parallel} 整除",
              file=sys.stderr)
        return 2
    dp_size = world // model_parallel

    revoke = None if args.no_watchdog else _load_revoke()
    watchdog = None
    if revoke is not None:
        watchdog = revoke.CommWatchdog(timeout_s=args.hang_timeout, exit_code=124)
        watchdog.capture(force=True)
        watchdog.touch()
        watchdog.start()

    def beat(tag: str = "") -> None:
        if watchdog is not None:
            watchdog.touch()
        _log(f"beat {tag}")

    rc = 0
    t_start = time.perf_counter()
    try:
        # ---- 1. set_device + init_process_group(device_id) ----
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=PG_TIMEOUT_S),
            device_id=device,
        )
        _log(f"init_process_group done world={world} dp={dp_size} pp={args.pp_size}")
        beat("pg")

        # ---- 2. device mesh（和真实 run 同一套 dim 顺序）----
        if args.tp_size > 1:
            device_mesh = init_device_mesh(
                "cuda", (dp_size, args.pp_size, args.tp_size),
                mesh_dim_names=("dp", "pp", "tp"),
            )
        else:
            device_mesh = init_device_mesh(
                "cuda", (dp_size, args.pp_size), mesh_dim_names=("dp", "pp")
            )
        dp_mesh = device_mesh["dp"]
        pp_mesh = device_mesh["pp"]
        _log(f"device_mesh done stage={pp_mesh.get_local_rank()} dp_rank={dp_mesh.get_local_rank()}")
        beat("mesh")

        torch.backends.cuda.matmul.allow_tf32 = True

        # ---- 3. cfg / tokenizer / seed / model / partition ----
        cfg = entry.TrainConfig(
            model_name=args.model_name,
            tokenizer_name=args.tokenizer_name,
            dataset_name_or_path="HuggingFaceFW/fineweb",
            dataset_config=None,
            text_field="text",
            seq_len=args.seq_len,
            per_device_batch_size=args.per_device_batch_size,
            grad_accum_steps=1,
            lr=args.lr,
            warmup_ratio=0.02,
            max_tokens=0,
            max_steps=1,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
            clip_norm=1.0,
            log_interval=1,
            save_interval=1000,
            save_dir="/tmp/pp_vote_full_ckpt",
            num_workers=0,
            use_flash_attn=args.use_flash_attn,
            bf16=True,
            fp16=False,
            activation_checkpointing=False,
            init_from_pretrained=args.init_from_pretrained,
            dataset_cache_path=None,
        )
        tokenizer = entry.build_tokenizer(cfg, local_files_only=True)
        entry._set_seed(args.seed)
        _log("tokenizer ready")
        beat("tokenizer")

        model = entry.build_qwen_model(cfg, local_files_only=True)
        _log("model built (meta)")
        beat("model")

        stage_idx = pp_mesh.get_local_rank()
        stage_model = entry.partition_qwen_model(
            model, stage_idx, args.pp_size, debug_nan_steps=args.debug_nan_steps
        )
        _log(f"partitioned stage={stage_idx}")
        beat("partition")

        def loss_fn(output, target):
            return entry.vocab_parallel_lm_loss(
                output,
                target,
                ignore_index=tokenizer.pad_token_id,
                tp_mesh=device_mesh["tp"] if args.tp_size > 1 else None,
                debug=args.debug_nan_steps > 0,
            )

        wrapper_args = SimpleNamespace(
            using_polar=False,
            method="none",
            polar_hook="none",
            dataset_config=None,
            init_from_pretrained=args.init_from_pretrained,
            lr=args.lr,
            train_mode="pp_vote_repro",
            local_sgd_mode="blocking",
            seq_len=args.seq_len,
        )

        # ---- 4. PolarParallel：to_empty + 参数重置 + DP 参数广播 + 建 schedule ----
        _log("PolarParallel init start（含 DP 参数广播）")
        trainer = entry.PolarParallel(
            args=wrapper_args,
            device_mesh=device_mesh,
            micro_batches=args.micro_batches,
            loss_fn=loss_fn,
            stage_model=stage_model,
            dataloader=None,
            comm_timing=-1,
            use_local_sgd=False,
            local_sgd_steps=1,
            baseline_mode="manual",
        )
        _log("PolarParallel init done")
        beat("wrapper")

        # ---- 5. 单次 schedule.step：投票就在这里 ----
        batch = _synthetic_batch(args, tokenizer, device)
        losses = []
        for step in range(args.steps):
            _log(f"schedule.step {step} enter "
                 f"(is_first={trainer.stage.is_first} is_last={trainer.stage.is_last})")
            t0 = time.perf_counter()
            if trainer.stage.is_first:
                trainer.schedule.step(batch["input_ids"],
                                      attention_mask=batch["attention_mask"])
            elif trainer.stage.is_last:
                trainer.schedule.step(target=batch["labels"], losses=losses,
                                      attention_mask=batch["attention_mask"])
            else:
                trainer.schedule.step(attention_mask=batch["attention_mask"])
            _log(f"schedule.step {step} done in {time.perf_counter() - t0:.2f}s")
            beat(f"step{step}")

        if trainer.stage.is_last:
            loss = torch.stack(losses).mean()
            _log(f"loss={float(loss):.6f}")
        _log(f"SUMMARY PASS steps={args.steps} "
             f"total={time.perf_counter() - t_start:.1f}s")
    except Exception as exc:  # noqa: BLE001
        _log(f"SUMMARY FAIL {type(exc).__name__}: {exc}")
        rc = 1
    finally:
        if watchdog is not None:
            watchdog.stop()
        # 和真实入口一样的有序 teardown：先子组后 world
        try:
            entry.shutdown_distributed()
        except Exception:  # noqa: BLE001
            pass
        try:
            torch.cuda.synchronize()
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
