import argparse
import csv
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.profiler

import bitscom
from bitscom.api import LowBitGroup


def _make_run_stats(losses: List[float], total_time_s: float, batch_size: int, world_size: int) -> Dict[str, float | List[float]]:
    total_samples = float(batch_size * len(losses) * world_size)
    safe_time = max(total_time_s, 1e-9)
    return {
        "losses": losses,
        "avg_step_time_ms": (safe_time / max(len(losses), 1)) * 1000.0,
        "throughput_samples_per_s": total_samples / safe_time,
    }


def _moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    out: List[float] = []
    running = 0.0
    for idx, val in enumerate(values):
        running += val
        if idx >= window:
            running -= values[idx - window]
        out.append(running / float(min(idx + 1, window)))
    return out


def _build_resnet50_model(num_classes: int):
    import torchvision.models as tvm

    return tvm.resnet50(num_classes=num_classes)


def _build_bert_model(num_labels: int, hidden_size: int, num_layers: int, num_heads: int):
    from transformers import BertConfig, BertForSequenceClassification

    cfg = BertConfig(
        vocab_size=30522,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=512,
        num_labels=num_labels,
    )
    return BertForSequenceClassification(cfg)


def _build_gpt2_model(hidden_size: int, num_layers: int, num_heads: int):
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=50257,
        n_embd=hidden_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_positions=512,
        n_ctx=512,
    )
    return GPT2LMHeadModel(cfg)


def _build_dataset_resnet50(
    *,
    rank: int,
    dataset_size: int,
    image_size: int,
    num_classes: int,
    base_seed: int,
) -> Dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(base_seed + rank * 10_000)

    # Build low-frequency class prototypes to make the visual task easier for ResNet50.
    low_res = 12
    coarse = torch.randn(num_classes, 3, low_res, low_res, generator=gen, dtype=torch.float32) * 0.6
    prototypes = F.interpolate(
        coarse,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    y = torch.randint(0, num_classes, (dataset_size,), generator=gen, dtype=torch.int64)
    noise = torch.randn(dataset_size, 3, image_size, image_size, generator=gen, dtype=torch.float32) * 0.01
    x = prototypes[y] + noise
    return {"x": x, "y": y}


def _build_dataset_bert(
    *,
    rank: int,
    dataset_size: int,
    seq_len: int,
    num_labels: int,
    base_seed: int,
) -> Dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(base_seed + rank * 20_000)

    input_ids = torch.randint(0, 30522, (dataset_size, seq_len), generator=gen, dtype=torch.int64)
    attention_mask = torch.ones(dataset_size, seq_len, dtype=torch.int64)
    # Inject a strong label token cue at position 0 to make the task clearly learnable.
    labels = torch.randint(0, num_labels, (dataset_size,), generator=gen, dtype=torch.int64)
    input_ids[:, 0] = labels + 101
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _build_dataset_gpt2(
    *,
    rank: int,
    dataset_size: int,
    seq_len: int,
    base_seed: int,
) -> Dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(base_seed + rank * 30_000)

    # Use deterministic arithmetic-progress sequences so next-token prediction is learnable.
    base_vocab = 512
    starts = torch.randint(0, base_vocab, (dataset_size, 1), generator=gen, dtype=torch.int64)
    steps = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)
    input_ids = (starts + steps) % base_vocab
    attention_mask = torch.ones(dataset_size, seq_len, dtype=torch.int64)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _get_model_lr(model_name: str, mode: str, args: argparse.Namespace) -> float:
    if model_name == "resnet50":
        if mode == "lowbit_sr":
            return args.lr_resnet_sr
        return args.lr_resnet
    if model_name == "bert":
        return args.lr_bert
    if model_name == "gpt2":
        return args.lr_gpt2
    raise ValueError(f"unsupported model: {model_name}")


def _get_train_cfg(model_name: str, args: argparse.Namespace) -> Dict[str, float]:
    is_nlp = model_name in {"bert", "gpt2"}
    if is_nlp:
        steps = args.nlp_steps if args.nlp_steps is not None else args.steps
        batch_size = args.nlp_batch_size if args.nlp_batch_size is not None else args.batch_size
        dataset_size = args.nlp_dataset_size if args.nlp_dataset_size is not None else args.dataset_size
        grad_clip_norm = (
            args.nlp_grad_clip_norm if args.nlp_grad_clip_norm is not None else args.grad_clip_norm
        )
        warmup_ratio = args.nlp_warmup_ratio if args.nlp_warmup_ratio is not None else args.warmup_ratio
        min_lr_ratio = args.nlp_min_lr_ratio if args.nlp_min_lr_ratio is not None else args.min_lr_ratio
    else:
        steps = args.resnet_steps if args.resnet_steps is not None else args.steps
        batch_size = args.resnet_batch_size if args.resnet_batch_size is not None else args.batch_size
        dataset_size = args.resnet_dataset_size if args.resnet_dataset_size is not None else args.dataset_size
        grad_clip_norm = (
            args.resnet_grad_clip_norm if args.resnet_grad_clip_norm is not None else args.grad_clip_norm
        )
        warmup_ratio = (
            args.resnet_warmup_ratio if args.resnet_warmup_ratio is not None else args.warmup_ratio
        )
        min_lr_ratio = (
            args.resnet_min_lr_ratio if args.resnet_min_lr_ratio is not None else args.min_lr_ratio
        )

    return {
        "steps": int(steps),
        "batch_size": int(batch_size),
        "dataset_size": int(dataset_size),
        "grad_clip_norm": float(grad_clip_norm),
        "warmup_ratio": float(warmup_ratio),
        "min_lr_ratio": float(min_lr_ratio),
    }


def _slice_batch(data: Dict[str, torch.Tensor], step: int, batch_size: int) -> Dict[str, torch.Tensor]:
    dataset_size = int(next(iter(data.values())).size(0))
    start = ((step - 1) * batch_size) % dataset_size
    end = start + batch_size

    out: Dict[str, torch.Tensor] = {}
    for k, v in data.items():
        if end <= dataset_size:
            out[k] = v[start:end]
        else:
            wrap = end - dataset_size
            out[k] = torch.cat([v[start:], v[:wrap]], dim=0)
    return out


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _sync_grads_bucketed(
    grads: List[torch.Tensor],
    group: LowBitGroup,
    world_size: int,
    bucket_numel: int,
) -> None:
    if not grads:
        return

    bucket: List[torch.Tensor] = []
    bucket_size = 0

    def _flush_bucket() -> None:
        nonlocal bucket, bucket_size
        if not bucket:
            return

        flat = torch.cat([g.contiguous().view(-1) for g in bucket], dim=0)
        group.all_reduce(flat, async_op=False)
        flat.div_(world_size)

        offset = 0
        for g in bucket:
            n = g.numel()
            g.copy_(flat[offset : offset + n].view_as(g))
            offset += n

        bucket = []
        bucket_size = 0

    for g in grads:
        n = int(g.numel())
        if bucket and bucket_size + n > bucket_numel:
            _flush_bucket()
        bucket.append(g)
        bucket_size += n

    _flush_bucket()


def _run_train(
    *,
    model_name: str,
    mode: str,
    model: torch.nn.Module,
    dataset: Dict[str, torch.Tensor],
    train_cfg: Dict[str, float],
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    lowbit_pg,
    nccl_pg,
    device: torch.device,
) -> Dict[str, float | List[float]]:
    model.train()
    model.to(device)

    if mode == "nocompress":
        group = LowBitGroup(bitwidth=8, stochastic_rounding=False, process_group=nccl_pg)
    elif mode == "lowbit_no_sr":
        group = LowBitGroup(bitwidth=args.bitwidth, stochastic_rounding=False, process_group=lowbit_pg)
    elif mode == "lowbit_sr":
        group = LowBitGroup(bitwidth=args.bitwidth, stochastic_rounding=True, process_group=lowbit_pg)
    else:
        raise ValueError(f"unknown mode: {mode}")

    lr = _get_model_lr(model_name, mode, args)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    total_steps = int(train_cfg["steps"])
    batch_size = int(train_cfg["batch_size"])
    grad_clip_norm = float(train_cfg["grad_clip_norm"])
    warmup_ratio = float(train_cfg["warmup_ratio"])
    min_lr_ratio = float(train_cfg["min_lr_ratio"])

    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def _lr_lambda(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        progress = (step_idx - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    losses: List[float] = []
    step_times: List[float] = []

    torch.cuda.synchronize(device)

    mode_label = "no_compression" if mode == "nocompress" else mode

    profile_this_run = (
        bool(args.profile_trace)
        and rank == 0
        and model_name == args.profile_model
        and mode_label in set(args.profile_modes)
        and args.profile_steps > 0
    )
    max_profile_steps = min(total_steps, int(args.profile_steps))

    profiler_ctx = nullcontext()
    prof = None
    if profile_this_run:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler_ctx = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=False,
        )

    with profiler_ctx as prof:
        for step in range(1, total_steps + 1):
            t0 = time.perf_counter()
            batch = _to_device(_slice_batch(dataset, step, batch_size), device)

            optimizer.zero_grad(set_to_none=True)

            if model_name == "resnet50":
                logits = model(batch["x"])
                loss = F.cross_entropy(logits, batch["y"])
            elif model_name == "bert":
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = out.loss
            elif model_name == "gpt2":
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"],
                )
                loss = out.loss
            else:
                raise ValueError(f"unsupported model: {model_name}")

            loss.backward()

            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not args.disable_grad_bucket and args.grad_bucket_numel > 0:
                _sync_grads_bucketed(
                    grads=grads,
                    group=group,
                    world_size=world_size,
                    bucket_numel=int(args.grad_bucket_numel),
                )
            else:
                for g in grads:
                    group.all_reduce(g, async_op=False)
                    g.div_(world_size)

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()
            scheduler.step()

            loss_scalar = loss.detach().to(torch.float32).view(1)
            gathered = [torch.zeros_like(loss_scalar) for _ in range(world_size)]
            dist.all_gather(gathered, loss_scalar, group=nccl_pg)
            mean_loss = torch.stack(gathered).mean()
            losses.append(float(mean_loss.item()))

            torch.cuda.synchronize(device)
            step_times.append(time.perf_counter() - t0)

            if profile_this_run and step <= max_profile_steps:
                prof.step()

    if profile_this_run:
        os.makedirs(args.out_dir, exist_ok=True)
        trace_path = os.path.join(
            args.out_dir,
            f"trace_{model_name}_{mode_label}_bw{args.bitwidth}.json",
        )
        prof.export_chrome_trace(trace_path)
        print(f"Saved trace: {trace_path}")

    return _make_run_stats(
        losses=losses,
        total_time_s=sum(step_times),
        batch_size=batch_size,
        world_size=world_size,
    )


def _save_outputs(
    *,
    model_name: str,
    bitwidth: int,
    world_size: int,
    out_dir: str,
    losses_no_sr: List[float],
    losses_sr: List[float],
    losses_nocompress: List[float],
    smooth_window: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"multimodel_{model_name}_loss_curve_bw{bitwidth}.csv")
    png_path = os.path.join(out_dir, f"multimodel_{model_name}_loss_curve_bw{bitwidth}.png")

    rows = [
        (step, losses_no_sr[step - 1], losses_sr[step - 1], losses_nocompress[step - 1])
        for step in range(1, len(losses_no_sr) + 1)
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss_lowbit_no_sr", "loss_lowbit_sr", "loss_no_compression"])
        writer.writerows(rows)

    steps = [r[0] for r in rows]
    no_sr = [r[1] for r in rows]
    sr = [r[2] for r in rows]
    no_comp = [r[3] for r in rows]
    window = max(1, int(smooth_window))

    plt.figure(figsize=(8.5, 5.0))
    plt.plot(steps, no_sr, color="#d62728", alpha=0.45, linewidth=1.3)
    plt.plot(steps, sr, color="#1f77b4", alpha=0.45, linewidth=1.3)
    plt.plot(steps, no_comp, color="#2ca02c", alpha=0.45, linewidth=1.3)

    plt.plot(
        steps,
        _moving_average(no_sr, window),
        color="#d62728",
        linewidth=2.0,
        label=f"Lowbit no SR (MA{window})",
    )
    plt.plot(
        steps,
        _moving_average(sr, window),
        color="#1f77b4",
        linewidth=2.0,
        label=f"Lowbit with SR (MA{window})",
    )
    plt.plot(
        steps,
        _moving_average(no_comp, window),
        color="#2ca02c",
        linewidth=2.0,
        label=f"No compression (MA{window})",
    )

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"{model_name} E2E Convergence | bitwidth={bitwidth} | world_size={world_size}")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)

    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")


def _save_throughput_summary(
    *,
    out_dir: str,
    bitwidth: int,
    rows: List[Tuple[str, str, float, float]],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"multimodel_throughput_summary_bw{bitwidth}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "method", "avg_step_time_ms", "throughput_samples_per_s"])
        writer.writerows(rows)
    print(f"Saved: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model e2e benchmark: lowbit SR vs no-SR vs no-compression")
    parser.add_argument("--models", nargs="+", default=["resnet50", "bert", "gpt2"], choices=["resnet50", "bert", "gpt2"])
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--resnet-steps", type=int, default=None)
    parser.add_argument("--resnet-batch-size", type=int, default=16)
    parser.add_argument("--resnet-dataset-size", type=int, default=None)
    parser.add_argument("--resnet-grad-clip-norm", type=float, default=None)
    parser.add_argument("--resnet-warmup-ratio", type=float, default=None)
    parser.add_argument("--resnet-min-lr-ratio", type=float, default=None)
    parser.add_argument("--nlp-steps", type=int, default=160)
    parser.add_argument("--nlp-batch-size", type=int, default=4)
    parser.add_argument("--nlp-dataset-size", type=int, default=1024)
    parser.add_argument("--nlp-grad-clip-norm", type=float, default=0.8)
    parser.add_argument("--nlp-warmup-ratio", type=float, default=0.08)
    parser.add_argument("--nlp-min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--lr-resnet", type=float, default=2e-4)
    parser.add_argument("--lr-resnet-sr", type=float, default=1.5e-4)
    parser.add_argument("--lr-bert", type=float, default=2.5e-4)
    parser.add_argument("--lr-gpt2", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-seed", type=int, default=17)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--grad-bucket-numel", type=int, default=4_000_000)
    parser.add_argument("--disable-grad-bucket", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=4)
    parser.add_argument("--profile-trace", action="store_true")
    parser.add_argument("--profile-model", type=str, default="resnet50", choices=["resnet50", "bert", "gpt2"])
    parser.add_argument(
        "--profile-modes",
        nargs="+",
        default=["lowbit_no_sr", "no_compression"],
        choices=["lowbit_no_sr", "lowbit_sr", "no_compression"],
    )
    parser.add_argument("--profile-steps", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="benchmarks/outputs")
    parser.add_argument("--resnet-image-size", type=int, default=224)
    parser.add_argument("--resnet-num-classes", type=int, default=20)
    parser.add_argument("--bert-seq-len", type=int, default=64)
    parser.add_argument("--bert-hidden", type=int, default=256)
    parser.add_argument("--bert-layers", type=int, default=4)
    parser.add_argument("--bert-heads", type=int, default=4)
    parser.add_argument("--bert-num-labels", type=int, default=2)
    parser.add_argument("--gpt2-seq-len", type=int, default=64)
    parser.add_argument("--gpt2-hidden", type=int, default=256)
    parser.add_argument("--gpt2-layers", type=int, default=4)
    parser.add_argument("--gpt2-heads", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    bitscom.init(bitwidth=args.bitwidth)
    dist.init_process_group(backend="lowbit")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 2:
        raise RuntimeError("Need world_size >= 2")

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Build a full-precision NCCL group for no-compression baseline.
    nccl_pg = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    lowbit_pg = dist.group.WORLD
    throughput_rows: List[Tuple[str, str, float, float]] = []

    try:
        for model_name in args.models:
            train_cfg = _get_train_cfg(model_name, args)
            if rank == 0:
                print(
                    f"[bench] model={model_name} steps={train_cfg['steps']} "
                    f"batch={train_cfg['batch_size']} dataset={train_cfg['dataset_size']} "
                    f"bitwidth={args.bitwidth}"
                )

            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

            if model_name == "resnet50":
                model_a = _build_resnet50_model(args.resnet_num_classes)
                model_b = _build_resnet50_model(args.resnet_num_classes)
                model_c = _build_resnet50_model(args.resnet_num_classes)
                dataset = _build_dataset_resnet50(
                    rank=rank,
                    dataset_size=train_cfg["dataset_size"],
                    image_size=args.resnet_image_size,
                    num_classes=args.resnet_num_classes,
                    base_seed=args.data_seed,
                )
            elif model_name == "bert":
                model_a = _build_bert_model(args.bert_num_labels, args.bert_hidden, args.bert_layers, args.bert_heads)
                model_b = _build_bert_model(args.bert_num_labels, args.bert_hidden, args.bert_layers, args.bert_heads)
                model_c = _build_bert_model(args.bert_num_labels, args.bert_hidden, args.bert_layers, args.bert_heads)
                dataset = _build_dataset_bert(
                    rank=rank,
                    dataset_size=train_cfg["dataset_size"],
                    seq_len=args.bert_seq_len,
                    num_labels=args.bert_num_labels,
                    base_seed=args.data_seed,
                )
            elif model_name == "gpt2":
                model_a = _build_gpt2_model(args.gpt2_hidden, args.gpt2_layers, args.gpt2_heads)
                model_b = _build_gpt2_model(args.gpt2_hidden, args.gpt2_layers, args.gpt2_heads)
                model_c = _build_gpt2_model(args.gpt2_hidden, args.gpt2_layers, args.gpt2_heads)
                dataset = _build_dataset_gpt2(
                    rank=rank,
                    dataset_size=train_cfg["dataset_size"],
                    seq_len=args.gpt2_seq_len,
                    base_seed=args.data_seed,
                )
            else:
                raise ValueError(model_name)

            run_no_sr = _run_train(
                model_name=model_name,
                mode="lowbit_no_sr",
                model=model_a,
                dataset=dataset,
                train_cfg=train_cfg,
                args=args,
                rank=rank,
                world_size=world_size,
                lowbit_pg=lowbit_pg,
                nccl_pg=nccl_pg,
                device=device,
            )
            run_sr = _run_train(
                model_name=model_name,
                mode="lowbit_sr",
                model=model_b,
                dataset=dataset,
                train_cfg=train_cfg,
                args=args,
                rank=rank,
                world_size=world_size,
                lowbit_pg=lowbit_pg,
                nccl_pg=nccl_pg,
                device=device,
            )
            run_nocompress = _run_train(
                model_name=model_name,
                mode="nocompress",
                model=model_c,
                dataset=dataset,
                train_cfg=train_cfg,
                args=args,
                rank=rank,
                world_size=world_size,
                lowbit_pg=lowbit_pg,
                nccl_pg=nccl_pg,
                device=device,
            )

            if rank == 0:
                throughput_rows.extend(
                    [
                        (
                            model_name,
                            "lowbit_no_sr",
                            float(run_no_sr["avg_step_time_ms"]),
                            float(run_no_sr["throughput_samples_per_s"]),
                        ),
                        (
                            model_name,
                            "lowbit_sr",
                            float(run_sr["avg_step_time_ms"]),
                            float(run_sr["throughput_samples_per_s"]),
                        ),
                        (
                            model_name,
                            "no_compression",
                            float(run_nocompress["avg_step_time_ms"]),
                            float(run_nocompress["throughput_samples_per_s"]),
                        ),
                    ]
                )
                _save_outputs(
                    model_name=model_name,
                    bitwidth=args.bitwidth,
                    world_size=world_size,
                    out_dir=args.out_dir,
                    losses_no_sr=run_no_sr["losses"],
                    losses_sr=run_sr["losses"],
                    losses_nocompress=run_nocompress["losses"],
                    smooth_window=args.smooth_window,
                )
                print(
                    f"[throughput] {model_name} lowbit_no_sr "
                    f"step_ms={run_no_sr['avg_step_time_ms']:.3f} "
                    f"samples/s={run_no_sr['throughput_samples_per_s']:.3f}"
                )
                print(
                    f"[throughput] {model_name} lowbit_sr "
                    f"step_ms={run_sr['avg_step_time_ms']:.3f} "
                    f"samples/s={run_sr['throughput_samples_per_s']:.3f}"
                )
                print(
                    f"[throughput] {model_name} no_compression "
                    f"step_ms={run_nocompress['avg_step_time_ms']:.3f} "
                    f"samples/s={run_nocompress['throughput_samples_per_s']:.3f}"
                )

            dist.barrier(group=nccl_pg)

        if rank == 0:
            _save_throughput_summary(
                out_dir=args.out_dir,
                bitwidth=args.bitwidth,
                rows=throughput_rows,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
