import argparse
import csv
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn as nn

import bitscom
from bitscom.api import LowBitGroup


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 256, hidden: int = 512, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_local_dataset(
    *,
    rank: int,
    dataset_size: int,
    in_dim: int,
    num_classes: int,
    base_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(base_seed + rank * 100_000)

    teacher_gen = torch.Generator(device="cpu")
    teacher_gen.manual_seed(base_seed + 9999)

    x = torch.randn(dataset_size, in_dim, generator=gen, dtype=torch.float32)
    teacher_w = torch.randn(in_dim, num_classes, generator=teacher_gen, dtype=torch.float32)
    teacher_b = torch.randn(num_classes, generator=teacher_gen, dtype=torch.float32)
    logits = x @ teacher_w + teacher_b
    y = torch.argmax(logits, dim=1).to(torch.int64)
    return x, y


def _make_batch(
    *,
    step: int,
    batch_size: int,
    dataset_x: torch.Tensor,
    dataset_y: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset_size = int(dataset_x.size(0))
    start = ((step - 1) * batch_size) % dataset_size
    end = start + batch_size

    if end <= dataset_size:
        x = dataset_x[start:end]
        y = dataset_y[start:end]
    else:
        wrap = end - dataset_size
        x = torch.cat([dataset_x[start:], dataset_x[:wrap]], dim=0)
        y = torch.cat([dataset_y[start:], dataset_y[:wrap]], dim=0)

    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def _moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    out: List[float] = []
    running = 0.0
    for idx, val in enumerate(values):
        running += val
        if idx >= window:
            running -= values[idx - window]
        denom = min(idx + 1, window)
        out.append(running / float(denom))
    return out


def _train_and_record_losses(
    *,
    stochastic_rounding: bool,
    comm_bitwidth: int,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> List[float]:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = TinyMLP(in_dim=args.in_dim, hidden=args.hidden, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)
    dataset_x, dataset_y = _build_local_dataset(
        rank=rank,
        dataset_size=args.dataset_size,
        in_dim=args.in_dim,
        num_classes=args.num_classes,
        base_seed=args.data_seed,
    )

    group = LowBitGroup(
        bitwidth=comm_bitwidth,
        stochastic_rounding=stochastic_rounding,
        process_group=dist.group.WORLD,
    )

    losses: List[float] = []

    for step in range(1, args.steps + 1):
        x, y = _make_batch(
            step=step,
            batch_size=args.batch_size,
            dataset_x=dataset_x,
            dataset_y=dataset_y,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        for p in model.parameters():
            if p.grad is None:
                continue
            group.all_reduce(p.grad, async_op=False)
            p.grad.div_(world_size)

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        optimizer.step()

        loss_scalar = loss.detach().to(torch.float32).view(1)
        gathered_losses = [torch.zeros_like(loss_scalar) for _ in range(world_size)]
        dist.all_gather(gathered_losses, loss_scalar)
        mean_loss = torch.stack(gathered_losses).mean()

        losses.append(float(mean_loss.item()))

    return losses


def _save_csv(path: str, rows: List[Tuple[int, float, float, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "loss_no_stochastic",
                "loss_stochastic",
                "loss_no_compression",
            ]
        )
        writer.writerows(rows)


def _plot(path: str, rows: List[Tuple[int, float, float, float]], bitwidth: int, world_size: int) -> None:
    steps = [r[0] for r in rows]
    loss_no_stochastic = [r[1] for r in rows]
    loss_stochastic = [r[2] for r in rows]
    loss_no_compression = [r[3] for r in rows]
    smooth_no_stochastic = _moving_average(loss_no_stochastic, window=10)
    smooth_stochastic = _moving_average(loss_stochastic, window=10)
    smooth_no_compression = _moving_average(loss_no_compression, window=10)

    plt.figure(figsize=(8.5, 5.0))
    plt.plot(steps, loss_no_stochastic, color="#d62728", alpha=0.28, linewidth=1.5)
    plt.plot(steps, loss_stochastic, color="#1f77b4", alpha=0.28, linewidth=1.5)
    plt.plot(steps, loss_no_compression, color="#2ca02c", alpha=0.28, linewidth=1.5)
    plt.plot(steps, smooth_no_stochastic, label="No Stochastic Rounding (MA10)", color="#d62728", linewidth=2.2)
    plt.plot(steps, smooth_stochastic, label="With Stochastic Rounding (MA10)", color="#1f77b4", linewidth=2.2)
    plt.plot(steps, smooth_no_compression, label="No Compression Baseline (MA10)", color="#2ca02c", linewidth=2.2)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(f"LowBit E2E Loss Convergence (bitwidth={bitwidth}, world_size={world_size})")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare lowbit e2e loss curves with/without stochastic rounding")
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--in-dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-seed", type=int, default=7)
    parser.add_argument("--dataset-size", type=int, default=8192)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--baseline-bitwidth", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="benchmarks/outputs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this e2e distributed benchmark")

    bitscom.init(bitwidth=args.bitwidth)
    dist.init_process_group(backend="lowbit")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 2:
        raise RuntimeError("This comparison requires world_size >= 2 to exercise distributed lowbit all_reduce")

    local_rank_env = os.environ.get("LOCAL_RANK")
    local_rank = int(local_rank_env) if local_rank_env is not None else rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    try:
        losses_no_stochastic = _train_and_record_losses(
            stochastic_rounding=False,
            comm_bitwidth=args.bitwidth,
            args=args,
            rank=rank,
            world_size=world_size,
            device=device,
        )

        losses_stochastic = _train_and_record_losses(
            stochastic_rounding=True,
            comm_bitwidth=args.bitwidth,
            args=args,
            rank=rank,
            world_size=world_size,
            device=device,
        )

        losses_no_compression = _train_and_record_losses(
            stochastic_rounding=False,
            comm_bitwidth=args.baseline_bitwidth,
            args=args,
            rank=rank,
            world_size=world_size,
            device=device,
        )

        if rank == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            csv_path = os.path.join(
                args.out_dir,
                f"e2e_loss_curve_stochastic_vs_nocompress_bw{args.bitwidth}.csv",
            )
            png_path = os.path.join(
                args.out_dir,
                f"e2e_loss_curve_stochastic_vs_nocompress_bw{args.bitwidth}.png",
            )
            rows = [
                (
                    step,
                    losses_no_stochastic[step - 1],
                    losses_stochastic[step - 1],
                    losses_no_compression[step - 1],
                )
                for step in range(1, args.steps + 1)
            ]
            _save_csv(csv_path, rows)
            _plot(png_path, rows, bitwidth=args.bitwidth, world_size=world_size)
            print(f"Saved: {csv_path}")
            print(f"Saved: {png_path}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
