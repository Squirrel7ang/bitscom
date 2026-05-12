import argparse
import csv
import os
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

import bitscom


def _make_allreduce_input(rank: int, numel: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + rank * 31)
    base = torch.linspace(-1.0, 1.0, steps=numel, dtype=torch.float32)
    noise = torch.randn(numel, generator=gen, dtype=torch.float32) * 0.02
    return (base + noise + rank * 0.05).to(device)


def _make_reduce_scatter_inputs(
    rank: int,
    world_size: int,
    numel: int,
    seed: int,
    device: torch.device,
) -> List[torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + rank * 47)
    base = torch.linspace(-1.0, 1.0, steps=numel, dtype=torch.float32)

    inputs = []
    for shard_idx in range(world_size):
        noise = torch.randn(numel, generator=gen, dtype=torch.float32) * 0.015
        shard = base + noise + rank * 0.04 + shard_idx * 0.01
        inputs.append(shard.to(device))
    return inputs


def _sync_and_time(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _plot(path: str, rows: List[Tuple[int, float, float, float]], collective: str, bitwidth: int) -> None:
    steps = [r[0] for r in rows]
    errors = [r[1] for r in rows]
    lowbit_ms = [r[2] for r in rows]
    baseline_ms = [r[3] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    axes[0].plot(steps, errors, color="#d62728", linewidth=1.8, label="lowbit vs baseline")
    axes[0].set_title("Relative Error")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("||lowbit-baseline|| / ||baseline||")
    axes[0].grid(True, linestyle=":")
    axes[0].legend()

    axes[1].plot(steps, lowbit_ms, color="#1f77b4", linewidth=1.8, label="lowbit")
    axes[1].plot(steps, baseline_ms, color="#2ca02c", linewidth=1.8, label="baseline")
    axes[1].set_title("Step Time (ms)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("ms")
    axes[1].grid(True, linestyle=":")
    axes[1].legend()

    fig.suptitle(f"{collective} compare | bitwidth={bitwidth}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare lowbit collective vs NCCL baseline")
    parser.add_argument("--collective", type=str, default="allreduce", choices=["allreduce", "reduce_scatter"])
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--numel", type=int, default=1 << 18)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=str, default="benchmarks/outputs")
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

    nccl_pg = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    lowbit_pg = dist.group.WORLD

    rows: List[Tuple[int, float, float, float]] = []

    for step in range(1, args.steps + 1):
        step_seed = args.seed + step * 13
        if args.collective == "allreduce":
            base_tensor = _make_allreduce_input(rank, args.numel, step_seed, device)
            lowbit_tensor = base_tensor.clone()

            baseline_time = _sync_and_time(
                lambda: dist.all_reduce(base_tensor, op=dist.ReduceOp.SUM, group=nccl_pg)
            )
            lowbit_time = _sync_and_time(
                lambda: dist.all_reduce(lowbit_tensor, op=dist.ReduceOp.SUM, group=lowbit_pg)
            )

            ref = base_tensor
            out = lowbit_tensor
        else:
            base_inputs = _make_reduce_scatter_inputs(rank, world_size, args.numel, step_seed, device)
            lowbit_inputs = [t.clone() for t in base_inputs]
            base_out = torch.empty_like(base_inputs[0])
            lowbit_out = torch.empty_like(base_inputs[0])

            baseline_time = _sync_and_time(
                lambda: dist.reduce_scatter(
                    base_out,
                    base_inputs,
                    op=dist.ReduceOp.SUM,
                    group=nccl_pg,
                )
            )
            lowbit_time = _sync_and_time(
                lambda: dist.reduce_scatter(
                    lowbit_out,
                    lowbit_inputs,
                    op=dist.ReduceOp.SUM,
                    group=lowbit_pg,
                )
            )

            ref = base_out
            out = lowbit_out

        diff = (out - ref).to(torch.float32)
        denom = ref.to(torch.float32).norm().item() + 1e-12
        rel_err = diff.norm().item() / denom

        rel_err_tensor = torch.tensor([rel_err], dtype=torch.float32, device=device)
        dist.all_reduce(rel_err_tensor, op=dist.ReduceOp.MAX)

        lowbit_ms = torch.tensor([lowbit_time * 1000.0], device=device)
        baseline_ms = torch.tensor([baseline_time * 1000.0], device=device)
        dist.all_reduce(lowbit_ms, op=dist.ReduceOp.MAX)
        dist.all_reduce(baseline_ms, op=dist.ReduceOp.MAX)

        if rank == 0:
            rows.append((step, float(rel_err_tensor.item()), float(lowbit_ms.item()), float(baseline_ms.item())))

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        csv_path = os.path.join(
            args.out_dir,
            f"collective_compare_{args.collective}_bw{args.bitwidth}.csv",
        )
        png_path = os.path.join(
            args.out_dir,
            f"collective_compare_{args.collective}_bw{args.bitwidth}.png",
        )

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "rel_error", "lowbit_time_ms", "baseline_time_ms"])
            writer.writerows(rows)

        _plot(png_path, rows, args.collective, args.bitwidth)
        print(f"Saved: {csv_path}")
        print(f"Saved: {png_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
