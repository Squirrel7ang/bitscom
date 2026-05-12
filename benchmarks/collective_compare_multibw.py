import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def _read_csv(path: Path) -> Tuple[float, float, float]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty csv: {path}")

    final_err = float(rows[-1]["rel_error"])
    lowbit_ms = sum(float(r["lowbit_time_ms"]) for r in rows) / len(rows)
    base_ms = sum(float(r["baseline_time_ms"]) for r in rows) / len(rows)
    return final_err, lowbit_ms, base_ms


def _plot(path: Path, bitwidths: List[int], errors: List[float], lowbit_ms: List[float], base_ms: List[float], collective: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    axes[0].plot(bitwidths, errors, marker="o", color="#d62728")
    axes[0].set_title("Final relative error")
    axes[0].set_xlabel("Bitwidth")
    axes[0].set_ylabel("||lowbit-baseline|| / ||baseline||")
    axes[0].grid(True, linestyle=":")

    axes[1].plot(bitwidths, lowbit_ms, marker="o", color="#1f77b4", label="lowbit")
    axes[1].plot(bitwidths, base_ms, marker="o", color="#2ca02c", label="baseline")
    axes[1].set_title("Avg step time (ms)")
    axes[1].set_xlabel("Bitwidth")
    axes[1].set_ylabel("ms")
    axes[1].grid(True, linestyle=":")
    axes[1].legend()

    fig.suptitle(f"{collective} compare (multi-bitwidth)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep bitwidths for lowbit collective compare")
    parser.add_argument("--collective", type=str, default="allreduce", choices=["allreduce", "reduce_scatter"])
    parser.add_argument("--bitwidths", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--numel", type=int, default=1 << 18)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default="benchmarks/outputs")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    worker = root / "collective_compare.py"
    out_dir = Path(args.out_dir)

    if not args.skip_run:
        for bw in args.bitwidths:
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                str(args.nproc),
                str(worker),
                "--collective",
                args.collective,
                "--bitwidth",
                str(bw),
                "--steps",
                str(args.steps),
                "--numel",
                str(args.numel),
                "--seed",
                str(args.seed),
                "--out-dir",
                str(out_dir),
                *args.extra_args,
            ]
            subprocess.run(cmd, check=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    bitwidths: List[int] = []
    errors: List[float] = []
    lowbit_ms: List[float] = []
    base_ms: List[float] = []

    for bw in args.bitwidths:
        csv_path = out_dir / f"collective_compare_{args.collective}_bw{bw}.csv"
        if not csv_path.exists():
            raise RuntimeError(f"missing csv: {csv_path}")
        final_err, avg_lowbit_ms, avg_base_ms = _read_csv(csv_path)
        bitwidths.append(int(bw))
        errors.append(final_err)
        lowbit_ms.append(avg_lowbit_ms)
        base_ms.append(avg_base_ms)

    summary_csv = out_dir / f"collective_compare_{args.collective}_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bitwidth", "final_rel_error", "lowbit_time_ms", "baseline_time_ms"])
        for idx, bw in enumerate(bitwidths):
            writer.writerow([bw, errors[idx], lowbit_ms[idx], base_ms[idx]])

    summary_png = out_dir / f"collective_compare_{args.collective}_summary.png"
    _plot(summary_png, bitwidths, errors, lowbit_ms, base_ms, args.collective)

    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_png}")


if __name__ == "__main__":
    main()
