import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


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


def _read_loss_csv(path: Path) -> Tuple[List[int], List[float]]:
    steps: List[int] = []
    losses: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 2:
            raise RuntimeError(f"Invalid loss csv: {path}")
        for row in reader:
            if not row:
                continue
            steps.append(int(row[0]))
            losses.append(float(row[1]))
    return steps, losses


def _read_throughput_csv(path: Path) -> List[Tuple[str, str, float, float]]:
    rows: List[Tuple[str, str, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                (
                    str(row["model"]),
                    str(row["method"]),
                    float(row["avg_step_time_ms"]),
                    float(row["throughput_samples_per_s"]),
                )
            )
    return rows


def _write_combined_loss(
    *,
    out_path: Path,
    steps: List[int],
    losses_by_mode: Dict[str, List[float]],
) -> None:
    modes = list(losses_by_mode.keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", *[f"loss_{m}" for m in modes]])
        for idx, step in enumerate(steps):
            writer.writerow([step, *[losses_by_mode[m][idx] for m in modes]])


def _plot_loss_curves(
    *,
    out_path: Path,
    model_name: str,
    bitwidth: int,
    steps: List[int],
    losses_by_mode: Dict[str, List[float]],
    smooth_window: int,
) -> None:
    plt.figure(figsize=(9.0, 5.2))
    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#8c564b",
    ]
    for idx, (mode, losses) in enumerate(losses_by_mode.items()):
        color = colors[idx % len(colors)]
        plt.plot(steps, losses, color=color, alpha=0.35, linewidth=1.2)
        plt.plot(
            steps,
            _moving_average(losses, smooth_window),
            color=color,
            linewidth=2.0,
            label=f"{mode} (MA{smooth_window})",
        )

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"EF mode comparison | {model_name} | bitwidth={bitwidth}")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)


def _plot_throughput_summary(
    *,
    out_path: Path,
    models: List[str],
    modes: List[str],
    summary_rows: List[Tuple[str, str, float, float]],
) -> None:
    by_model_mode: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for model, mode, step_ms, throughput in summary_rows:
        by_model_mode[(model, mode)] = (step_ms, throughput)

    x = list(range(len(models)))
    width = 0.8 / max(len(modes), 1)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for idx, mode in enumerate(modes):
        offsets = [i - 0.4 + width / 2 + idx * width for i in x]
        step_vals = [by_model_mode.get((m, mode), (0.0, 0.0))[0] for m in models]
        thr_vals = [by_model_mode.get((m, mode), (0.0, 0.0))[1] for m in models]
        axes[0].bar(offsets, step_vals, width=width, label=mode)
        axes[1].bar(offsets, thr_vals, width=width, label=mode)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_title("Avg step time (ms)")
    axes[0].grid(True, axis="y", linestyle=":")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_title("Throughput (samples/s)")
    axes[1].grid(True, axis="y", linestyle=":")

    fig.suptitle("Error-feedback overhead summary")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)


def _run_worker(
    *,
    worker_script: Path,
    ef_mode: str,
    models: List[str],
    bitwidth: int,
    nproc: int,
    out_dir: str,
    extra_args: List[str],
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc),
        str(worker_script),
        "--ef-mode",
        ef_mode,
        "--bitwidth",
        str(bitwidth),
        "--out-dir",
        out_dir,
        "--models",
        *models,
        *extra_args,
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EF-mode sweeps and summarize loss/throughput")
    parser.add_argument("--modes", nargs="+", default=["none", "legacy", "ef21", "ef21_plus"])
    parser.add_argument("--models", nargs="+", default=["resnet50", "bert", "gpt2"])
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default="benchmarks/outputs")
    parser.add_argument("--smooth-window", type=int, default=6)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    worker_script = root / "multimodel_error_feedback_e2e_curve.py"
    out_dir = Path(args.out_dir)

    modes = [m.replace("ef21+", "ef21_plus") for m in args.modes]

    if not args.skip_run:
        for mode in modes:
            _run_worker(
                worker_script=worker_script,
                ef_mode=mode,
                models=args.models,
                bitwidth=args.bitwidth,
                nproc=args.nproc,
                out_dir=str(out_dir),
                extra_args=args.extra_args,
            )

    out_dir.mkdir(parents=True, exist_ok=True)

    combined_throughput: List[Tuple[str, str, float, float]] = []

    for mode in modes:
        safe_mode = mode.replace("+", "plus")
        summary_path = out_dir / f"ef_{safe_mode}_throughput_summary_bw{args.bitwidth}.csv"
        if summary_path.exists():
            combined_throughput.extend(_read_throughput_csv(summary_path))
        else:
            raise RuntimeError(f"missing throughput summary: {summary_path}")

    throughput_csv = out_dir / f"ef_modes_throughput_summary_bw{args.bitwidth}.csv"
    with throughput_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "method", "avg_step_time_ms", "throughput_samples_per_s"])
        writer.writerows(combined_throughput)

    throughput_png = out_dir / f"ef_modes_throughput_summary_bw{args.bitwidth}.png"
    _plot_throughput_summary(
        out_path=throughput_png,
        models=args.models,
        modes=modes,
        summary_rows=combined_throughput,
    )

    for model_name in args.models:
        steps_ref: List[int] | None = None
        losses_by_mode: Dict[str, List[float]] = {}

        for mode in modes:
            safe_mode = mode.replace("+", "plus")
            csv_path = out_dir / f"ef_{safe_mode}_{model_name}_loss_curve_bw{args.bitwidth}.csv"
            if not csv_path.exists():
                raise RuntimeError(f"missing loss curve: {csv_path}")
            steps, losses = _read_loss_csv(csv_path)
            if steps_ref is None:
                steps_ref = steps
            elif steps != steps_ref:
                raise RuntimeError(f"inconsistent steps for model={model_name} mode={mode}")
            losses_by_mode[mode] = losses

        if steps_ref is None:
            continue

        combined_csv = out_dir / f"ef_modes_{model_name}_loss_curve_bw{args.bitwidth}.csv"
        _write_combined_loss(out_path=combined_csv, steps=steps_ref, losses_by_mode=losses_by_mode)

        combined_png = out_dir / f"ef_modes_{model_name}_loss_curve_bw{args.bitwidth}.png"
        _plot_loss_curves(
            out_path=combined_png,
            model_name=model_name,
            bitwidth=args.bitwidth,
            steps=steps_ref,
            losses_by_mode=losses_by_mode,
            smooth_window=args.smooth_window,
        )

    print(f"Saved: {throughput_csv}")
    print(f"Saved: {throughput_png}")


if __name__ == "__main__":
    main()
