import argparse
import json
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass
class Window:
    start_us: float
    end_us: float


def _load_events(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("traceEvents", [])


def _extract_profiler_steps(events: Iterable[dict]) -> list[dict]:
    steps = []
    for e in events:
        if e.get("ph") != "X":
            continue
        name = e.get("name")
        if not isinstance(name, str):
            continue
        if not name.startswith("ProfilerStep"):
            continue
        if "ts" not in e or "dur" not in e:
            continue
        steps.append(e)
    steps.sort(key=lambda x: x["ts"])
    return steps


def _extract_all_complete_ops(events: Iterable[dict]) -> list[dict]:
    ops = []
    for e in events:
        if e.get("ph") != "X":
            continue
        if "ts" not in e or "dur" not in e:
            continue
        ops.append(e)
    ops.sort(key=lambda x: x["ts"])
    return ops


def _build_window(steps: list[dict], skip_steps: int, keep_steps: int) -> Window:
    if not steps:
        raise RuntimeError("No ProfilerStep events found in trace")
    if skip_steps >= len(steps):
        raise RuntimeError(
            f"skip_steps={skip_steps} is too large for {len(steps)} profiler steps"
        )

    start_idx = skip_steps
    if keep_steps <= 0:
        end_idx = len(steps) - 1
    else:
        end_idx = min(len(steps) - 1, start_idx + keep_steps - 1)

    start_us = float(steps[start_idx]["ts"])
    end_us = float(steps[end_idx]["ts"] + steps[end_idx]["dur"])
    return Window(start_us=start_us, end_us=end_us)


def _build_window_by_time(ops: list[dict], skip_ratio: float) -> Window:
    if not ops:
        raise RuntimeError("No complete ops found in trace")
    first_ts = float(min(e["ts"] for e in ops))
    last_end = float(max(e["ts"] + e["dur"] for e in ops))
    span = max(last_end - first_ts, 1.0)
    start_us = first_ts + span * min(max(skip_ratio, 0.0), 0.95)
    return Window(start_us=start_us, end_us=last_end)


def _events_in_window(events: Iterable[dict], window: Window) -> list[dict]:
    out = []
    for e in events:
        if e.get("ph") != "X":
            continue
        if "ts" not in e or "dur" not in e:
            continue
        ts = float(e["ts"])
        te = ts + float(e["dur"])
        if ts >= window.start_us and te <= window.end_us:
            out.append(e)
    return out


def _summarize(events: list[dict], topk: int):
    total_ms = sum(float(e["dur"]) for e in events) / 1000.0
    by_name_ms = Counter()
    by_name_count = Counter()

    for e in events:
        name = e.get("name", "<unknown>")
        by_name_ms[name] += float(e["dur"]) / 1000.0
        by_name_count[name] += 1

    comm_prefixes = ("nccl:", "gloo:")
    comm_ms = 0.0
    for n, ms in by_name_ms.items():
        if isinstance(n, str) and n.startswith(comm_prefixes):
            comm_ms += ms

    print(f"events in window: {len(events)}")
    print(f"window total op time: {total_ms:.3f} ms")
    print(f"window comm op time (nccl/gloo): {comm_ms:.3f} ms")
    if total_ms > 0:
        print(f"window comm share: {100.0 * comm_ms / total_ms:.2f}%")

    print("top ops by total ms:")
    for name, ms in by_name_ms.most_common(topk):
        print(f"  {name}: {ms:.3f} ms, count={by_name_count[name]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a stable step window from a PyTorch profiler Chrome trace"
    )
    parser.add_argument("--trace", required=True, help="Path to trace json")
    parser.add_argument(
        "--skip-steps",
        type=int,
        default=3,
        help="How many initial ProfilerStep events to skip as warmup",
    )
    parser.add_argument(
        "--keep-steps",
        type=int,
        default=0,
        help="How many steps to keep after warmup (0 means all remaining)",
    )
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--fallback-skip-ratio",
        type=float,
        default=0.2,
        help="When no ProfilerStep exists, skip this fraction of early timeline as warmup",
    )
    args = parser.parse_args()

    events = _load_events(args.trace)
    steps = _extract_profiler_steps(events)
    ops = _extract_all_complete_ops(events)

    print(f"trace: {args.trace}")
    print(f"profiler steps found: {len(steps)}")

    if steps:
        window = _build_window(steps, args.skip_steps, args.keep_steps)
    else:
        window = _build_window_by_time(ops, args.fallback_skip_ratio)
        print(
            "no ProfilerStep found; falling back to time-based stable window with "
            f"skip_ratio={args.fallback_skip_ratio}"
        )
    print(
        f"stable window us: [{window.start_us:.0f}, {window.end_us:.0f}] "
        f"(duration {(window.end_us - window.start_us)/1000.0:.3f} ms)"
    )

    filtered = _events_in_window(events, window)
    _summarize(filtered, args.topk)


if __name__ == "__main__":
    main()
