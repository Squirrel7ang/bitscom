"""
稀疏化 reduce_scatter「精度统一」规则的跨机验证（双机，torchrun 启动）。

背景
----
priority（重要行）/ non-priority（非重要行）各自的通信精度满足：
    Full > Quant(位宽越大越精确) > Discard
规则：只要 non-priority 精度 >= priority 精度，就无需区分两种行，
整块用 non-priority 的 mode/位宽统一通信；严格更高时额外告警。
两边都是 Discard 时退化为零梯度（告警 + 跳过通信）——但这不应当发生。

本测试通过环境变量选择一组 (pri_mode, pri_bw, nonpri_mode, nonpri_bw)，
在 Python 侧复算同一套判定规则，然后校验 C++ 侧行为一致：

  * 统一为 Full     → 结果应与全精度求和**精确相等**（走普通 NCCL reduce_scatter）
  * 统一为 Quant    → 结果与全精度求和接近（有限且误差在量化量级内）
  * 统一为 Discard  → 结果恒为零
  * 仍区分两种行    → 结果有限、形状正确

启动（先 master 后 worker）：
    bash tests/run_sparse_unify.sh 0 'PRI_MODE=0 PRI_BW=4 NONPRI_MODE=0 NONPRI_BW=4'
    bash tests/run_sparse_unify.sh 1 'PRI_MODE=0 PRI_BW=4 NONPRI_MODE=0 NONPRI_BW=4'
"""

import os

import torch
import torch.distributed as dist

import bitscom
from bitscom.quantization import DEFAULT_BLOCK_SIZE


MODE_NAME = {0: "Full", 1: "Quant", 2: "Discard"}


def log(*args):
    print(*args, flush=True)


def resolve_plan(pri_mode, pri_bw, nonpri_mode, nonpri_bw):
    """复算 C++ resolveSparse：返回 (unify, uni_mode, uni_bw, degenerate, warn)。"""

    def level(mode):
        return 2 if mode == 0 else (1 if mode == 1 else 0)

    lp, ln = level(pri_mode), level(nonpri_mode)
    if lp != ln:
        cmp = ln - lp
    elif nonpri_mode == 1:
        cmp = nonpri_bw - pri_bw
    else:
        cmp = 0
    if cmp < 0:
        return False, None, None, False, False
    uni_mode = nonpri_mode
    return True, uni_mode, nonpri_bw, uni_mode == 2, cmp > 0


def bench(rank, world_size, n):
    expected_val = float(world_size * (world_size - 1) // 2)
    input_list = [torch.full((n,), float(rank), device="cuda") for _ in range(world_size)]
    output = torch.zeros(n, device="cuda")

    dist.barrier()
    # warm-up（不计结果）
    dist.reduce_scatter(output=output, input_list=input_list)
    torch.cuda.synchronize()

    output.zero_()
    dist.reduce_scatter(output=output, input_list=input_list)
    torch.cuda.synchronize()

    expected = torch.full((n,), expected_val, device="cuda")
    max_err = (output - expected).abs().max().item()
    finite = torch.isfinite(output).all().item()
    return finite, max_err


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    pri_mode = int(os.environ["PRI_MODE"])
    pri_bw = int(os.environ.get("PRI_BW", "4"))
    nonpri_mode = int(os.environ["NONPRI_MODE"])
    nonpri_bw = int(os.environ.get("NONPRI_BW", "4"))

    torch.cuda.set_device(local_rank)

    bitscom.init(
        bitwidth=16,
        error_feedback=False,
        error_feedback_mode="none",
        block_size=DEFAULT_BLOCK_SIZE,
        sparse_enabled=True,
        sparse_projection_rank=4,
        sparse_compression_ratio=0.1,
        sparse_row_width=128,
        sparse_priority_mode=pri_mode,
        sparse_priority_quantize_bitwidth=pri_bw,
        sparse_non_priority_mode=nonpri_mode,
        sparse_non_priority_quantize_bitwidth=nonpri_bw,
    )
    dist.init_process_group(backend="lowbit")

    unify, uni_mode, uni_bw, degenerate, warn = resolve_plan(
        pri_mode, pri_bw, nonpri_mode, nonpri_bw
    )
    if rank == 0:
        log(
            f"[cfg] pri={MODE_NAME[pri_mode]}({pri_bw}) "
            f"nonpri={MODE_NAME[nonpri_mode]}({nonpri_bw}) => "
            f"unify={unify} uni={MODE_NAME.get(uni_mode)} warp={warn} degenerate={degenerate}"
        )

    n = 1024 * 1024
    finite, max_err = bench(rank, world_size, n)

    # ---- 判定 ----
    ok = finite
    detail = f"finite={finite} max_abs_err={max_err:.6f}"
    if degenerate:
        ok = ok and max_err < 1e-6  # 应为全零
        detail += " expected=zeros"
    elif unify and uni_mode == 0:
        # 统一 Full = 普通 NCCL reduce_scatter，应精确相等
        ok = ok and max_err < 1e-4
        detail += " expected=exact"
    elif unify and uni_mode == 1:
        # 统一 Quant：量化误差，给个宽松上界
        ok = ok and max_err < 0.05 * max(1.0, abs(float(world_size * (world_size - 1) // 2)))
        detail += " expected=approx"

    dist.barrier()
    if rank == 0:
        log("=" * 60)
        log(f"RESULT pri={MODE_NAME[pri_mode]}({pri_bw}) nonpri={MODE_NAME[nonpri_mode]}({nonpri_bw}) "
            f": {'PASS' if ok else 'FAIL'}  ({detail})")
        log("=" * 60)

    dist.destroy_process_group()
    if not ok and rank == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
