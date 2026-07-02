"""
测试：逐行混合量化 — 部分行 dense、部分行 quantize 的正确性和精度。

测试策略：
  - HalfDenseStrategy: 前一半行走 dense NCCL，后一半行走 lowbit 量化
  - AllDenseStrategy:   全 dense，等价于标准 NCCL allreduce
  - FullQuantizationStrategy: isActive=False，退化为全量化

参考值通过 dist.all_gather（ProcessGroupLowBit 直接转发给 NCCL）
得到各 rank 的原始 tensor，然后本地求和。

运行方式（>=2 GPU，单机）：
    torchrun --nproc_per_node=2 bitscom/tests/test_row_partition_strategy.py

跨节点：
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
             --master_addr=<addr> --master_port=29500 \
             bitscom/tests/test_row_partition_strategy.py
"""

import os
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d


def log(msg):
    print(f"[rank {dist.get_rank()}] {msg}", flush=True)


# ================================================================
#  策略类 — Python 子类化 TensorPartitionStrategy
# ================================================================

def make_half_dense_strategy(row_size: int):
    """
    返回一个 Python 子类化的 TensorPartitionStrategy：
      - 前 num_rows//2 行 → dense  (NCCL allreduce，不量化)
      - 剩余行           → quantize (lowbit 量化)
    """
    from bitscom._lowbit_c import TensorPartitionStrategy, QuantizationSegment

    class HalfDenseStrategy(TensorPartitionStrategy):
        def __init__(self, row_size):
            super().__init__()
            self._row_size = row_size

        def prepare(self, tensors, rank, world_size, comm_ptr, stream_ptr):
            return True  # 不需要预处理

        def partition(self, flat_tensor):
            num_rows = flat_tensor.numel() // self._row_size
            split = max(1, num_rows // 2)

            seg_dense = QuantizationSegment()
            seg_dense.offset = 0
            seg_dense.numel = split * self._row_size
            seg_dense.quantize = False

            seg_quant = QuantizationSegment()
            seg_quant.offset = split * self._row_size
            seg_quant.numel = (num_rows - split) * self._row_size
            seg_quant.quantize = True

            return [seg_dense, seg_quant]

        def name(self):
            return "half_dense"

        def is_active(self):
            return True

    return HalfDenseStrategy(row_size)


def make_all_dense_strategy(row_size: int):
    """全 dense 策略 — 所有行都不量化。"""
    from bitscom._lowbit_c import TensorPartitionStrategy, QuantizationSegment

    class AllDenseStrategy(TensorPartitionStrategy):
        def __init__(self, row_size):
            super().__init__()
            self._row_size = row_size

        def prepare(self, tensors, rank, world_size, comm_ptr, stream_ptr):
            return True

        def partition(self, flat_tensor):
            seg = QuantizationSegment()
            seg.offset = 0
            seg.numel = flat_tensor.numel()
            seg.quantize = False
            return [seg]

        def name(self):
            return "all_dense"

        def is_active(self):
            return True

    return AllDenseStrategy(row_size)


# ================================================================
#  ProcessGroupLowBit 后端操作（通过 C++ helper 间接访问）
# ================================================================

def get_default_pg():
    """返回默认 process group。"""
    return c10d._get_default_group()


def set_strategy_on_pg(strategy):
    """在默认 lowbit process group 上设置分区策略。传 None 清除策略。"""
    from bitscom._lowbit_c import _set_strategy_on_pg
    _set_strategy_on_pg(get_default_pg(), strategy)


def get_strategy_name_from_pg() -> str:
    """获取当前策略名称。"""
    from bitscom._lowbit_c import _get_strategy_name_from_pg
    return _get_strategy_name_from_pg(get_default_pg())


def compute_dense_reference(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    通过 all_gather 获取所有 rank 的原始 tensor，本地求和得到 dense allreduce 参考值。
    ProcessGroupLowBit 的 allgather 方法直接转发给底层 NCCL，不经过量化。
    """
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return sum(gathered)


# ================================================================
#  测试用例
# ================================================================

def test_half_dense_correctness():
    """
    核心测试：前一半行 dense、后一半行量化，验证 allreduce 结果与参考一致。

    数据流：
        tensor [128行 × 64列] = 8192 元素
          ├── 行 0..63  → segment[offset=0,     numel=4096, quantize=False] → NCCL dense
          └── 行 64..127 → segment[offset=4096, numel=4096, quantize=True]  → lowbit 4-bit
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    row_size = 64
    num_rows = 128
    total_elements = num_rows * row_size  # 8192

    assert total_elements % world_size == 0, (
        f"numel ({total_elements}) must be divisible by world_size ({world_size}) "
        f"for the lowbit alltoall protocol"
    )

    # ── 设置策略 ──
    strategy = make_half_dense_strategy(row_size=row_size)
    set_strategy_on_pg(strategy)
    log(f"strategy: {get_strategy_name_from_pg()}")

    # ── 准备数据 ──
    torch.manual_seed(42 + rank)
    tensor = torch.randn(total_elements, dtype=torch.float32, device=device)

    # 参考值：通过 all_gather 获取各 rank 原始数据再求和
    ref = compute_dense_reference(tensor, world_size)

    # ── 混合策略 allreduce ──
    test_t = tensor.clone()
    dist.all_reduce(test_t, op=dist.ReduceOp.SUM)

    # ── 比较 ──
    # dense 半部分误差应非常小（纯 NCCL），量化半部分有 4-bit 量化误差
    # 整体容忍度：2% 相对误差 + 量化 scaling 的绝对误差
    max_err = (test_t - ref).abs().max().item()
    mean_err = (test_t - ref).abs().mean().item()

    # 分行统计误差
    dense_part_test = test_t[:4096]
    dense_part_ref = ref[:4096]
    quant_part_test = test_t[4096:]
    quant_part_ref = ref[4096:]

    dense_max = (dense_part_test - dense_part_ref).abs().max().item()
    quant_max = (quant_part_test - quant_part_ref).abs().max().item()
    dense_mean = (dense_part_test - dense_part_ref).abs().mean().item()
    quant_mean = (quant_part_test - quant_part_ref).abs().mean().item()

    log(f"overall  max_err={max_err:.4f}  mean_err={mean_err:.4f}")
    log(f"dense    max_err={dense_max:.6f}  mean_err={dense_mean:.6f}")
    log(f"quant    max_err={quant_max:.4f}  mean_err={quant_mean:.4f}")

    # dense 半部分应该非常精确（纯 NCCL）
    assert dense_max < 1e-4, \
        f"Dense half should be exact (NCCL), got max err {dense_max}"

    # 量化半部分容忍量化误差
    rtol = 0.03
    atol = 0.5
    torch.testing.assert_close(
        test_t, ref,
        rtol=rtol, atol=atol,
        msg=lambda msg: (
            f"Half-dense strategy allreduce mismatch!\n"
            f"  dense max err: {dense_max:.6f}\n"
            f"  quant max err: {quant_max:.4f}\n"
            f"  {msg}"
        ),
    )
    log("PASSED: half-dense correctness")
    set_strategy_on_pg(None)


def test_all_dense_strategy():
    """
    全 dense 策略：所有行走 NCCL，结果应与 all_gather 参考完全一致。
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    row_size = 32
    num_rows = 64
    total_elements = num_rows * row_size

    set_strategy_on_pg(make_all_dense_strategy(row_size=row_size))

    torch.manual_seed(123 + rank)
    tensor = torch.randn(total_elements, dtype=torch.float32, device=device)

    ref = compute_dense_reference(tensor, world_size)

    test_t = tensor.clone()
    dist.all_reduce(test_t, op=dist.ReduceOp.SUM)

    max_err = (test_t - ref).abs().max().item()
    log(f"all-dense max error: {max_err:.6f}")

    # 全 dense 等价于 NCCL，应该严格一致
    torch.testing.assert_close(test_t, ref, rtol=1e-5, atol=1e-5)
    log("PASSED: all-dense strategy")
    set_strategy_on_pg(None)


def test_full_quantization_fallback():
    """
    FullQuantizationStrategy.isActive()=False → 走回全量化路径。
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    from bitscom._lowbit_c import FullQuantizationStrategy

    set_strategy_on_pg(FullQuantizationStrategy())
    log(f"strategy name: {get_strategy_name_from_pg()}")

    row_size = 32
    num_rows = 64
    total_elements = num_rows * row_size

    torch.manual_seed(77 + rank)
    tensor = torch.randn(total_elements, dtype=torch.float32, device=device)

    ref = compute_dense_reference(tensor, world_size)

    test_t = tensor.clone()
    dist.all_reduce(test_t, op=dist.ReduceOp.SUM)

    torch.testing.assert_close(test_t, ref, rtol=0.03, atol=0.5)
    log("PASSED: full-quantization fallback")
    set_strategy_on_pg(None)


def test_strategy_clear_restore():
    """
    验证策略的 set → clear → re-set 生命周期。
    """
    # Set
    set_strategy_on_pg(make_all_dense_strategy(row_size=32))
    assert get_strategy_name_from_pg() == "all_dense"

    # Clear
    set_strategy_on_pg(None)
    assert get_strategy_name_from_pg() == "none", \
        "strategy should be 'none' after clearing"

    # Re-set
    set_strategy_on_pg(make_half_dense_strategy(row_size=32))
    assert get_strategy_name_from_pg() == "half_dense"
    log(f"restored strategy: {get_strategy_name_from_pg()}")

    set_strategy_on_pg(None)
    log("PASSED: strategy clear/restore lifecycle")


def test_partition_coverage():
    """
    验证 partition() 返回的 segments 完整覆盖整个 tensor。
    """
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    strategy = make_half_dense_strategy(row_size=64)
    flat = torch.randn(64 * 16, device=device)

    segments = strategy.partition(flat)

    total = sum(s.numel for s in segments)
    assert total == flat.numel(), \
        f"partition coverage: {total} != {flat.numel()}"

    # 检查 offset 连续
    pos = 0
    for s in segments:
        assert s.offset == pos, \
            f"segment offset {s.offset} != expected {pos}"
        pos += s.numel

    log(f"segments: {segments}")
    log("PASSED: partition coverage check")


def make_sparse_drop_strategy(row_size: int, keep_ratio: float):
    """
    稀疏化策略：随机丢掉一部分行（drop=True，恢复为 0），
    剩余行全部走量化。
    """
    from bitscom._lowbit_c import TensorPartitionStrategy, QuantizationSegment

    class SparseDropStrategy(TensorPartitionStrategy):
        def __init__(self, row_size, keep_ratio):
            super().__init__()
            self._row_size = row_size
            self._keep_ratio = keep_ratio
            self._drop_rows = set()

        def prepare(self, tensors, rank, world_size, comm_ptr, stream_ptr):
            # 用确定性方式选 dropout 行（基于 rank + seed 保证所有 rank 一致）
            flat = tensors[0]
            num_rows = flat.numel() // self._row_size
            torch.manual_seed(42)
            mask = torch.rand(num_rows) < (1.0 - self._keep_ratio)
            self._drop_rows = set(mask.nonzero(as_tuple=True)[0].tolist())
            return True

        def partition(self, flat_tensor):
            num_rows = flat_tensor.numel() // self._row_size
            segments = []
            for r in range(num_rows):
                seg = QuantizationSegment()
                seg.offset = r * self._row_size
                seg.numel = self._row_size
                if r in self._drop_rows:
                    seg.drop = True
                    seg.quantize = False
                else:
                    seg.drop = False
                    seg.quantize = True
                segments.append(seg)
            return segments

        def name(self):
            return "sparse_drop"

        def is_active(self):
            return True

    return SparseDropStrategy(row_size, keep_ratio)


def test_sparse_drop_strategy():
    """
    稀疏化 + 量化：随机丢掉 40% 的行（恢复为 0），剩余 60% 走量化。
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    row_size = 32
    num_rows = 128
    total_elements = num_rows * row_size

    strategy = make_sparse_drop_strategy(row_size=row_size, keep_ratio=0.6)
    set_strategy_on_pg(strategy)

    torch.manual_seed(99 + rank)
    tensor = torch.randn(total_elements, dtype=torch.float32, device=device)

    # 参考值：全量 dense allreduce
    ref = compute_dense_reference(tensor, world_size)

    # 稀疏+量化 allreduce
    test_t = tensor.clone()
    dist.all_reduce(test_t, op=dist.ReduceOp.SUM)

    # 验证：drop 行应该为 0
    flat = tensor.view(num_rows, row_size)
    test_flat = test_t.view(num_rows, row_size)
    ref_flat = ref.view(num_rows, row_size)

    drop_rows = strategy._drop_rows
    keep_rows = set(range(num_rows)) - drop_rows

    if drop_rows:
        drop_r = next(iter(drop_rows))
        drop_val = test_flat[drop_r].abs().max().item()
        log(f"sample drop row {drop_r}: max abs = {drop_val:.6f} (should be ~0)")
        assert drop_val < 1e-5, \
            f"Dropped row {drop_r} should be zero, got max abs {drop_val}"

    # keep 行应该接近参考值（量化容差内）
    if keep_rows:
        keep_r = next(iter(keep_rows))
        keep_max_err = (test_flat[keep_r] - ref_flat[keep_r]).abs().max().item()
        log(f"sample keep row {keep_r}: max err vs ref = {keep_max_err:.4f}")
        assert keep_max_err < 0.5, \
            f"Kept row {keep_r} error too large: {keep_max_err}"

    # 整体误差（drop 行预期为 0，且 ref 在 drop 行上有值）
    # 所以整体误差会比较大——只验证 drop 行正确置零即可
    for r in drop_rows:
        assert test_flat[r].abs().max().item() < 1e-5, \
            f"Dropped row {r} not zero! max={test_flat[r].abs().max().item()}"

    log(f"drop rows: {len(drop_rows)}, keep rows: {len(keep_rows)}")
    log("PASSED: sparse drop strategy")

    set_strategy_on_pg(None)


# ================================================================
#  ARC-TOP-K 策略 — pre-prepare 方式（避免 prepare() 里递归调 allreduce）
# ================================================================

def make_arctopk_strategy(row_size: int, important_rows: set,
                           important_quantize: bool,
                           non_important_mode: str):
    """
    通用 ARC-TOP-K 分区策略。

    Args:
        row_size:          每行元素数
        important_rows:    重要行索引集合
        important_quantize: True=重要行量化, False=重要行 dense
        non_important_mode: "quantize" | "dense" | "drop"
    """
    from bitscom._lowbit_c import TensorPartitionStrategy, QuantizationSegment

    class ArcTopkStrategy(TensorPartitionStrategy):
        def __init__(self, row_size, important_rows, important_quantize,
                     non_important_mode):
            super().__init__()
            self._row_size = row_size
            self._important_rows = important_rows
            self._important_quantize = important_quantize
            self._non_important_mode = non_important_mode

        def prepare(self, tensors, rank, world_size, comm_ptr, stream_ptr):
            # ARC-TOP-K priority exchange 已在外部做完，这里无需操作
            return True

        def partition(self, flat_tensor):
            num_rows = flat_tensor.numel() // self._row_size
            segments = []
            for r in range(num_rows):
                seg = QuantizationSegment()
                seg.offset = r * self._row_size
                seg.numel = self._row_size

                if r in self._important_rows:
                    seg.quantize = self._important_quantize
                    seg.drop = False
                else:
                    if self._non_important_mode == "drop":
                        seg.drop = True
                        seg.quantize = False
                    elif self._non_important_mode == "dense":
                        seg.drop = False
                        seg.quantize = False
                    else:  # "quantize"
                        seg.drop = False
                        seg.quantize = True

                segments.append(seg)
            return segments

        def name(self):
            return (f"arctopk(imp={'q' if self._important_quantize else 'd'},"
                    f"ni={self._non_important_mode})")

        def is_active(self):
            return True

    return ArcTopkStrategy(row_size, important_rows, important_quantize,
                           non_important_mode)


def _select_topk_rows(tensor: torch.Tensor, num_rows: int, row_size: int,
                      topk_ratio: float) -> set:
    """
    计算全局 top-k 重要行：算每行 L2 norm → allreduce 求和 → 选 top-k。
    在设策略前调用，避免策略递归。
    """
    world_size = dist.get_world_size()
    # 每行 norm 作为 priority
    priorities = tensor.view(num_rows, row_size).norm(dim=1)
    dist.all_reduce(priorities, op=dist.ReduceOp.SUM)
    k = max(1, int(num_rows * topk_ratio))
    _, indices = torch.topk(priorities, k)
    return set(indices.tolist())


# ================================================================
#  四种配置的 ARC-TOP-K 测试
# ================================================================

def _run_arctopk_test(row_size, num_rows, topk_ratio,
                       important_quantize, non_important_mode,
                       label, rank, world_size, device):
    """通用 ARC-TOP-K 测试框架。"""
    total_elements = num_rows * row_size
    assert total_elements % world_size == 0

    torch.manual_seed(42 + rank)
    tensor = torch.randn(total_elements, dtype=torch.float32, device=device)

    # 1) 策略外做 priority exchange + top-k 选择
    important_rows = _select_topk_rows(tensor, num_rows, row_size, topk_ratio)

    # 2) 设策略
    strategy = make_arctopk_strategy(
        row_size, important_rows, important_quantize, non_important_mode)
    set_strategy_on_pg(strategy)
    log(f"[{label}] {strategy.name()} topk={len(important_rows)}/{num_rows}")

    # 3) 参考值
    ref = compute_dense_reference(tensor, world_size)

    # 4) allreduce
    test_t = tensor.clone()
    dist.all_reduce(test_t, op=dist.ReduceOp.SUM)

    # 5) 分行验证
    ref_flat = ref.view(num_rows, row_size)
    test_flat = test_t.view(num_rows, row_size)
    non_important_rows = set(range(num_rows)) - important_rows

    errors = []
    for r in range(num_rows):
        err = (test_flat[r] - ref_flat[r]).abs().max().item()
        errors.append(err)

    # --- 重要行验证 ---
    if important_rows:
        imp_errs = [errors[r] for r in important_rows]
        imp_max = max(imp_errs)
        log(f"[{label}] important rows: max_err={imp_max:.6f}  "
            f"mean_err={sum(imp_errs)/len(imp_errs):.6f}")
        if important_quantize:
            # 量化 → 有误差容忍
            assert imp_max < 0.5, \
                f"Important (quantized) error too large: {imp_max}"
        else:
            # dense → 应该非常精确
            assert imp_max < 1e-4, \
                f"Important (dense) error too large: {imp_max}"

    # --- 非重要行验证 ---
    if non_important_rows:
        ni_errs = [errors[r] for r in non_important_rows]
        ni_max = max(ni_errs)
        log(f"[{label}] non-important rows: max_err={ni_max:.6f}  "
            f"mean_err={sum(ni_errs)/len(ni_errs):.6f}")
        if non_important_mode == "drop":
            # 应为 0
            assert ni_max < 1e-5, \
                f"Non-important (dropped) should be zero: {ni_max}"
        elif non_important_mode == "dense":
            assert ni_max < 1e-4, \
                f"Non-important (dense) error too large: {ni_max}"
        else:  # quantize
            assert ni_max < 0.5, \
                f"Non-important (quantized) error too large: {ni_max}"

    log(f"[{label}] PASSED")
    set_strategy_on_pg(None)


def test_arctopk_v1_important_dense_ni_quantize():
    """
    配置 1: 重要行 → dense, 非重要行 → quantize（经典 ARC-TOP-K）
    """
    _run_arctopk_test(
        row_size=32, num_rows=128, topk_ratio=0.2,
        important_quantize=False, non_important_mode="quantize",
        label="v1:imp=dense,ni=quantize",
        rank=dist.get_rank(), world_size=dist.get_world_size(),
        device=torch.device(f"cuda:{dist.get_rank()}"))


def test_arctopk_v2_important_dense_ni_drop():
    """
    配置 2: 重要行 → dense, 非重要行 → drop（纯稀疏化: ARC-TOP-K 但不量化）
    """
    _run_arctopk_test(
        row_size=32, num_rows=128, topk_ratio=0.2,
        important_quantize=False, non_important_mode="drop",
        label="v2:imp=dense,ni=drop",
        rank=dist.get_rank(), world_size=dist.get_world_size(),
        device=torch.device(f"cuda:{dist.get_rank()}"))


def test_arctopk_v3_important_quantize_ni_drop():
    """
    配置 3: 重要行 → quantize, 非重要行 → drop
            （重要行也量化，非重要行直接丢弃）
    """
    _run_arctopk_test(
        row_size=32, num_rows=128, topk_ratio=0.2,
        important_quantize=True, non_important_mode="drop",
        label="v3:imp=quantize,ni=drop",
        rank=dist.get_rank(), world_size=dist.get_world_size(),
        device=torch.device(f"cuda:{dist.get_rank()}"))


def test_arctopk_v4_important_quantize_ni_dense():
    """
    配置 4: 重要行 → quantize, 非重要行 → dense
            （反直觉但合法的配置: 重要行量化、非重要行精确）
    """
    _run_arctopk_test(
        row_size=32, num_rows=128, topk_ratio=0.2,
        important_quantize=True, non_important_mode="dense",
        label="v4:imp=quantize,ni=dense",
        rank=dist.get_rank(), world_size=dist.get_world_size(),
        device=torch.device(f"cuda:{dist.get_rank()}"))


def test_arctopk_v5_large():
    """
    配置 5: 大 tensor (256行×256列), topk_ratio=0.05
            huge 行 dense，其余量化
    """
    _run_arctopk_test(
        row_size=256, num_rows=256, topk_ratio=0.05,
        important_quantize=False, non_important_mode="quantize",
        label="v5:large imp=dense,ni=quantize",
        rank=dist.get_rank(), world_size=dist.get_world_size(),
        device=torch.device(f"cuda:{dist.get_rank()}"))


# ================================================================
#  main
# ================================================================

def main():
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if not torch.cuda.is_available():
        print("CUDA not available, skipping integration tests.")
        return

    torch.cuda.set_device(rank)

    # 1) 注册 lowbit backend
    import bitscom
    bitscom.init(bitwidth=4, error_feedback=False)

    # 2) 初始化 process group（使用 lowbit backend）
    dist.init_process_group(backend="lowbit")

    log(f"initialized world_size={world_size}")

    # 3) 运行测试
    test_partition_coverage()
    test_strategy_clear_restore()
    test_all_dense_strategy()
    test_full_quantization_fallback()
    test_half_dense_correctness()
    test_sparse_drop_strategy()
    test_arctopk_v1_important_dense_ni_quantize()
    test_arctopk_v2_important_dense_ni_drop()
    test_arctopk_v3_important_quantize_ni_drop()
    test_arctopk_v4_important_quantize_ni_dense()
    test_arctopk_v5_large()

    # 4) 清理
    dist.destroy_process_group()

    if rank == 0:
        print("\n=== All row partition strategy tests PASSED ===")


if __name__ == "__main__":
    main()
