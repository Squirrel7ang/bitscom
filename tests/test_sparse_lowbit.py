"""
测试稀疏化 ARC-Top-K 量化 allreduce 逻辑的正确性。

本文件为**单进程纯 Python 参考测试**——在 CPU 上用纯 PyTorch 模拟
pack→alltoall→sum→allgather 流水线，不需要 NCCL、不需要 GPU、不需要
编译 C++ extension。适用于 CI 环境或快速逻辑验证。

如需真正的分布式 NCCL 多卡测试，请使用:
    test_sparse_lowbit_distributed.py

测试覆盖：
1. 手搓例子：显式打印稀疏化量化的中间步骤，用于人工验证
2. 随机例子：判断稀疏化量化张量和全精度 allreduce 结果的距离
3. 不同 priority / non-priority 通信模式组合
4. 矩阵维度对齐验证

运行方式
--------
# 单进程运行全部（无需 GPU / NCCL / C++ extension）
pytest tests/test_sparse_lowbit.py -v -s

# 只跑手搓例子（查看详细输出）
pytest tests/test_sparse_lowbit.py::TestSparseLowbitHandcrafted -v -s

# 只跑随机例子
pytest tests/test_sparse_lowbit.py::TestSparseLowbitRandom -v -s

# 只跑矩阵维度测试
pytest tests/test_sparse_lowbit.py::TestMatrixAlignment -v -s

# 只跑 mode 组合测试
pytest tests/test_sparse_lowbit.py::TestModeCombinations -v -s

# 代码中量化函数自带 CPU fallback 路径，无需 GPU 即可运行：
#   quantize_tensor:          if is_cuda and _HAS_CUDA_KERNELS → fused CUDA, else CPU pytorch
#   quantize_pack_tensor_blockwise: 同上
#   unpack_dequantize_tensor_blockwise: 同上
#   pack_lowbit / unpack_lowbit:       同上
"""

import math
from typing import Tuple

import pytest
import torch

# 在单进程中模拟 bitscom 量化函数（不依赖 C++ extension）
from bitscom.quantization import (
    DEFAULT_BLOCK_SIZE,
    pack_lowbit,
    quantize_pack_tensor_blockwise,
    unpack_dequantize_tensor_blockwise,
    unpack_lowbit,
)


# ============================================================
# 辅助函数：纯 Python 实现 ARC-Top-K + 量化 allreduce 参考逻辑
# ============================================================

def _cal_max_factor(size: int) -> int:
    """找到 size 的最大 2 的幂因子 n，使得 n * m = size 且 n^2 < size"""
    factor = 1
    while size % factor == 0 and size // factor > factor:
        factor *= 2
    return factor


def _quantized_allreduce_sim(
    inputs: list[torch.Tensor],
    bitwidth: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> list[torch.Tensor]:
    """
    模拟量化 allreduce: pack → alltoall → unpack/sum → repack → allgather → unpack。
    返回每个 rank 的 reduced 结果。
    """
    world_size = len(inputs)
    flats = [t.contiguous().view(-1).to(torch.float32) for t in inputs]
    numel = int(flats[0].numel())
    if numel == 0:
        return [flat.clone() for flat in flats]

    pad = (world_size - (numel % world_size)) % world_size
    if pad:
        flats = [torch.cat([f, torch.zeros(pad, dtype=f.dtype)], dim=0) for f in flats]

    shard_len = flats[0].numel() // world_size

    # ---- Phase 1: pack + alltoall ----
    packed_per_rank: list[list[torch.Tensor]] = []
    scales_per_rank: list[list[torch.Tensor]] = []
    for flat in flats:
        shards = list(flat.split(shard_len))
        pk = []
        sc = []
        for shard in shards:
            p, s, _ = quantize_pack_tensor_blockwise(
                shard, bitwidth, block_size=block_size, stochastic_rounding=False
            )
            pk.append(p)
            sc.append(s)
        packed_per_rank.append(pk)
        scales_per_rank.append(sc)

    # ---- Phase 2: unpack received + sum + repack ----
    reduced_packed = []
    reduced_scales = []
    for dst in range(world_size):
        local_sum = torch.zeros(shard_len, dtype=torch.float32)
        for src in range(world_size):
            fp = unpack_dequantize_tensor_blockwise(
                packed_per_rank[src][dst],
                scales_per_rank[src][dst],
                bitwidth,
                shard_len,
                block_size=block_size,
                dtype=torch.float32,
            )
            local_sum.add_(fp)
        p, s, _ = quantize_pack_tensor_blockwise(
            local_sum, bitwidth, block_size=block_size, stochastic_rounding=False
        )
        reduced_packed.append(p)
        reduced_scales.append(s)

    # ---- Phase 3: allgather + unpack ----
    out_shards = []
    for r in range(world_size):
        fp = unpack_dequantize_tensor_blockwise(
            reduced_packed[r],
            reduced_scales[r],
            bitwidth,
            shard_len,
            block_size=block_size,
            dtype=torch.float32,
        )
        out_shards.append(fp)

    restored = torch.cat(out_shards, dim=0)[:numel]
    return [restored.clone() for _ in range(world_size)]


def _sparse_allreduce_sim(
    inputs: list[torch.Tensor],
    compression_ratio: float,
    projection_rank: int,
    priority_mode: str,          # "full" | "quantize"
    priority_bitwidth: int,
    non_priority_mode: str,      # "full" | "quantize" | "discard"
    non_priority_bitwidth: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    verbose: bool = False,
) -> list[torch.Tensor]:
    """
    模拟 ARC-Top-K 稀疏化量化 allreduce。
    在单进程中用多个 tensor 代表各 rank 的本地数据。
    """
    world_size = len(inputs)
    flats = [t.contiguous().view(-1).to(torch.float32) for t in inputs]
    d = int(flats[0].numel())
    assert all(f.numel() == d for f in flats), "all ranks must have same numel"
    if d == 0:
        return [f.clone() for f in flats]

    # ---- 1. Reshape to n×m ----
    n = _cal_max_factor(d)
    m = d // n
    Gs = [f.view(n, m) for f in flats]
    if verbose:
        print(f"[sparse_sim] d={d}, n={n}, m={m}")
        for rank_idx, G in enumerate(Gs):
            print(f"[sparse_sim] rank={rank_idx} G[:3,:3] =\n{G[:3, :3]}")

    # ---- 2. Random projection (same V for all ranks in simulation) ----
    r = projection_rank
    V = torch.randn(m, r, dtype=torch.float32)
    Ps = [torch.matmul(G, V) / math.sqrt(float(r)) for G in Gs]
    if verbose:
        for rank_idx, P in enumerate(Ps):
            print(f"[sparse_sim] rank={rank_idx} P[:3,:] =\n{P[:3, :]}")

    # ---- 3. Global priority = mean(P) ----
    P_global = sum(Ps) / float(world_size)  # allreduce avg
    if verbose:
        print(f"[sparse_sim] P_global[:3,:] =\n{P_global[:3, :]}")

    # ---- 4. Score = row-wise squared L2 norm ----
    score = (P_global * P_global).sum(dim=1)  # shape [n]
    K = max(1, int(n * compression_ratio))
    _, indices = torch.topk(score, K)
    if verbose:
        print(f"[sparse_sim] score[:10] = {score[:10]}")
        print(f"[sparse_sim] K={K}, priority_indices = {indices.tolist()}")

    # ---- 5. Complement indices ----
    mask = torch.zeros(n, dtype=torch.bool)
    mask[indices] = True
    comp_indices = torch.arange(n)[~mask]

    # ---- 6-8. Communicate priority / non-priority rows ----
    reduced_priority_per_rank = []
    reduced_non_priority_per_rank = []

    for rank_idx in range(world_size):
        pri_rows = Gs[rank_idx][indices, :]      # K×m
        nonpri_rows = Gs[rank_idx][comp_indices, :]  # (n-K)×m

        # Priority rows
        if priority_mode == "full":
            all_pri = torch.stack([Gs[r][indices, :] for r in range(world_size)])
            reduced_pri = all_pri.sum(dim=0)
        elif priority_mode == "quantize":
            reduced_pri_flat = _quantized_allreduce_sim(
                [pri_rows.contiguous().view(-1) for _ in range(world_size)],
                bitwidth=priority_bitwidth,
                block_size=block_size,
            )[0]
            reduced_pri = reduced_pri_flat.view(K, m)
        else:  # "discard"
            reduced_pri = torch.zeros_like(pri_rows)

        # Non-priority rows
        nonK = n - K
        if nonK > 0:
            if non_priority_mode == "full":
                all_nonpri = torch.stack([Gs[r][comp_indices, :] for r in range(world_size)])
                reduced_nonpri = all_nonpri.sum(dim=0)
            elif non_priority_mode == "quantize":
                reduced_nonpri_flat = _quantized_allreduce_sim(
                    [nonpri_rows.contiguous().view(-1) for _ in range(world_size)],
                    bitwidth=non_priority_bitwidth,
                    block_size=block_size,
                )[0]
                reduced_nonpri = reduced_nonpri_flat.view(nonK, m)
            else:
                reduced_nonpri = torch.zeros_like(nonpri_rows)
        else:
            reduced_nonpri = torch.zeros(0, m)

        # ---- 9. Reconstruct ----
        result = torch.zeros(n, m)
        result[indices] = reduced_pri
        if nonK > 0:
            result[comp_indices] = reduced_nonpri

        reduced_priority_per_rank.append(reduced_pri)
        reduced_non_priority_per_rank.append(reduced_nonpri)

        result_flat = result.view(-1)
        flats[rank_idx] = result_flat

    if verbose:
        for rank_idx in range(world_size):
            print(f"[sparse_sim] rank={rank_idx} reduced_priority[:2,:] =\n{reduced_priority_per_rank[rank_idx][:2, :]}")
            if n - K > 0:
                print(f"[sparse_sim] rank={rank_idx} reduced_non_priority[:2,:] =\n{reduced_non_priority_per_rank[rank_idx][:2, :]}")
            print(f"[sparse_sim] rank={rank_idx} final flat[:8] = {flats[rank_idx][:8]}")

    return flats


# ============================================================
# 手搓例子测试 — 打印中间步骤供人工验证
# ============================================================

class TestSparseLowbitHandcrafted:
    """手搓小例子，显式输出稀疏化量化过程的中间结果。"""

    @pytest.mark.parametrize("seed", [42])
    def test_handcrafted_small_2rank(self, seed):
        """
        2 rank, 8 element tensor (→ 2×4 matrix), priority=full, non_priority=quantize_int4
        手动跟踪每一步的数值变化。
        """
        torch.manual_seed(seed)
        d = 8
        n = _cal_max_factor(d)  # 2
        m = d // n               # 4
        assert n == 2 and m == 4

        rank0 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float32)
        rank1 = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=torch.float32)

        print(f"\n=== Handcrafted Test: 2-rank, {d} elements ===")
        print(f"rank0 = {rank0}")
        print(f"rank1 = {rank1}")
        print(f"n(rows)={n}, m(cols)={m}")
        print(f"G0 (rank0 as {n}×{m}):\n{rank0.view(n, m)}")
        print(f"G1 (rank1 as {n}×{m}):\n{rank1.view(n, m)}")

        # 全精度 allreduce 作为 ground truth
        true_sum = rank0 + rank1
        print(f"True allreduce SUM = {true_sum}")

        result = _sparse_allreduce_sim(
            [rank0, rank1],
            compression_ratio=0.5,       # K = 1 (top 1 row)
            projection_rank=4,
            priority_mode="full",
            priority_bitwidth=4,
            non_priority_mode="quantize",
            non_priority_bitwidth=4,
            verbose=True,
        )

        for rank_idx, r in enumerate(result):
            print(f"Result rank={rank_idx}: {r}")
            err = (r - true_sum).abs().max().item()
            print(f"  max_abs_error vs true SUM: {err:.6f}")

        # 误差应该在可接受范围内（量化引入了误差）
        for r in result:
            assert (r - true_sum).abs().max().item() < 1.0, "Error too large"

    @pytest.mark.parametrize("seed", [99])
    def test_handcrafted_discard_non_priority(self, seed):
        """priority=full, non_priority=discard: 验证非重要行被置零。"""
        torch.manual_seed(seed)
        d = 16
        n = _cal_max_factor(d)
        m = d // n
        # n=4, m=4

        rank0 = torch.arange(1, 17, dtype=torch.float32)
        rank1 = torch.arange(0.5, 8.5, 0.5, dtype=torch.float32)

        print(f"\n=== Discard Non-Priority Test ===")
        print(f"n={n}, m={m}")

        result = _sparse_allreduce_sim(
            [rank0, rank1],
            compression_ratio=0.25,       # K = 1 (top 1 row out of 4)
            projection_rank=4,
            priority_mode="full",
            priority_bitwidth=4,
            non_priority_mode="discard",
            non_priority_bitwidth=4,
            verbose=True,
        )

        true_sum = rank0 + rank1
        for rank_idx, r in enumerate(result):
            G_result = r.view(n, m)
            G_true = true_sum.view(n, m)

            # 找到 priority 行和 non-priority 行
            V = torch.randn(m, 4)
            P_global = (torch.matmul(rank0.view(n, m), V) +
                        torch.matmul(rank1.view(n, m), V)) / (2.0 * 2.0)
            score = (P_global * P_global).sum(dim=1)
            K = max(1, int(n * 0.25))
            _, indices = torch.topk(score, K)
            mask = torch.zeros(n, dtype=torch.bool)
            mask[indices] = True

            print(f"Priority indices: {indices.tolist()}")
            print(f"G_result:\n{G_result}")
            print(f"G_true:\n{G_true}")

            # Priority rows: 应该接近 true sum
            pri_err = (G_result[indices] - G_true[indices]).abs().max().item()
            print(f"Priority rows max error: {pri_err:.6f}")
            assert pri_err < 0.1, f"Priority row error too large: {pri_err}"

            # Non-priority rows: 应该为 0（discard）
            comp_indices = torch.arange(n)[~mask]
            if len(comp_indices) > 0:
                nonpri_max = G_result[comp_indices].abs().max().item()
                print(f"Non-priority rows max abs (should be 0): {nonpri_max:.6f}")
                assert nonpri_max < 1e-6, f"Non-priority rows should be zero, got {nonpri_max}"


# ============================================================
# 随机例子测试 — 比较稀疏化结果与全精度 allreduce
# ============================================================

class TestSparseLowbitRandom:
    """随机生成数据，验证稀疏化量化结果与全精度 allreduce 的接近程度。"""

    @pytest.mark.parametrize("case", [
        {"numel": 128, "world_size": 2, "compression_ratio": 0.5, "pri": "full", "nonpri": "quantize"},
        {"numel": 256, "world_size": 4, "compression_ratio": 0.25, "pri": "full", "nonpri": "quantize"},
        {"numel": 512, "world_size": 2, "compression_ratio": 0.5, "pri": "quantize", "nonpri": "quantize"},
        {"numel": 128, "world_size": 2, "compression_ratio": 0.5, "pri": "quantize", "nonpri": "discard"},
        {"numel": 256, "world_size": 4, "compression_ratio": 0.25, "pri": "full", "nonpri": "discard"},
    ])
    def test_random_closeness(self, case):
        """
        随机生成多 rank 数据，对比 ARC-Top-K sparse allreduce 结果与全精度 allreduce 的误差。
        """
        torch.manual_seed(12345)
        numel = case["numel"]
        world_size = case["world_size"]
        ratio = case["compression_ratio"]
        pri = case["pri"]
        nonpri = case["nonpri"]

        # 生成各 rank 的随机数据
        inputs = []
        for r in range(world_size):
            t = torch.randn(numel, dtype=torch.float32) * 1.5 + r * 0.1
            inputs.append(t)

        # 全精度 allreduce SUM
        true_sum = sum(inputs)

        # 稀疏化量化 allreduce
        result = _sparse_allreduce_sim(
            inputs,
            compression_ratio=ratio,
            projection_rank=8,
            priority_mode=pri,
            priority_bitwidth=4,
            non_priority_mode=nonpri,
            non_priority_bitwidth=4,
            verbose=False,
        )

        for rank_idx, r in enumerate(result):
            abs_err = (r - true_sum).abs()
            rel_err = abs_err / (true_sum.abs() + 1e-8)

            max_abs_err = abs_err.max().item()
            mean_abs_err = abs_err.mean().item()
            max_rel_err = rel_err.max().item()
            mean_rel_err = rel_err.mean().item()

            print(f"\n[Random Test] numel={numel}, ws={world_size}, ratio={ratio}, "
                  f"pri={pri}, nonpri={nonpri}")
            print(f"  rank={rank_idx}: max_abs_err={max_abs_err:.6f}, "
                  f"mean_abs_err={mean_abs_err:.6f}, "
                  f"max_rel_err={max_rel_err:.4f}, "
                  f"mean_rel_err={mean_rel_err:.4f}")

            # 基本检查：误差不超过合理范围
            # discard 模式会导致较大的误差（因为非重要行被丢弃）
            if nonpri == "discard":
                # discard 模式下，非重要行全被丢弃，误差取决于这些行的值
                # 这里只检查不会 NaN/Inf
                assert torch.isfinite(r).all(), "Result contains NaN/Inf"
            else:
                # 全精度或量化模式，相对误差应可控
                assert max_rel_err < 5.0, f"Relative error too large: {max_rel_err}"


# ============================================================
# 矩阵维度对齐测试
# ============================================================

class TestMatrixAlignment:
    """验证 ARC-Top-K 中各矩阵维度匹配。"""

    def test_all_dimensions_match(self):
        """遍历多种 d 和 r，逐矩阵检查维度。"""
        test_configs = [
            (128, 4),
            (256, 8),
            (512, 16),
            (1024, 4),
            (100, 4),   # d=100: calMaxFactor→4, n=4, m=25
            (97, 4),     # d=97 (prime): calMaxFactor→1, n=1, m=97
        ]

        for d, r in test_configs:
            n = _cal_max_factor(d)
            m = d // n
            assert n * m == d, f"n*m != d: {n}*{m} != {d}"

            G = torch.randn(n, m)
            V = torch.randn(m, r)

            # G: n×m, V: m×r → P: n×r
            P = torch.matmul(G, V) / math.sqrt(float(r))
            assert P.shape == (n, r), f"P shape mismatch: {P.shape} vs ({n}, {r})"

            # P: n×r, P^T: r×n → P@P^T: n×n, score: n
            score_full = torch.diag(torch.matmul(P, P.t()))
            score_opt = (P * P).sum(dim=1)
            assert score_full.shape == (n,), f"score shape: {score_full.shape}"
            assert score_opt.shape == (n,), f"score_opt shape: {score_opt.shape}"

            # 验证优化前后的 score 一致（相差应在浮点精度内）
            diff = (score_full - score_opt).abs().max().item()
            assert diff < 1e-5, f"score mismatch at d={d}, r={r}: diff={diff}"

            # Top-K
            K = max(1, int(n * 0.5))
            _, indices = torch.topk(score_opt, K)
            assert indices.shape == (K,), f"indices shape: {indices.shape} vs ({K},)"

            # 提取行
            pri_rows = G[indices]
            assert pri_rows.shape == (K, m), f"pri_rows shape: {pri_rows.shape} vs ({K}, {m})"

            # 互补索引
            mask = torch.zeros(n, dtype=torch.bool)
            mask[indices] = True
            comp = torch.arange(n)[~mask]
            assert comp.numel() == n - K, f"comp numel: {comp.numel()} vs {n-K}"
            nonpri_rows = G[comp]
            assert nonpri_rows.shape[0] == n - K, f"nonpri_rows rows: {nonpri_rows.shape[0]} vs {n-K}"
            assert nonpri_rows.shape[1] == m, f"nonpri_rows cols: {nonpri_rows.shape[1]} vs {m}"

            # 重建
            result = torch.zeros(n, m)
            result[indices] = pri_rows
            if n - K > 0:
                result[comp] = nonpri_rows
            # 重建结果应与原始 G 一致（因为我们只是提取再放回，没有做通信）
            assert (result - G).abs().max().item() < 1e-6, "Reconstruction mismatch"

            print(f"  d={d:5d}, n={n:4d}, m={m:4d}, r={r:2d}, K={K}: ✓ all dims match, score opt consistent")


# ============================================================
# 不同 mode 组合测试
# ============================================================

class TestModeCombinations:
    """测试所有 priority / non-priority mode 组合不会崩溃。"""

    modes = ["full", "quantize", "discard"]

    @pytest.mark.parametrize("pri_mode", modes)
    @pytest.mark.parametrize("nonpri_mode", modes)
    def test_mode_combo(self, pri_mode, nonpri_mode):
        """逐一测试 3×3 = 9 种 mode 组合。"""
        torch.manual_seed(42)
        d = 128
        world_size = 2
        inputs = [torch.randn(d) * 1.2 + r * 0.1 for r in range(world_size)]

        try:
            result = _sparse_allreduce_sim(
                inputs,
                compression_ratio=0.5,
                projection_rank=4,
                priority_mode=pri_mode,
                priority_bitwidth=4,
                non_priority_mode=nonpri_mode,
                non_priority_bitwidth=4,
                verbose=False,
            )
        except Exception as e:
            pytest.fail(f"Mode combo pri={pri_mode}, nonpri={nonpri_mode} failed: {e}")

        for r in result:
            assert r.shape == inputs[0].shape, "Shape mismatch"
            assert torch.isfinite(r).all(), f"Non-finite values in result for {pri_mode}/{nonpri_mode}"

        print(f"  pri={pri_mode:8s}, nonpri={nonpri_mode:8s}: ✓")
