"""
稀疏化 ARC-Top-K reduce_scatter 单进程逻辑测试。

本文件不需要 NCCL、GPU、C++ extension。
分布式 NCCL 测试请用: test_sparse_reduce_scatter_distributed.py

运行方式
--------
# 全部
pytest tests/test_sparse_reduce_scatter.py -v -s

# 只跑手搓例子
pytest tests/test_sparse_reduce_scatter.py::TestHandcraftedRS -v -s

# 只跑 mode 组合
pytest tests/test_sparse_reduce_scatter.py::TestRSModeCombos -v -s
"""

import math

import pytest
import torch

from bitscom.quantization import (
    quantize_pack_tensor_blockwise,
    unpack_dequantize_tensor_blockwise,
)


def _cal_max_factor(size: int) -> int:
    factor = 1
    while size % factor == 0 and size // factor > factor:
        factor *= 2
    while factor > 1 and size % factor != 0:
        factor //= 2
    return factor


class TestHandcraftedRS:
    """手搓 reduce_scatter 例子 — 验证 ARC 选行 + 重建正确。"""

    def test_arc_identifies_large_rows(self):
        """
        8×8 矩阵，其中行 0 和行 3 设为大值。
        ARC 应该准确选出这些行作为 priority。
        """
        d = 64
        n = _cal_max_factor(d)  # 8
        m = d // n               # 8

        # 手搓矩阵: 行 0 大正值，行 3 大负值
        data = torch.zeros(n, m)
        data[0] = torch.tensor([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
        data[1] = torch.randn(m) * 0.5
        data[2] = torch.randn(m) * 0.3
        data[3] = torch.tensor([-50.0, -51.0, -52.0, -53.0, -54.0, -55.0, -56.0, -57.0])
        for i in range(4, n):
            data[i] = torch.randn(m) * 0.2

        # ARC
        r = 4
        V = torch.randn(m, r, dtype=torch.float32)
        P = data @ V / math.sqrt(float(r))
        score = (P * P).sum(dim=1)
        K = max(1, int(n * 0.25))  # K = 2

        _, indices = torch.topk(score, K)

        print(f"\nScore per row: {score.tolist()}")
        print(f"Priority indices (K={K}): {indices.tolist()}")
        print(f"Expected: [0, 3] (rows with large values)")

        # 行 0 和行 3 应该被选中（score 最高）
        assert 0 in indices.tolist(), f"Row 0 (large values) not selected! indices={indices.tolist()}"
        assert 3 in indices.tolist(), f"Row 3 (large values) not selected! indices={indices.tolist()}"

        # 验证 score 最高的两行正是行 0 和行 3
        top2_rows = set(indices.tolist())
        assert top2_rows == {0, 3}, f"Expected top rows [0,3], got {top2_rows}"

        print("✓ ARC correctly identified priority rows")

    def test_reconstruct_splits_priority_and_nonpriority(self):
        """验证重建: priority 行来自 reduced_priority, non-priority 行置零。"""
        n, m = 8, 4
        d = n * m  # 32

        # 数据
        full = torch.zeros(n, m)
        full[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])    # priority
        full[3] = torch.tensor([5.0, 6.0, 7.0, 8.0])    # priority
        full[1] = torch.tensor([0.1, 0.2, 0.3, 0.4])    # non-priority
        full[2] = torch.tensor([0.5, 0.6, 0.7, 0.8])    # non-priority
        full[4] = torch.tensor([0.9, 1.0, 1.1, 1.2])    # non-priority
        full[5] = torch.tensor([1.3, 1.4, 1.5, 1.6])    # non-priority
        full[6] = torch.tensor([1.7, 1.8, 1.9, 2.0])    # non-priority
        full[7] = torch.tensor([2.1, 2.2, 2.3, 2.4])    # non-priority

        # 模拟 reduced 后的 priority 行 (sum of all ranks)
        reduced_pri = full[[0, 3]].clone()  # 全精度: 直接就是原始值
        reduced_nonpri = torch.zeros(6, m)   # discard 模式

        indices = torch.tensor([0, 3])
        comp = torch.tensor([1, 2, 4, 5, 6, 7])

        result = torch.zeros(n, m)
        result[indices] = reduced_pri
        result[comp] = reduced_nonpri

        print(f"\nResult matrix:\n{result}")
        print(f"Expected: rows 0,3 have original values; others are zeros")

        # 验证
        assert (result[0] - full[0]).abs().max().item() < 1e-6
        assert (result[3] - full[3]).abs().max().item() < 1e-6
        for i in [1, 2, 4, 5, 6, 7]:
            assert result[i].abs().max().item() < 1e-6, f"Row {i} should be zero!"

        # 取 world_size=2 时 rank 0 的 shard: 前 d/2 = 16 个 flat 元素 → rows 0,1,2,3
        # 其中 row 0 和 row 3 是 priority，row 1,2 是 non-priority
        world_size = 2
        obs = result.view(-1)[:d//world_size].view(4, 4)
        exp = full[:4]  # rows 0-3

        print(f"\nRank 0 output (rows 0-3):\n{obs}")
        print(f"Expected:\n{exp}")
        print(f"priority_indices={indices.tolist()}")

        assert (obs[0] - exp[0]).abs().max().item() < 1e-6, "Row 0 (priority) mismatch"
        assert (obs[3] - exp[3]).abs().max().item() < 1e-6, "Row 3 (priority) mismatch"
        assert obs[1:3].abs().max().item() < 1e-6, "Rows 1-2 (non-priority) should be zero"
        print("✓ Reconstruction correct")


class TestRSModeCombos:
    """所有 mode 组合不应崩溃且输出有限。"""

    @pytest.mark.parametrize("pri", [0, 1, 2])
    @pytest.mark.parametrize("nonpri", [0, 1, 2])
    def test_mode_combo_no_crash(self, pri, nonpri):
        torch.manual_seed(42)
        n, m = 8, 8
        full = torch.randn(n, m)
        # ARC
        r = 4
        V = torch.randn(m, r)
        P = full @ V / math.sqrt(float(r))
        score = (P * P).sum(dim=1)
        K = max(1, int(n * 0.5))
        _, idx = torch.topk(score, K)
        comp = torch.arange(n)[~torch.isin(torch.arange(n), idx)]
        # 通信模拟
        pri_rows = full[idx]
        nonpri_rows = full[comp] if comp.numel() > 0 else torch.zeros(0, m)
        # priority
        if pri == 0:
            rp = pri_rows
        elif pri == 1:
            p, s, _ = quantize_pack_tensor_blockwise(
                pri_rows.contiguous().view(-1), 4, stochastic_rounding=False)
            rp = unpack_dequantize_tensor_blockwise(
                p, s, 4, K*m, dtype=torch.float32).view(K, m)
        else:
            rp = torch.zeros_like(pri_rows)
        # non-priority
        if nonpri == 0:
            rnp = nonpri_rows
        elif nonpri == 1:
            nk = comp.numel()
            if nk > 0:
                p, s, _ = quantize_pack_tensor_blockwise(
                    nonpri_rows.contiguous().view(-1), 4, stochastic_rounding=False)
                rnp = unpack_dequantize_tensor_blockwise(
                    p, s, 4, nk*m, dtype=torch.float32).view(nk, m)
            else:
                rnp = nonpri_rows
        else:
            rnp = torch.zeros_like(nonpri_rows)
        assert torch.isfinite(rp).all(), f"pri={pri}: NaN"
        assert torch.isfinite(rnp).all(), f"nonpri={nonpri}: NaN"
        print(f"  pri={pri} nonpri={nonpri}: ✓")
