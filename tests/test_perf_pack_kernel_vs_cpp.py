"""基准：C++ pack/unpack（pack_ref/unpack_ref）vs 融合 CUDA kernel。

对 4MB (1024x1024) 和 16MB (2048x2048) 的 FP32 张量各量化 32 次，
比较两条路径的耗时，并顺带核对两者产出的 packed 字节是否一致。

运行：
    python tests/test_perf_pack_kernel_vs_cpp.py
"""

import time

import torch

from bitscom._lowbit_c import pack_ref, unpack_ref
from bitscom._lowbit_cuda import (
    blockwise_quantize_pack_cuda,
    blockwise_unpack_dequantize_cuda,
)

BLOCK_SIZE = 256
WARMUP = 5
ITERS = 32


def bench(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms / call


def check_equal(x, bitwidth):
    """核对 pack_ref 与 kernel 的 packed 字节是否一致，返回两者的产物。"""
    p_cpp, s_cpp = pack_ref(x, bitwidth, BLOCK_SIZE)
    p_k, s_k, numel = blockwise_quantize_pack_cuda(x, bitwidth, BLOCK_SIZE, False)
    same_len = p_cpp.numel() == p_k.numel()
    if same_len:
        diff = (p_cpp.view(torch.uint8) != p_k.view(torch.uint8)).sum().item()
    else:
        diff = -1
    smax = (s_cpp.float() - s_k.float()).abs().max().item() if s_cpp.numel() == s_k.numel() else float("nan")
    return (p_cpp, s_cpp), (p_k, s_k, numel), same_len, p_cpp.numel(), diff, smax


def run(size, bitwidth):
    n = size * size
    mib = n * 4 / 1024 / 1024
    x = torch.randn(size, size, device="cuda", dtype=torch.float32)

    # ---------- pack ----------
    p_cpp, s_cpp = pack_ref(x, bitwidth, BLOCK_SIZE)

    t_cpp = bench(lambda: pack_ref(x, bitwidth, BLOCK_SIZE))
    t_k = bench(lambda: blockwise_quantize_pack_cuda(x, bitwidth, BLOCK_SIZE, False))

    (p_cpp, s_cpp), (p_k, s_k, numel), same_len, pbytes, byte_diff, smax = check_equal(x, bitwidth)

    print(f"\n== pack  {size}x{size} ({mib:.0f} MB)  bitwidth={bitwidth} ==")
    print(f"   C++  pack_ref                 : {t_cpp:8.3f} ms")
    print(f"   CUDA blockwise_quantize_pack  : {t_k:8.3f} ms")
    print(f"   speedup (cpp/cuda)            : {t_cpp / t_k:8.2f}x")
    print(f"   packed bytes={pbytes}  len_equal={same_len}  "
          f"byte_mismatch={byte_diff} ({0 if not pbytes else 100.0 * byte_diff / pbytes:.4f}%)  "
          f"scale_maxdiff={smax:.2e}")

    # ---------- unpack ----------
    device = x.device
    u_cpp = unpack_ref(p_cpp, n, s_cpp, device, torch.float32, bitwidth, BLOCK_SIZE)

    t_ucpp = bench(lambda: unpack_ref(p_cpp, n, s_cpp, device, torch.float32, bitwidth, BLOCK_SIZE))
    t_uk = bench(lambda: blockwise_unpack_dequantize_cuda(
        p_k, s_k, bitwidth, numel=n, block_size=BLOCK_SIZE, dtype=torch.float32))

    print(f"   -- unpack --")
    print(f"   C++  unpack_ref               : {t_ucpp:8.3f} ms")
    print(f"   CUDA blockwise_unpack_dequant : {t_uk:8.3f} ms")
    print(f"   speedup (cpp/cuda)            : {t_ucpp / t_uk:8.2f}x")


def main():
    assert torch.cuda.is_available(), "需要 CUDA"
    print(f"device={torch.cuda.get_device_name(0)}  block_size={BLOCK_SIZE}  iters={ITERS}")
    for size in (1024, 2048):
        for bw in (2, 4):
            run(size, bw)


if __name__ == "__main__":
    main()
