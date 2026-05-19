import pytest
import torch

from bitscom.quantization import (
    SUPPORTED_BITWIDTHS,
    _quant_bounds,
    _HAS_CUDA_KERNELS,
    compress_tensor,
    decompress_tensor,
    pack_lowbit,
    quantize_pack_tensor_blockwise,
    quantize_tensor,
    roundtrip_tensor,
    unpack_dequantize_tensor_blockwise,
    unpack_lowbit,
    validate_bitwidth,
)


def test_validate_bitwidth_accepts_supported_values():
    for bitwidth in SUPPORTED_BITWIDTHS:
        validate_bitwidth(bitwidth)


@pytest.mark.parametrize("bitwidth", [0, 3, 5, 7, 9, 15, 32])
def test_validate_bitwidth_rejects_unsupported_values(bitwidth):
    with pytest.raises(ValueError):
        validate_bitwidth(bitwidth)


@pytest.mark.parametrize("bitwidth", SUPPORTED_BITWIDTHS)
def test_pack_unpack_preserves_quantized_values(bitwidth):
    qmin, qmax = _quant_bounds(bitwidth)
    q = torch.arange(qmin, qmax + 1, dtype=torch.int16)

    packed, numel = pack_lowbit(q, bitwidth)
    restored = unpack_lowbit(packed, bitwidth, numel)

    assert restored.dtype == torch.int16
    assert restored.numel() == q.numel()
    assert torch.equal(restored, q)


@pytest.mark.parametrize("bitwidth", SUPPORTED_BITWIDTHS)
def test_roundtrip_error_bounded_by_quant_step(bitwidth):
    x = torch.linspace(-3.0, 3.0, steps=4096, dtype=torch.float32).reshape(64, 64)

    q, scale = quantize_tensor(x, bitwidth)
    y = roundtrip_tensor(x, bitwidth)

    assert y.shape == x.shape
    assert y.dtype == x.dtype

    max_err = torch.max(torch.abs(y - x)).item()
    assert max_err <= (scale + 1e-6)
    assert q.dtype == torch.int16


def test_compress_decompress_dtype_override():
    x = torch.randn(128, dtype=torch.float32)
    compressed = compress_tensor(x, bitwidth=4)

    y = decompress_tensor(compressed, dtype=torch.float64)

    assert y.dtype == torch.float64
    assert y.shape == x.shape
    assert compressed.packed.dtype == torch.uint8


@pytest.mark.parametrize("bitwidth", [2, 4, 8])
def test_blockwise_quantize_pack_roundtrip_cpu(bitwidth):
    x = torch.linspace(-3.0, 3.0, steps=4097, dtype=torch.float32)

    packed, scales, numel = quantize_pack_tensor_blockwise(
        x,
        bitwidth=bitwidth,
        block_size=256,
    )
    y = unpack_dequantize_tensor_blockwise(
        packed,
        scales,
        bitwidth=bitwidth,
        numel=numel,
        block_size=256,
        dtype=x.dtype,
        device=x.device,
    )

    assert packed.dtype == torch.uint8
    assert scales.dtype == torch.float16
    assert y.shape == x.shape
    assert torch.max(torch.abs(y - x)).item() <= float(scales.max().item()) + 1e-6


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUDA_KERNELS,
    reason="CUDA quantization extension is not available",
)
def test_quantize_tensor_cuda_matches_cpu_for_2bit():
    x_cpu = torch.randn(4096, dtype=torch.float32)
    x_cuda = x_cpu.cuda()

    q_cpu, s_cpu = quantize_tensor(x_cpu, bitwidth=2)
    q_cuda, s_cuda = quantize_tensor(x_cuda, bitwidth=2)

    assert abs(s_cpu - s_cuda) <= 1e-5
    assert torch.equal(q_cpu, q_cuda.cpu())


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUDA_KERNELS,
    reason="CUDA quantization extension is not available",
)
def test_pack_unpack_cuda_roundtrip_for_2bit():
    x = torch.randn(2048, device="cuda", dtype=torch.float32)
    q, _ = quantize_tensor(x, bitwidth=2)

    packed, numel = pack_lowbit(q, bitwidth=2)
    restored_q = unpack_lowbit(packed, bitwidth=2, numel=numel)

    assert packed.is_cuda
    assert restored_q.is_cuda
    assert torch.equal(restored_q, q)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUDA_KERNELS,
    reason="CUDA quantization extension is not available",
)
@pytest.mark.parametrize("bitwidth", [2, 4, 8])
def test_blockwise_quantize_pack_cuda_roundtrip(bitwidth):
    x = torch.linspace(-3.0, 3.0, steps=4097, device="cuda", dtype=torch.float32)

    packed, scales, numel = quantize_pack_tensor_blockwise(
        x,
        bitwidth=bitwidth,
        block_size=256,
    )
    y = unpack_dequantize_tensor_blockwise(
        packed,
        scales,
        bitwidth=bitwidth,
        numel=numel,
        block_size=256,
        dtype=x.dtype,
        device=x.device,
    )

    assert packed.is_cuda
    assert scales.is_cuda
    assert y.is_cuda
    assert y.shape == x.shape
    assert torch.max(torch.abs(y - x)).item() <= float(scales.max().item()) + 1e-6


def test_stochastic_rounding_cpu_is_statistically_unbiased():
    # Keep one element at max magnitude so 2-bit qmax=1 leads to scale=1.
    x = torch.full((4096,), 0.3, dtype=torch.float32)
    x[0] = 1.0

    means = []
    for _ in range(128):
        q, _ = quantize_tensor(x, bitwidth=2, stochastic_rounding=True)
        means.append(float(q[1:].to(torch.float32).mean().item()))

    avg = sum(means) / len(means)
    assert abs(avg - 0.3) < 0.05


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUDA_KERNELS,
    reason="CUDA quantization extension is not available",
)
def test_stochastic_rounding_cuda_is_statistically_unbiased():
    x_cpu = torch.full((8192,), 0.3, dtype=torch.float32)
    x_cpu[0] = 1.0
    x_cuda = x_cpu.cuda()

    means = []
    for _ in range(64):
        q_cuda, _ = quantize_tensor(x_cuda, bitwidth=2, stochastic_rounding=True)
        means.append(float(q_cuda[1:].to(torch.float32).mean().item()))

    avg = sum(means) / len(means)
    assert abs(avg - 0.3) < 0.05
