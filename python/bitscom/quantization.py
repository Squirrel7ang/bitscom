"""Quantization and bit-packing utilities for low-bit communication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

try:
    from bitscom._lowbit_cuda import (
        blockwise_quantize_pack_cuda,
        blockwise_unpack_dequantize_cuda,
        dequantize_cuda,
        pack_lowbit_cuda,
        quantize_cuda,
        unpack_lowbit_cuda,
    )

    _HAS_CUDA_KERNELS = True
except Exception:  # pragma: no cover - depends on extension build environment
    blockwise_quantize_pack_cuda = None
    blockwise_unpack_dequantize_cuda = None
    dequantize_cuda = None
    pack_lowbit_cuda = None
    quantize_cuda = None
    unpack_lowbit_cuda = None
    _HAS_CUDA_KERNELS = False


SUPPORTED_BITWIDTHS = (1, 2, 4, 8, 12, 16)
DEFAULT_BLOCK_SIZE = 256


def validate_bitwidth(bitwidth: int) -> None:
    if bitwidth not in SUPPORTED_BITWIDTHS:
        raise ValueError(
            f"bitwidth must be one of {SUPPORTED_BITWIDTHS}, got {bitwidth}"
        )


def _quant_bounds(bitwidth: int) -> Tuple[int, int]:
    if bitwidth == 1:
        # 1-bit uses an unsigned 2-level representation to keep pack/unpack simple.
        return 0, 1
    qmax = (1 << (bitwidth - 1)) - 1
    qmin = -(1 << (bitwidth - 1))
    return qmin, qmax


@dataclass(frozen=True)
class CompressedTensor:
    packed: torch.Tensor
    scale: float
    numel: int
    shape: Tuple[int, ...]
    bitwidth: int
    dtype: torch.dtype

    @property
    def packed_bytes(self) -> int:
        return int(self.packed.numel())


def quantize_tensor(
    tensor: torch.Tensor,
    bitwidth: int,
    *,
    stochastic_rounding: bool = False,
) -> Tuple[torch.Tensor, float]:
    validate_bitwidth(bitwidth)
    if not tensor.is_floating_point():
        raise TypeError("quantize_tensor expects a floating-point tensor")

    if tensor.is_cuda and _HAS_CUDA_KERNELS and bitwidth in (1, 2, 4, 8):
        q, scale = quantize_cuda(tensor, bitwidth, stochastic_rounding)
        return q, float(scale)

    qmin, qmax = _quant_bounds(bitwidth)
    x = tensor.detach().to(torch.float32).contiguous()
    max_abs = float(x.abs().max().item())
    if max_abs == 0.0:
        # Use a unit scale for all-zero tensors so the downstream code can
        # treat zero as a normal quantized value without special-casing.
        scale = 1.0
    else:
        # Symmetric max-abs scaling keeps the implementation simple and makes
        # the quantized range easy to reason about across CPU and CUDA paths.
        scale = max_abs / float(qmax)

    scaled = x / scale
    if stochastic_rounding:
        lower = torch.floor(scaled)
        prob = torch.clamp(scaled - lower, min=0.0, max=1.0)
        q = torch.where(torch.rand_like(prob) < prob, lower + 1.0, lower)
    else:
        q = torch.round(scaled)
    q = q.clamp(qmin, qmax).to(torch.int16)
    return q, scale


def quantize_tensor_blockwise(
    tensor: torch.Tensor,
    bitwidth: int,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    stochastic_rounding: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_bitwidth(bitwidth)
    if not tensor.is_floating_point():
        raise TypeError("quantize_tensor_blockwise expects a floating-point tensor")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")

    x = tensor.detach().to(torch.float32).contiguous().view(-1)
    numel = int(x.numel())
    if numel == 0:
        return x.to(torch.int16), torch.empty(0, dtype=torch.float16, device=x.device)

    qmin, qmax = _quant_bounds(bitwidth)
    num_blocks = (numel + block_size - 1) // block_size
    padded = num_blocks * block_size
    if padded != numel:
        x = torch.cat(
            [x, torch.zeros(padded - numel, dtype=x.dtype, device=x.device)],
            dim=0,
        )

    blocks = x.view(num_blocks, block_size)
    abs_blocks = blocks.abs()
    max_abs = abs_blocks.max(dim=1).values
    scale = max_abs / float(qmax)
    scale = torch.where(max_abs > 0, scale, torch.ones_like(scale))
    scale_half = scale.to(torch.float16)

    scale_for_norm = scale_half.to(torch.float32)
    normalized = abs_blocks / scale_for_norm.unsqueeze(1)
    if stochastic_rounding:
        lower = torch.floor(normalized)
        prob = torch.clamp(normalized - lower, min=0.0, max=1.0)
        mag = torch.where(torch.rand_like(prob) < prob, lower + 1.0, lower)
    else:
        mag = torch.round(normalized)

    if bitwidth == 1:
        signed = mag
    else:
        signed = mag * blocks.sign()
    q = signed.clamp(qmin, qmax).to(torch.int16).view(-1)[:numel]
    return q, scale_half


def quantize_pack_tensor_blockwise(
    tensor: torch.Tensor,
    bitwidth: int,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    stochastic_rounding: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Blockwise quantize and pack, using a fused CUDA path when available."""

    validate_bitwidth(bitwidth)
    if not tensor.is_floating_point():
        raise TypeError("quantize_pack_tensor_blockwise expects a floating-point tensor")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")

    if (
        tensor.is_cuda
        and _HAS_CUDA_KERNELS
        and blockwise_quantize_pack_cuda is not None
        and bitwidth in (1, 2, 4, 8)
        and block_size % (8 // bitwidth) == 0
        and tensor.dtype in (torch.float16, torch.float32)
    ):
        packed, scales, numel = blockwise_quantize_pack_cuda(
            tensor,
            bitwidth,
            block_size,
            stochastic_rounding,
        )
        return packed, scales, int(numel)

    q, scales = quantize_tensor_blockwise(
        tensor,
        bitwidth,
        block_size=block_size,
        stochastic_rounding=stochastic_rounding,
    )
    packed, numel = pack_lowbit(q, bitwidth)
    return packed, scales, numel


def dequantize_tensor_blockwise(
    q_tensor: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    shape: Sequence[int] | None = None,
) -> torch.Tensor:
    if device is None:
        device = q_tensor.device
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")

    q = q_tensor.to(torch.float32).contiguous().view(-1)
    numel = int(q.numel())
    if numel == 0:
        out = q
        if shape is not None:
            out = out.view(*shape)
        return out.to(device=device, dtype=dtype)

    num_blocks = int(scales.numel())
    expected_blocks = (numel + block_size - 1) // block_size
    if num_blocks != expected_blocks:
        raise ValueError(
            f"scales length {num_blocks} does not match expected blocks {expected_blocks}"
        )

    padded = num_blocks * block_size
    if padded != numel:
        q = torch.cat(
            [q, torch.zeros(padded - numel, dtype=q.dtype, device=q.device)],
            dim=0,
        )

    q_blocks = q.view(num_blocks, block_size)
    scale = scales.to(device=q.device, dtype=torch.float32).view(num_blocks, 1)
    out = (q_blocks * scale).view(-1)[:numel]
    if shape is not None:
        out = out.view(*shape)
    return out.to(device=device, dtype=dtype)


def unpack_dequantize_tensor_blockwise(
    packed: torch.Tensor,
    scales: torch.Tensor,
    bitwidth: int,
    numel: int,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    shape: Sequence[int] | None = None,
) -> torch.Tensor:
    """Unpack and blockwise dequantize, using a fused CUDA path when available."""

    validate_bitwidth(bitwidth)
    if device is None:
        device = packed.device
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")

    if (
        packed.is_cuda
        and scales.is_cuda
        and _HAS_CUDA_KERNELS
        and blockwise_unpack_dequantize_cuda is not None
        and bitwidth in (1, 2, 4, 8)
        and block_size % (8 // bitwidth) == 0
        and dtype in (torch.float16, torch.float32)
    ):
        out = blockwise_unpack_dequantize_cuda(
            packed,
            scales,
            bitwidth,
            numel,
            block_size,
            dtype,
        )
        if shape is not None:
            out = out.view(*shape)
        return out.to(device=device, dtype=dtype)

    q = unpack_lowbit(packed, bitwidth, numel)
    out = dequantize_tensor_blockwise(
        q,
        scales,
        block_size=block_size,
        dtype=dtype,
        device=device,
        shape=shape,
    )
    return out


def dequantize_tensor(
    q_tensor: torch.Tensor,
    scale: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    shape: Sequence[int] | None = None,
) -> torch.Tensor:
    if device is None:
        device = q_tensor.device

    if q_tensor.is_cuda and _HAS_CUDA_KERNELS and q_tensor.dtype == torch.int16:
        out = dequantize_cuda(q_tensor, float(scale))
    else:
        out = q_tensor.to(torch.float32) * float(scale)

    if shape is not None:
        out = out.view(*shape)
    return out.to(device=device, dtype=dtype)


def pack_lowbit(q_tensor: torch.Tensor, bitwidth: int) -> Tuple[torch.Tensor, int]:
    """Pack signed quantized values to a contiguous uint8 tensor."""

    validate_bitwidth(bitwidth)
    if q_tensor.is_cuda and _HAS_CUDA_KERNELS and bitwidth in (1, 2, 4, 8):
        # The CUDA fast path only covers bitwidths that map cleanly to byte
        # packing, which keeps the kernel interface compact.
        numel = int(q_tensor.numel())
        return pack_lowbit_cuda(q_tensor, bitwidth), numel

    qmin, _ = _quant_bounds(bitwidth)
    values = (q_tensor.to(torch.int32).contiguous().view(-1) - qmin).to(torch.int32)
    numel = int(values.numel())

    if bitwidth in (1, 2, 4, 8):
        # For these bitwidths, bit-shift packing is enough and keeps the Python
        # fallback easy to audit.
        per_byte = 8 // bitwidth
        pad = (-numel) % per_byte
        if pad:
            values = torch.cat(
                [values, torch.zeros(pad, dtype=torch.int32, device=values.device)]
            )
        values = values.view(-1, per_byte)
        shifts = torch.arange(per_byte, dtype=torch.int32, device=values.device) * bitwidth
        packed = torch.sum(values << shifts, dim=1).to(torch.uint8).contiguous()
        return packed, numel

    if bitwidth == 12:
        # 12-bit is handled separately because it does not align to byte
        # boundaries and needs a custom 3-byte packing layout.
        pad = numel % 2
        if pad:
            values = torch.cat(
                [values, torch.zeros(1, dtype=torch.int32, device=values.device)]
            )
        values = values.view(-1, 2)
        a = values[:, 0]
        b = values[:, 1]
        byte0 = (a & 0xFF).to(torch.uint8)
        byte1 = (((a >> 8) & 0x0F) | ((b & 0x0F) << 4)).to(torch.uint8)
        byte2 = ((b >> 4) & 0xFF).to(torch.uint8)
        packed = torch.stack((byte0, byte1, byte2), dim=1).reshape(-1).contiguous()
        return packed, numel

    # bitwidth == 16
    packed = values.to(torch.int16).view(torch.uint8).contiguous()
    return packed, numel


def unpack_lowbit(packed: torch.Tensor, bitwidth: int, numel: int) -> torch.Tensor:
    """Unpack a uint8-packed tensor back to signed int16 quantized values."""

    validate_bitwidth(bitwidth)
    qmin, _ = _quant_bounds(bitwidth)
    packed = packed.contiguous().view(-1)

    if packed.is_cuda and _HAS_CUDA_KERNELS and bitwidth in (1, 2, 4, 8):
        # Match the CUDA packing path above so the encoded representation stays
        # symmetric between CPU and GPU.
        return unpack_lowbit_cuda(packed, bitwidth, numel)

    if bitwidth in (1, 2, 4, 8):
        per_byte = 8 // bitwidth
        shifts = torch.arange(per_byte, dtype=torch.int32, device=packed.device) * bitwidth
        mask = (1 << bitwidth) - 1
        expanded = ((packed.to(torch.int32).unsqueeze(1) >> shifts) & mask).reshape(-1)
        values = expanded[:numel]
        return (values + qmin).to(torch.int16)

    if bitwidth == 12:
        triples = packed.view(-1, 3).to(torch.int32)
        a = triples[:, 0] | ((triples[:, 1] & 0x0F) << 8)
        b = ((triples[:, 1] >> 4) & 0x0F) | (triples[:, 2] << 4)
        values = torch.stack((a, b), dim=1).reshape(-1)[:numel]
        return (values + qmin).to(torch.int16)

    # bitwidth == 16
    values = packed.view(torch.int16).to(torch.int32)[:numel]
    return (values + qmin).to(torch.int16)


def compress_tensor(tensor: torch.Tensor, bitwidth: int) -> CompressedTensor:
    q, scale = quantize_tensor(tensor, bitwidth)
    packed, numel = pack_lowbit(q, bitwidth)
    return CompressedTensor(
        packed=packed,
        scale=scale,
        numel=numel,
        shape=tuple(tensor.shape),
        bitwidth=bitwidth,
        dtype=tensor.dtype,
    )


def decompress_tensor(
    compressed: CompressedTensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    q = unpack_lowbit(compressed.packed, compressed.bitwidth, compressed.numel)
    return dequantize_tensor(
        q,
        compressed.scale,
        dtype=dtype or compressed.dtype,
        device=device,
        shape=compressed.shape,
    )


def roundtrip_tensor(tensor: torch.Tensor, bitwidth: int) -> torch.Tensor:
    compressed = compress_tensor(tensor, bitwidth)
    return decompress_tensor(compressed, dtype=tensor.dtype, device=tensor.device)
