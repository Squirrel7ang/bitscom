"""
bitscom: Low-bit distributed communication primitives for PyTorch.

用法:
    import bitscom

    # 注册 lowbit backend
    bitscom.init()

    # 初始化 process group
    import torch.distributed as dist
    dist.init_process_group(backend="lowbit")

    # 使用低比特通信
    group = bitscom.LowBitGroup(bitwidth=4)
    group.all_reduce(tensor)
"""

from .lowbit_backend import register_lowbit_backend, is_extension_available
from .api import LowBitGroup
from .quantization import DEFAULT_BLOCK_SIZE, SUPPORTED_BITWIDTHS

__all__ = [
    "register_lowbit_backend",
    "is_extension_available",
    "LowBitGroup",
    "SUPPORTED_BITWIDTHS",
    "init",
]

__version__ = "0.1.0"


def init(
    bitwidth: int = 4,
    error_feedback: bool = False,
    error_feedback_mode: str | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    stage2_error_feedback: bool | None = None,
    sparse_enabled: bool = False,
    sparse_projection_rank: int = 4,
    sparse_compression_ratio: float = 0.1,
    sparse_priority_mode: int = 0,       # 0=kFull, 1=kQuantize, 2=kDiscard
    sparse_priority_quantize_bitwidth: int = 4,
    sparse_non_priority_mode: int = 1,   # 0=kFull, 1=kQuantize, 2=kDiscard
    sparse_non_priority_quantize_bitwidth: int = 4,
):
    """
    初始化 bitscom：注册 lowbit backend。
    应在 torch.distributed.init_process_group 之前调用。
    """
    register_lowbit_backend(
        bitwidth=bitwidth,
        error_feedback=error_feedback,
        error_feedback_mode=error_feedback_mode,
        block_size=block_size,
        stage2_error_feedback=stage2_error_feedback,
        sparse_enabled=sparse_enabled,
        sparse_projection_rank=sparse_projection_rank,
        sparse_compression_ratio=sparse_compression_ratio,
        sparse_priority_mode=sparse_priority_mode,
        sparse_priority_quantize_bitwidth=sparse_priority_quantize_bitwidth,
        sparse_non_priority_mode=sparse_non_priority_mode,
        sparse_non_priority_quantize_bitwidth=sparse_non_priority_quantize_bitwidth,
    )
