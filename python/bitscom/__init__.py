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
from .quantization import SUPPORTED_BITWIDTHS

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
):
    """
    初始化 bitscom：注册 lowbit backend。
    应在 torch.distributed.init_process_group 之前调用。
    """
    register_lowbit_backend(
        bitwidth=bitwidth,
        error_feedback=error_feedback,
        error_feedback_mode=error_feedback_mode,
    )
