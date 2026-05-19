"""
注册 lowbit backend 到 torch.distributed。

用法:
    import bitscom
    bitscom.init()
    torch.distributed.init_process_group(backend="lowbit", ...)
"""

import torch
import torch.distributed as dist

from .quantization import DEFAULT_BLOCK_SIZE

# 导入 C++ extension
try:
    from bitscom._lowbit_c import create_backend, ProcessGroupLowBit, LowBitOptions

    _HAS_EXTENSION = True
    _EXTENSION_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised via unit tests
    create_backend = None
    ProcessGroupLowBit = None
    LowBitOptions = None
    _HAS_EXTENSION = False
    _EXTENSION_IMPORT_ERROR = exc


_REGISTERED = False
_BACKEND_BITWIDTH = 4
_BACKEND_ERROR_FEEDBACK = False
_BACKEND_ERROR_FEEDBACK_MODE = "none"
_BACKEND_BLOCK_SIZE = DEFAULT_BLOCK_SIZE
_BACKEND_STAGE2_ERROR_FEEDBACK = False

_VALID_EF_MODES = {
    "auto",
    "none",
    "legacy",
    "ef",
    "ef21",
    "ef21_plus",
    "ef21+",
}


def _create_lowbit_pg(store, rank, size, timeout):
    """
    工厂函数，由 torch.distributed 在 init_process_group(backend="lowbit") 时调用。
    签名需要匹配 torch.distributed.Backend.register_backend 的要求。
    """
    if not _HAS_EXTENSION:
        raise RuntimeError(
            "bitscom C++ extension is not available. "
            "Please build/install the package first."
        ) from _EXTENSION_IMPORT_ERROR
    return create_backend(
        store=store,
        rank=rank,
        size=size,
        timeout=timeout,
        bitwidth=_BACKEND_BITWIDTH,
        error_feedback=_BACKEND_ERROR_FEEDBACK,
        error_feedback_mode=_BACKEND_ERROR_FEEDBACK_MODE,
        block_size=_BACKEND_BLOCK_SIZE,
        stage2_error_feedback=_BACKEND_STAGE2_ERROR_FEEDBACK,
    )


def _normalize_error_feedback_mode(error_feedback_mode: str | None) -> str:
    if error_feedback_mode is None:
        return "none"
    mode = str(error_feedback_mode).strip().lower()
    if mode not in _VALID_EF_MODES:
        raise ValueError(
            "error_feedback_mode must be one of: "
            + ", ".join(sorted(_VALID_EF_MODES))
        )
    if mode == "ef21+":
        return "ef21_plus"
    if mode == "ef":
        return "legacy"
    return mode


def register_lowbit_backend(
    bitwidth: int = 4,
    error_feedback: bool = False,
    error_feedback_mode: str | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    stage2_error_feedback: bool | None = None,
):
    """
    将 'lowbit' 注册为 torch.distributed 的可用 backend。
    注册后即可使用:
        dist.init_process_group(backend="lowbit", ...)

    error_feedback_mode: none / legacy / ef21 / ef21_plus
    block_size: block quantization size
    stage2_error_feedback: enable error feedback on the second quantization stage
    """
    global _REGISTERED
    global _BACKEND_BITWIDTH
    global _BACKEND_ERROR_FEEDBACK
    global _BACKEND_ERROR_FEEDBACK_MODE
    global _BACKEND_BLOCK_SIZE
    global _BACKEND_STAGE2_ERROR_FEEDBACK

    if bitwidth not in (1, 2, 4, 8, 12, 16):
        raise ValueError(
            f"bitwidth must be one of (1, 2, 4, 8, 12, 16), got {bitwidth}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")

    if error_feedback_mode is None:
        resolved_mode = "legacy" if error_feedback else "none"
    else:
        resolved_mode = _normalize_error_feedback_mode(error_feedback_mode)
        if resolved_mode == "auto":
            resolved_mode = "legacy" if error_feedback else "none"
        error_feedback = resolved_mode != "none"

    if stage2_error_feedback is None:
        resolved_stage2 = resolved_mode == "ef21_plus"
    else:
        resolved_stage2 = bool(stage2_error_feedback)

    if _REGISTERED:
        if (
            bitwidth != _BACKEND_BITWIDTH
            or error_feedback != _BACKEND_ERROR_FEEDBACK
            or resolved_mode != _BACKEND_ERROR_FEEDBACK_MODE
            or block_size != _BACKEND_BLOCK_SIZE
            or resolved_stage2 != _BACKEND_STAGE2_ERROR_FEEDBACK
        ):
            raise RuntimeError(
                "lowbit backend is already registered with different options: "
                f"bitwidth={_BACKEND_BITWIDTH}, you specified bitwidth={bitwidth}, "
                f"error_feedback={_BACKEND_ERROR_FEEDBACK}, you specified error_feedback={error_feedback}, "
                f"error_feedback_mode={_BACKEND_ERROR_FEEDBACK_MODE}, you specified error_feedback_mode={error_feedback_mode}, "
                f"block_size={_BACKEND_BLOCK_SIZE}, you specified block_size={block_size}, "
                f"stage2_error_feedback={_BACKEND_STAGE2_ERROR_FEEDBACK}, you specified stage2_error_feedback={resolved_stage2}"
            )
        return

    _BACKEND_BITWIDTH = bitwidth
    _BACKEND_ERROR_FEEDBACK = bool(error_feedback)
    _BACKEND_ERROR_FEEDBACK_MODE = resolved_mode
    _BACKEND_BLOCK_SIZE = int(block_size)
    _BACKEND_STAGE2_ERROR_FEEDBACK = resolved_stage2

    if not _HAS_EXTENSION:
        raise RuntimeError(
            "bitscom C++ extension is not available. "
            "Install with `pip install -e .` before registering backend."
        ) from _EXTENSION_IMPORT_ERROR

    dist.Backend.register_backend(
        name="lowbit",
        func=_create_lowbit_pg,
        devices=["cpu", "cuda"],
    )
    _REGISTERED = True


def is_extension_available() -> bool:
    return _HAS_EXTENSION
