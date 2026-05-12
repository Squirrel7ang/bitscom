"""
注册 lowbit backend 到 torch.distributed。

用法:
    import bitscom
    bitscom.init()
    torch.distributed.init_process_group(backend="lowbit", ...)
"""

import torch
import torch.distributed as dist

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
):
    """
    将 'lowbit' 注册为 torch.distributed 的可用 backend。
    注册后即可使用:
        dist.init_process_group(backend="lowbit", ...)

    error_feedback_mode: none / legacy / ef21 / ef21_plus
    """
    global _REGISTERED
    global _BACKEND_BITWIDTH
    global _BACKEND_ERROR_FEEDBACK
    global _BACKEND_ERROR_FEEDBACK_MODE

    if bitwidth not in (1, 2, 4, 8, 12, 16):
        raise ValueError(
            f"bitwidth must be one of (1, 2, 4, 8, 12, 16), got {bitwidth}"
        )

    if error_feedback_mode is None:
        resolved_mode = "legacy" if error_feedback else "none"
    else:
        resolved_mode = _normalize_error_feedback_mode(error_feedback_mode)
        if resolved_mode == "auto":
            resolved_mode = "legacy" if error_feedback else "none"
        error_feedback = resolved_mode != "none"

    if _REGISTERED:
        if (
            bitwidth != _BACKEND_BITWIDTH
            or error_feedback != _BACKEND_ERROR_FEEDBACK
            or resolved_mode != _BACKEND_ERROR_FEEDBACK_MODE
        ):
            raise RuntimeError(
                "lowbit backend is already registered with different options: "
                f"bitwidth={_BACKEND_BITWIDTH}, "
                f"error_feedback={_BACKEND_ERROR_FEEDBACK}, "
                f"error_feedback_mode={_BACKEND_ERROR_FEEDBACK_MODE}"
            )
        return

    _BACKEND_BITWIDTH = bitwidth
    _BACKEND_ERROR_FEEDBACK = bool(error_feedback)
    _BACKEND_ERROR_FEEDBACK_MODE = resolved_mode

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
