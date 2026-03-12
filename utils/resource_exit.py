import errno
import sys
from functools import wraps

try:
    import torch
except Exception:
    torch = None

CUDA_OOM_EXIT_CODE = 61
MEMORY_OOM_EXIT_CODE = 62
DISK_FULL_EXIT_CODE = 63

_CUDA_OOM_PATTERNS = (
    "cuda out of memory",
    "cuda oom",
    "cublas_status_alloc_failed",
    "cuda error: out of memory",
    "hip out of memory",
)

_MEMORY_OOM_PATTERNS = (
    "out of memory",
    "can't allocate memory",
    "cannot allocate memory",
    "std::bad_alloc",
    "defaultcpuallocator: can't allocate memory",
    "defaultcpuallocator: not enough memory",
)

_DISK_FULL_PATTERNS = (
    "no space left on device",
)

_EXIT_MESSAGES = {
    CUDA_OOM_EXIT_CODE: "检测到显存不足（CUDA OOM）。",
    MEMORY_OOM_EXIT_CODE: "检测到内存不足（OOM）。",
    DISK_FULL_EXIT_CODE: "检测到磁盘空间不足（No space left on device）。",
}


def _iter_exception_messages(exc):
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current)
        if message:
            yield message.lower()
        current = current.__cause__ or current.__context__


def _is_cuda_oom(exc):
    if torch is not None:
        torch_oom = getattr(torch, "OutOfMemoryError", None)
        if torch_oom is not None and isinstance(exc, torch_oom):
            return True

    return any(
        any(pattern in message for pattern in _CUDA_OOM_PATTERNS)
        for message in _iter_exception_messages(exc)
    )


def _is_disk_full(exc):
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
        return True

    return any(
        any(pattern in message for pattern in _DISK_FULL_PATTERNS)
        for message in _iter_exception_messages(exc)
    )


def _is_memory_oom(exc):
    if isinstance(exc, MemoryError):
        return True

    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOMEM:
        return True

    if _is_cuda_oom(exc) or _is_disk_full(exc):
        return False

    return any(
        any(pattern in message for pattern in _MEMORY_OOM_PATTERNS)
        for message in _iter_exception_messages(exc)
    )


def classify_resource_error(exc):
    if _is_cuda_oom(exc):
        return CUDA_OOM_EXIT_CODE
    if _is_memory_oom(exc):
        return MEMORY_OOM_EXIT_CODE
    if _is_disk_full(exc):
        return DISK_FULL_EXIT_CODE
    return None


def exit_for_resource_error(exc, stage_desc):
    code = classify_resource_error(exc)
    if code is None:
        return False

    print(f"{stage_desc}失败：{_EXIT_MESSAGES[code]}")
    sys.exit(code)


def run_with_resource_error_handling(stage_desc, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        if not exit_for_resource_error(exc, stage_desc):
            raise


def resource_error_handler(stage_desc):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return run_with_resource_error_handling(stage_desc, fn, *args, **kwargs)

        return wrapper

    return decorator
