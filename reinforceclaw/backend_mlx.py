"""MLX/Metal runtime helpers for background-safe training."""

from __future__ import annotations

import gc
import os
import subprocess
import time
from dataclasses import dataclass

_PAGE_SIZE = None
_DRAIN_BACKEND = None
_VM_STAT_KEYS = {"Pages free", "Pages inactive", "Pages speculative"}
_AVAILABLE_CACHE = (0.0, None)


def _sysctl_int(name: str) -> int | None:
    try:
        out = subprocess.check_output(["sysctl", "-n", name], text=True, timeout=2).strip()
        return int(out)
    except Exception:
        return None


def _available_bytes() -> int | None:
    global _PAGE_SIZE, _AVAILABLE_CACHE
    now = time.monotonic()
    if now - _AVAILABLE_CACHE[0] < 1.0:
        return _AVAILABLE_CACHE[1]
    try:
        if _PAGE_SIZE is None:
            _PAGE_SIZE = _sysctl_int("hw.pagesize") or 4096
        out = subprocess.check_output(["vm_stat"], text=True, timeout=2)
        pages = 0
        for line in out.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip() in _VM_STAT_KEYS:
                pages += int(value.strip().rstrip("."))
        _AVAILABLE_CACHE = (now, pages * _PAGE_SIZE)
        return _AVAILABLE_CACHE[1]
    except Exception:
        _AVAILABLE_CACHE = (now, None)
        return None


@dataclass(frozen=True)
class MLXHardware:
    device_name: str
    total_memory_bytes: int
    available_memory_bytes: int | None
    recommended_working_set_bytes: int | None
    system_total_memory_bytes: int = 0
    system_available_memory_bytes: int | None = None
    unified_memory: bool = True
    backend: str = "mlx"


class MLXBackend:
    name = "mlx"
    unified_memory = True

    def __init__(self):
        import mlx.core as mx

        self.mx = mx

    def hardware(self) -> MLXHardware:
        info = self.device_info()
        total = int(info.get("memory_size") or _sysctl_int("hw.memsize") or 0)
        available = _available_bytes()
        return MLXHardware(
            device_name=str(info.get("architecture") or info.get("name") or "apple"),
            total_memory_bytes=total,
            available_memory_bytes=available,
            recommended_working_set_bytes=int(info.get("max_recommended_working_set_size") or 0) or None,
            system_total_memory_bytes=total,
            system_available_memory_bytes=available,
        )

    def device_info(self) -> dict:
        try:
            return self.mx.device_info()
        except Exception:
            return {}

    def apply_limits(self, limit_bytes: int, cache_fraction: float = 0.25) -> None:
        max_wired = int(self.device_info().get("max_recommended_working_set_size") or limit_bytes)
        for fn, value in (
            (getattr(self.mx, "set_memory_limit", None), int(limit_bytes)),
            (getattr(self.mx, "set_wired_limit", None), min(int(limit_bytes), max_wired)),
            (getattr(self.mx, "set_cache_limit", None), max(0, int(limit_bytes * cache_fraction))),
        ):
            try:
                if callable(fn):
                    fn(value)
            except Exception:
                pass

    def active_memory_bytes(self) -> int:
        try:
            return int(self.mx.get_active_memory())
        except Exception:
            return 0

    def cache_memory_bytes(self) -> int:
        try:
            return int(self.mx.get_cache_memory())
        except Exception:
            return 0

    def peak_memory_bytes(self) -> int:
        try:
            return int(self.mx.get_peak_memory())
        except Exception:
            return 0

    def current_memory_bytes(self) -> int:
        return self.active_memory_bytes() + self.cache_memory_bytes()

    def reset_peak_memory(self) -> None:
        try:
            self.mx.reset_peak_memory()
        except Exception:
            pass

    def clear_cache(self) -> None:
        for fn in (getattr(self.mx, "clear_cache", None), getattr(getattr(self.mx, "metal", None), "clear_cache", None)):
            try:
                if callable(fn):
                    fn()
                    return
            except Exception:
                pass

    def clear_all(self) -> None:
        self.synchronize()
        gc.collect()
        self.clear_cache()
        self.synchronize()

    def preferred_dtype(self):
        return None

    def synchronize(self) -> None:
        try:
            self.mx.synchronize()
        except Exception:
            pass

    def memory_snapshot(self) -> dict[str, float]:
        return {
            "active_gb": self.active_memory_bytes() / 1e9,
            "cache_gb": self.cache_memory_bytes() / 1e9,
            "peak_gb": self.peak_memory_bytes() / 1e9,
        }


def mlx_drain(*, collect_garbage: bool = False) -> None:
    """Best-effort MLX/Metal cleanup for transient inference or teardown."""
    global _DRAIN_BACKEND
    try:
        backend = _DRAIN_BACKEND or MLXBackend()
        _DRAIN_BACKEND = backend
    except Exception:
        return
    backend.synchronize()
    if collect_garbage:
        gc.collect()
    backend.clear_cache()
    backend.synchronize()
