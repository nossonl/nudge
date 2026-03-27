"""MLX/Metal runtime helpers for background-safe training."""

from __future__ import annotations

import gc
import os
import subprocess
from dataclasses import dataclass


def _sysctl_int(name: str) -> int | None:
    try:
        out = subprocess.check_output(["sysctl", "-n", name], text=True).strip()
        return int(out)
    except Exception:
        return None


def _available_bytes() -> int | None:
    try:
        page_size = _sysctl_int("hw.pagesize") or 4096
        out = subprocess.check_output(["vm_stat"], text=True)
        wanted = {"Pages free", "Pages inactive", "Pages speculative"}
        pages = 0
        for line in out.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip() in wanted:
                pages += int(value.strip().rstrip("."))
        return pages * page_size
    except Exception:
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
        try:
            self.mx.set_memory_limit(int(limit_bytes))
        except Exception:
            pass
        try:
            max_wired = int(self.device_info().get("max_recommended_working_set_size") or limit_bytes)
            self.mx.set_wired_limit(min(int(limit_bytes), max_wired))
        except Exception:
            pass
        try:
            self.mx.set_cache_limit(max(0, int(limit_bytes * cache_fraction)))
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
        try:
            self.mx.clear_cache()
        except Exception:
            try:
                self.mx.metal.clear_cache()
            except Exception:
                pass

    def clear_all(self) -> None:
        self.synchronize()
        gc.collect()
        self.clear_cache()
        self.synchronize()

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
    try:
        backend = MLXBackend()
    except Exception:
        return
    backend.synchronize()
    if collect_garbage:
        gc.collect()
    backend.clear_cache()
    backend.synchronize()
