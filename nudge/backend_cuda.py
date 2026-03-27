"""CUDA/PyTorch runtime helpers for background-safe training.

Untested in this workspace on real CUDA hardware. Keep this path marked experimental
until it has live Linux GPU validation.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass


def _system_total_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        return 0


def _available_bytes() -> int | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


@dataclass(frozen=True)
class CUDAHardware:
    device_name: str
    total_memory_bytes: int
    available_memory_bytes: int | None
    system_total_memory_bytes: int
    system_available_memory_bytes: int | None
    unified_memory: bool = False
    backend: str = "cuda"


class CUDABackend:
    name = "cuda"
    unified_memory = False

    def __init__(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but torch.cuda is not available")
        self.torch = torch
        self.device = torch.device("cuda")

    def hardware(self) -> CUDAHardware:
        props = self.torch.cuda.get_device_properties(self.device)
        try:
            free_mem, _ = self.torch.cuda.mem_get_info(self.device)
            free_mem = int(free_mem)
        except Exception:
            free_mem = None
        return CUDAHardware(
            device_name=props.name,
            total_memory_bytes=int(props.total_memory),
            available_memory_bytes=free_mem,
            system_total_memory_bytes=_system_total_bytes(),
            system_available_memory_bytes=_available_bytes(),
        )

    def apply_limits(self, limit_bytes: int, cache_fraction: float = 0.25) -> None:
        total = max(1, self.hardware().total_memory_bytes)
        frac = min(0.98, max(0.05, limit_bytes / total))
        try:
            self.torch.cuda.set_per_process_memory_fraction(frac, device=self.device)
        except Exception:
            pass

    def active_memory_bytes(self) -> int:
        return int(self.torch.cuda.memory_allocated(self.device))

    def cache_memory_bytes(self) -> int:
        reserved = int(self.torch.cuda.memory_reserved(self.device))
        return max(0, reserved - self.active_memory_bytes())

    def peak_memory_bytes(self) -> int:
        return int(self.torch.cuda.max_memory_reserved(self.device))

    def current_memory_bytes(self) -> int:
        return int(self.torch.cuda.memory_reserved(self.device))

    def reset_peak_memory(self) -> None:
        try:
            self.torch.cuda.reset_peak_memory_stats(self.device)
        except Exception:
            pass

    def clear_cache(self) -> None:
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass

    def clear_all(self) -> None:
        gc.collect()
        self.clear_cache()

    def synchronize(self) -> None:
        try:
            self.torch.cuda.synchronize(self.device)
        except Exception:
            pass

    def preferred_dtype(self):
        if self.torch.cuda.is_bf16_supported():
            return self.torch.bfloat16
        return self.torch.float16

    def memory_snapshot(self) -> dict[str, float]:
        return {
            "active_gb": self.active_memory_bytes() / 1e9,
            "cache_gb": self.cache_memory_bytes() / 1e9,
            "peak_gb": self.peak_memory_bytes() / 1e9,
        }
