"""Adaptive LoRA trainer with MLX and CUDA backends."""

from __future__ import annotations

import gc
import hashlib
import fnmatch
import json
import os
import platform
import random
import re
import secrets
import shutil
import signal
import sys
import time
import struct
import fcntl
import inspect
import math
import statistics
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

from . import db
from .backend_cuda import CUDABackend
from .backend_mlx import MLXBackend, mlx_drain

GB = 1024 ** 3
MIN_TRAINING_BUDGET = 2 * GB
MAC_OS_RESERVE = 8 * GB
LINUX_OS_RESERVE = 4 * GB
TRAIN_LOG_PATH = Path.home() / ".reinforceclaw" / "train.log"
TRAIN_LOCK_PATH = Path.home() / ".reinforceclaw" / "train.lock"
CAFFEINATE_PID_PATH = Path.home() / ".reinforceclaw" / "caffeinate.pid"
TRAIN_LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_DETAIL_CHARS = 500
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_VERSION_RE = re.compile(r"(\d+)(?:\.(\d+))?")
BACKGROUND_IDLE_LOAD = 0.85
BACKGROUND_WINDOW_MINUTES = 180
# Module-level caches and one-time filesystem guards.
_LOG_SECURED = False
_LOG_DIR_SECURED = False
_REMOTE_CODE_WARNED = set()

# lazy MLX imports
mx = nn = optim = mlx_load = linear_to_lora_layers = tree_map = tree_unflatten = None
_HF_SOURCE_CACHE: dict[str, str] = {}
_HF_SOURCE_CACHE_MAX = 32
_CUDA_ACTIVITY_CACHE = (0.0, None)
_HF_ALLOW_PATTERNS = ("*.json", "*.jinja", "*.model", "*.txt", "*.safetensors", "*.index.json", "*.tiktoken")
_HF_IGNORE_PATTERNS = ("*.bin", "*.h5", "*.msgpack", "*.ot", "*.npz", "*.gguf", "*.onnx", "*.tflite", "*.zip")
_SENSITIVE_ENV = re.compile(r"(?i)(^|_)(api|auth|credential|key|password|pat|private|secret|session|token)($|_)")
_HF_ENV_KEYS = {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"}
_DROP_ENV_KEYS = {"AWS_ACCESS_KEY_ID", "AWS_SESSION_TOKEN", "DATABASE_URL", "SSH_AUTH_SOCK"}
_URL_CREDENTIALS = re.compile(r"^[a-z][a-z0-9+.-]*://[^/\s:@]+:[^@\s]+@", re.I)
_TOKENIZER_FILES = {
    "added_tokens.json",
    "chat_template.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}
_TOKENIZER_PATTERNS = ("tokenizer*.json", "tokenizer*.model", "tokenizer*.txt", "tokenizer*.jinja")
_TORCH_TARGET_CACHE_ATTR = "_reinforceclaw_target_modules_cache"
_TORCH_EXCLUDED_MODULES = {"audio_projector", "image_encoder", "mm_projector", "vision", "vision_model", "vision_tower"}


def _ensure_mlx():
    global mx, nn, optim, mlx_load, linear_to_lora_layers, tree_map, tree_unflatten
    if mx is not None:
        return
    import mlx.core
    import mlx.nn
    import mlx.optimizers
    import mlx.utils
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers as _linear_to_lora_layers

    mx, nn, optim = mlx.core, mlx.nn, mlx.optimizers
    mlx_load, linear_to_lora_layers = load, _linear_to_lora_layers
    tree_map, tree_unflatten = mlx.utils.tree_map, mlx.utils.tree_unflatten


def _scrub_secret(value):
    if isinstance(value, str):
        return db.redact_secrets(value)
    if isinstance(value, (bytes, bytearray)):
        return _scrub_secret(bytes(value).decode("utf-8", "replace"))
    if isinstance(value, dict):
        return {k: _scrub_secret(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub_secret(v) for v in value]
    return value


def _detail(value) -> str:
    return _scrub_secret(str(value))[:LOG_DETAIL_CHARS]


def _child_env(*, keep_hf=False):
    env = dict(os.environ)
    for key in tuple(env):
        if key in _HF_ENV_KEYS and keep_hf:
            continue
        if _SENSITIVE_ENV.search(key) or key in _DROP_ENV_KEYS or _URL_CREDENTIALS.search(str(env.get(key, ""))):
            env.pop(key, None)
    return env


def _log_event(event: str, **fields) -> None:
    global _LOG_DIR_SECURED, _LOG_SECURED
    if not _LOG_DIR_SECURED:
        db.secure_private_dir(TRAIN_LOG_PATH.parent)
        _LOG_DIR_SECURED = True
    try:
        if TRAIN_LOG_PATH.exists() and not TRAIN_LOG_PATH.is_symlink() and TRAIN_LOG_PATH.stat().st_size > TRAIN_LOG_MAX_BYTES:
            os.replace(TRAIN_LOG_PATH, TRAIN_LOG_PATH.with_suffix(".log.1"))
            _LOG_SECURED = False
    except OSError:
        pass
    record = _scrub_secret({"ts": datetime.now().astimezone().isoformat(timespec="seconds"), "event": event, **fields})
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | _O_NOFOLLOW
    try:
        fd = os.open(TRAIN_LOG_PATH, flags, 0o600)
    except OSError:
        return
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        if not _LOG_SECURED:
            os.fchmod(fh.fileno(), 0o600)
            _LOG_SECURED = True
        fh.write(json.dumps(record) + "\n")


def _trust_remote_code(config) -> bool:
    return bool(config.get("trust_remote_code", False))


def _warn_remote_code(config) -> None:
    if _trust_remote_code(config):
        model = str(config.get("model"))
        if model in _REMOTE_CODE_WARNED:
            return
        _REMOTE_CODE_WARNED.add(model)
        msg = f"WARNING: trust_remote_code=True executes model repo Python: {config.get('model')}"
        print(msg, file=sys.stderr)
        _log_event("remote_code_warning", model=config.get("model"))


def _guard_remote_code(config) -> None:
    if _trust_remote_code(config) and config.get("_background") and os.environ.get("REINFORCECLAW_ALLOW_REMOTE_CODE") != "1":
        raise RuntimeError("remote_code_background_blocked:set_REINFORCECLAW_ALLOW_REMOTE_CODE=1_to_allow")


def _sanitize_retry_config(config):
    return {
        k: v for k, v in config.items()
        if k == "token_clip" or (k not in {"_retry_token", "openclaw_secret"} and not _SENSITIVE_ENV.search(str(k)))
    }


def _write_private_text(path: Path, text: str) -> None:
    db.secure_private_dir(path.parent)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    committed = False
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
        db.secure_private_file(path)
        committed = True
    finally:
        if not committed:
            tmp.unlink(missing_ok=True)


def _hf_tokenizer_kwargs(config, *, local_files_only: bool) -> dict:
    return {"trust_remote_code": _trust_remote_code(config), "local_files_only": local_files_only}


def _allowed_tokenizer_file(name: str) -> bool:
    return name in _TOKENIZER_FILES or any(fnmatch.fnmatch(name, pattern) for pattern in _TOKENIZER_PATTERNS)


def _prepare_tokenizer_source(model_source: str, model_name: str) -> str:
    if "gemma-4" not in str(model_name).lower():
        return model_source
    root = Path(model_source)
    if not root.exists():
        return model_source
    config_path = root / "tokenizer_config.json"
    if not config_path.exists():
        return model_source
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return model_source
    extras = data.get("extra_special_tokens")
    if not isinstance(extras, list):
        return model_source
    token_keys = {
        key: value for key, value in data.items()
        if key.endswith("_token") and isinstance(value, str)
    }
    mapped = {}
    used = set()
    for idx, token in enumerate(extras):
        key = next((name for name, value in token_keys.items() if value == token and name not in used), None)
        key = key or f"extra_special_token_{idx}"
        mapped[key] = token
        used.add(key)
    data["extra_special_tokens"] = mapped
    target = db.secure_private_dir(
        Path.home() / ".reinforceclaw" / "tokenizers" / hashlib.sha256(str(root).encode()).hexdigest()[:16]
    )
    dirs = []
    for stale_dir in target.parent.iterdir():
        try:
            if stale_dir.is_dir():
                dirs.append((stale_dir.stat().st_mtime, stale_dir))
        except OSError:
            pass
    for _, stale_dir in sorted(dirs, reverse=True)[16:]:
        shutil.rmtree(stale_dir, ignore_errors=True)
    for stale in target.iterdir():
        if stale.name != "tokenizer_config.json" and not _allowed_tokenizer_file(stale.name):
            stale.unlink(missing_ok=True)
    for child in root.iterdir():
        if not child.is_file() or child.name == "tokenizer_config.json":
            continue
        if not _allowed_tokenizer_file(child.name):
            continue
        try:
            resolved_child = child.resolve(strict=True)
            resolved_root = root.resolve(strict=True)
        except OSError:
            continue
        if resolved_child != resolved_root and resolved_root not in resolved_child.parents:
            _log_event("tokenizer_file_skipped", reason="outside_model_dir", file=child.name)
            continue
        dest = target / child.name
        if dest.exists():
            continue
        try:
            os.symlink(child, dest)
        except FileExistsError:
            pass
        except OSError:
            try:
                shutil.copy2(child, dest)
            except FileExistsError:
                pass
    config = target / "tokenizer_config.json"
    _write_private_text(config, json.dumps(data, indent=2, sort_keys=True))
    return str(target)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _lock_pid() -> int | None:
    try:
        text = TRAIN_LOCK_PATH.read_text(encoding="utf-8").strip()
        return int(text) if text else None
    except (OSError, ValueError):
        return None


def _acquire_lock() -> int | None:
    db.secure_private_dir(TRAIN_LOCK_PATH.parent)
    if TRAIN_LOCK_PATH.is_symlink():
        _log_event("lock_path_rejected", reason="symlink")
        return None
    fd = os.open(TRAIN_LOCK_PATH, os.O_CREAT | os.O_RDWR | _O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stale_pid = _lock_pid()
        os.close(fd)
        if stale_pid is not None and not _pid_alive(stale_pid):
            try:
                TRAIN_LOCK_PATH.unlink(missing_ok=True)
                fd = os.open(TRAIN_LOCK_PATH, os.O_CREAT | os.O_RDWR | _O_NOFOLLOW, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                try:
                    os.close(fd)
                except OSError:
                    pass
                return None
        else:
            return None
    payload = str(os.getpid()).encode("utf-8")
    try:
        os.ftruncate(fd, 0)
        wrote = os.write(fd, payload)
    except OSError:
        wrote = 0
    if wrote != len(payload):
        _log_event("lock_pid_write_failed")
        _release_lock(fd)
        return None
    return fd


def _release_lock(fd: int | None) -> None:
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _set_low_priority() -> None:
    try:
        os.nice(19)
    except (OSError, PermissionError):
        pass
    if platform.system() == "Darwin":
        try:
            os.setpriority(os.PRIO_DARWIN_NONUI, 0, 0)
        except (AttributeError, OSError):
            pass


def _keep_awake() -> subprocess.Popen | None:
    """Prevent macOS sleep for the lifetime of the current process. Idempotent."""
    if platform.system() != "Darwin":
        return None
    _cleanup_caffeinate()
    try:
        proc = subprocess.Popen(
            ["caffeinate", "-isw", str(os.getpid())],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=_child_env(),
            start_new_session=True,
        )
        db.secure_private_dir(CAFFEINATE_PID_PATH.parent)
        fd = os.open(CAFFEINATE_PID_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(str(proc.pid))
        return proc
    except FileNotFoundError:
        return None


def _cleanup_caffeinate() -> None:
    try:
        pid = int(CAFFEINATE_PID_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        CAFFEINATE_PID_PATH.unlink(missing_ok=True)
    except PermissionError:
        pass


def _os_reserve_bytes(unified_memory: bool) -> int:
    return MAC_OS_RESERVE if unified_memory else LINUX_OS_RESERVE


def _load_ratio() -> float:
    if not hasattr(os, "getloadavg"):
        return 0.0
    try:
        return os.getloadavg()[0] / max(os.cpu_count() or 1, 1)
    except OSError:
        return 0.0


def _free_mlx() -> None:
    mlx_drain(collect_garbage=True)


def _hf_token() -> str | None:
    for name in _HF_ENV_KEYS:
        value = os.environ.get(name)
        if value:
            return value
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        value = token_path.read_text(encoding="utf-8").strip()
        if value:
            return value
    return None


def resolve_hf_model_source(model_name: str, *, trust_remote_code: bool = False) -> str:
    path = Path(model_name).expanduser()
    if path.exists():
        return str(path.resolve())
    cache_key = f"{model_name}|remote={int(trust_remote_code)}"
    cached = _HF_SOURCE_CACHE.get(cache_key)
    if cached and Path(cached).exists():
        _HF_SOURCE_CACHE[cache_key] = _HF_SOURCE_CACHE.pop(cache_key)
        return cached
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_name
    kwargs = {
        "repo_id": model_name,
        "allow_patterns": list(_HF_ALLOW_PATTERNS) + (["*.py"] if trust_remote_code else []),
        "ignore_patterns": list(_HF_IGNORE_PATTERNS),
    }
    token = _hf_token()
    if token:
        kwargs["token"] = token
    try:
        try:
            local_path = snapshot_download(resume_download=True, **kwargs)
        except TypeError:
            _log_event("hf_download_resume_unsupported", model=model_name)
            local_path = snapshot_download(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"huggingface_download_failed:{_scrub_secret(str(exc))}") from None
    local_path = str(Path(local_path).resolve())
    if len(_HF_SOURCE_CACHE) >= _HF_SOURCE_CACHE_MAX:
        _HF_SOURCE_CACHE.pop(next(iter(_HF_SOURCE_CACHE)))
    _HF_SOURCE_CACHE[cache_key] = local_path
    return local_path


def _feedback_source(config) -> str | None:
    value = config.get("feedback_source")
    return str(value) if value else None


def _select_backend(config):
    forced = config.get("compute_backend")
    if forced == "mlx":
        return MLXBackend()
    if forced == "cuda":
        return CUDABackend()
    return MLXBackend() if platform.system() == "Darwin" else CUDABackend()


def _has_minimum_headroom(hardware) -> bool:
    reserve = _os_reserve_bytes(hardware.unified_memory)
    available = hardware.available_memory_bytes
    if hardware.unified_memory:
        return available is None or available - reserve >= MIN_TRAINING_BUDGET
    host_available = getattr(hardware, "system_available_memory_bytes", None)
    return (
        available is not None and available >= MIN_TRAINING_BUDGET
        and host_available is not None and host_available - reserve >= MIN_TRAINING_BUDGET
    )


def _settle_backend_hardware(backend, hardware=None, rounds: int = 2, delay: float = 0.5):
    hardware = hardware or backend.hardware()
    if backend.name != "mlx":
        return hardware
    settled = hardware
    previous = hardware.available_memory_bytes
    for _ in range(max(1, rounds)):
        backend.clear_all()
        backend.synchronize()
        time.sleep(delay)
        current = backend.hardware()
        current_available = current.available_memory_bytes
        settled = current
        if previous is not None and current_available is not None:
            if abs(current_available - previous) <= max(256 * 1024 * 1024, int(previous * 0.05)):
                break
        elif previous == current_available:
            break
        previous = current_available
    return settled


def _scheduled_window_open(config) -> bool:
    schedule = config.get("train_schedule", "auto")
    if schedule in ("manual", "auto") or not config.get("_background"):
        return True
    try:
        hour, minute = (int(part) for part in schedule.split(":", 1))
    except ValueError:
        return False
    now = datetime.now().astimezone()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    window = timedelta(minutes=float(config.get("schedule_window_minutes", BACKGROUND_WINDOW_MINUTES)))
    if now < target:
        target -= timedelta(days=1)
    return target <= now < target + window


def _lora_init_kwargs(config) -> dict:
    """Opt-in LoRA init variants. Defaults to PEFT's gaussian. EVA needs activations,
    so we fall back to gaussian if the user enabled it but didn't supply a data loader."""
    choice = (config.get("lora_init") or "default").lower()
    if choice in ("default", "", "gaussian", "kaiming"):
        return {}
    if choice in ("pissa", "olora", "loftq"):
        return {"init_lora_weights": choice}
    if choice == "eva":
        _log_event("lora_init_skipped", init="eva", reason="requires_activation_data")
    return {}


def _build_optimizer_torch(torch, model, cfg, config):
    """AdamW with optional LoRA+ (higher LR on B matrices) and fused=True when CUDA
    supports it (~10-15% free speedup on modern GPUs). Ratio 0 / unset = vanilla."""
    ratio = float(config.get("lora_plus_ratio", 0.0) or 0.0)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    b_params = [p for n, p in trainable if ".lora_B" in n]

    def build(kwargs):
        if ratio <= 0 or not b_params:
            return torch.optim.AdamW([p for _, p in trainable], lr=cfg["lr"], **kwargs)
        other = [p for n, p in trainable if ".lora_B" not in n]
        return torch.optim.AdamW(
            [{"params": other, "lr": cfg["lr"]},
             {"params": b_params, "lr": cfg["lr"] * ratio}],
            **kwargs,
        )

    try:
        return build({"fused": True} if _fused_adamw_supported(torch) else {})
    except (TypeError, RuntimeError):
        return build({})


def _fused_adamw_supported(torch) -> bool:
    try:
        match = _VERSION_RE.match(str(torch.__version__))
        version = tuple(int(part or 0) for part in match.groups()) if match else (0, 0)
        return version >= (2, 0) and bool(torch.cuda.is_available())
    except Exception:
        return False


def _maybe_apply_liger(model, config) -> None:
    """Liger-Kernel fused ops (~20% throughput, ~60% CE-memory cut). Opt-in."""
    if not config.get("use_liger"):
        return
    try:
        from liger_kernel.transformers import _apply_liger_kernel_to_instance
        _apply_liger_kernel_to_instance(model=model)
        _log_event("liger_applied", model_type=getattr(getattr(model, "config", None), "model_type", "unknown"))
    except Exception as exc:
        _log_event("liger_skipped", detail=_detail(exc))


def _maybe_torch_compile(model, config):
    """torch.compile for kernel fusion. Opt-in — conflicts with some grad-checkpoint setups."""
    mode = config.get("compile_backend") or "none"
    if mode == "none":
        return model
    try:
        import torch
        compiled = torch.compile(model, mode=mode if mode in ("reduce-overhead", "max-autotune", "default") else "reduce-overhead")
        _log_event("torch_compile_applied", mode=mode)
        return compiled
    except Exception as exc:
        _log_event("torch_compile_skipped", detail=_detail(exc))
        return model


def _cuda_activity(backend) -> dict | None:
    global _CUDA_ACTIVITY_CACHE
    now = time.monotonic()
    if now - _CUDA_ACTIVITY_CACHE[0] < 2.0:
        return _CUDA_ACTIVITY_CACHE[1]
    device_index = getattr(getattr(backend, "device", None), "index", None)
    device_index = 0 if device_index is None else device_index
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            value = {"gpu_util": int(util.gpu), "mem_used": int(mem.used), "mem_total": int(mem.total), "source": "nvml"}
            _CUDA_ACTIVITY_CACHE = (now, value)
            return value
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            env=_child_env(),
            text=True,
            timeout=2,
        ).strip()
        if not out:
            return None
        util, used, total = (int(part.strip()) for part in out.split(",", 2))
        mib = 1024 * 1024
        value = {"gpu_util": util, "mem_used": used * mib, "mem_total": total * mib, "source": "nvidia-smi"}
        _CUDA_ACTIVITY_CACHE = (now, value)
        return value
    except Exception:
        pass
    try:
        hardware = backend.hardware()
        if hardware.available_memory_bytes is None:
            return None
        used = max(0, hardware.total_memory_bytes - hardware.available_memory_bytes)
        value = {"gpu_util": 0, "mem_used": used, "mem_total": hardware.total_memory_bytes, "source": "fallback"}
        _CUDA_ACTIVITY_CACHE = (now, value)
        return value
    except Exception:
        _CUDA_ACTIVITY_CACHE = (now, None)
        return None


def _background_block_reason(config, backend, hardware) -> str | None:
    """Busy-system guard. Applies to scheduled nightly runs too — a 3am kick-off
    should still stand down if another process is hammering the GPU, not trample it."""
    if not config.get("_background"):
        return None
    if not _scheduled_window_open(config):
        return "outside_schedule_window"
    if _load_ratio() > float(config.get("idle_load_threshold", BACKGROUND_IDLE_LOAD)):
        return "high_cpu_load"
    reserve = _os_reserve_bytes(hardware.unified_memory)
    if hardware.unified_memory:
        available = hardware.available_memory_bytes
        if available is not None and available < reserve + int(1.5 * MIN_TRAINING_BUDGET):
            return "memory_busy"
        return None
    host_available = getattr(hardware, "system_available_memory_bytes", None)
    host_busy = host_available is not None and host_available < reserve + MIN_TRAINING_BUDGET
    if host_busy:
        return "host_memory_busy"
    activity = _cuda_activity(backend)
    if activity is None:
        return "missing_cuda_idle_telemetry"
    if activity.get("source") != "fallback":
        if activity["gpu_util"] > int(config.get("cuda_idle_gpu_util", 80)):
            return "gpu_busy"
        mem_threshold = float(config.get("cuda_idle_mem_fraction", 0.80))
        if activity["mem_total"] and activity["mem_used"] / activity["mem_total"] > mem_threshold:
            return "gpu_memory_busy"
    gpu_available = hardware.available_memory_bytes
    gpu_busy = gpu_available is not None and gpu_available < max(MIN_TRAINING_BUDGET, int(hardware.total_memory_bytes * 0.20))
    if gpu_busy:
        return "low_free_vram"
    return None


def smoke_status(config, conn):
    cfg = {**config, "_background": True}
    trainable = db.count_trainable_untrained(conn, source=_feedback_source(cfg), model=cfg.get("model"))
    batch_min = cfg.get("batch_min", 32)
    if trainable < batch_min:
        return {"would_train": False, "reason": "below_threshold", "trainable": trainable, "batch_min": batch_min}
    compat = model_compatibility(cfg)
    if compat["ok"] is False:
        return {
            "would_train": False,
            "reason": compat["reason"],
            "detail": compat.get("detail"),
            "trainable": trainable,
            "batch_min": batch_min,
            "backend": compat.get("backend"),
        }
    try:
        backend = _select_backend(cfg)
    except Exception as exc:
        return {"would_train": False, "reason": "backend_unavailable", "detail": str(exc), "trainable": trainable, "batch_min": batch_min}
    if backend.name == "cuda":
        try:
            _torch_stack()
        except RuntimeError as exc:
            return {"would_train": False, "reason": str(exc), "backend": backend.name, "trainable": trainable, "batch_min": batch_min}
    hardware = backend.hardware()
    if not _has_minimum_headroom(hardware):
        return {"would_train": False, "reason": "insufficient_headroom", "backend": backend.name, "trainable": trainable, "batch_min": batch_min}
    block = _background_block_reason(cfg, backend, hardware)
    avail_gb, host_gb = _hardware_gbs(hardware)
    return {
        "would_train": block is None,
        "reason": block or "ready",
        "backend": backend.name,
        "trainable": trainable,
        "batch_min": batch_min,
        "schedule": cfg.get("train_schedule", "auto"),
        "available_gb": avail_gb,
        "host_available_gb": host_gb,
    }


def model_compatibility(config):
    from . import profile as _profile

    model = str(config.get("model", ""))
    mp = config.get("model_profile") if config.get("model_profile_model") == model else None
    try:
        prof = _profile.ModelProfile(**mp) if isinstance(mp, dict) else _profile.detect(model)
    except TypeError:
        prof = _profile.detect(model)
    if prof.provider == "ollama":
        return {"ok": False, "reason": "ollama_model_not_trainable",
                "detail": (f"{model} is an Ollama inference tag. You can rate its responses, "
                           "but training needs the matching local/HuggingFace weights first; "
                           "the trained adapter can then be attached back to the matching Ollama model.")}
    if prof.provider == "gguf":
        return {"ok": False, "reason": "gguf_models_not_trainable",
                "detail": (f"{model} is a GGUF/llama.cpp inference file. You can rate its responses, "
                           "but training needs the matching local/HuggingFace weights first; "
                           "the trained adapter can then be used for inference with matching GGUF/Ollama setups.")}
    if not prof.trainable:
        return {"ok": False, "reason": "cloud_api_not_trainable",
                "detail": (f"{model} is a closed API ({prof.family}) — weights are not public so RL can't update it. "
                           f"You can keep rating its responses; pick a local model as the train target (reinforceclaw init) "
                           f"and those ratings will feed into training a local adapter you control.")}
    try:
        backend = _select_backend(config)
    except Exception as exc:
        return {"ok": False, "reason": "backend_unavailable", "detail": str(exc)}

    if backend.name == "cuda":
        if prof.provider == "mlx":
            return {"ok": False, "reason": "mlx_model_on_cuda_backend", "backend": backend.name, "detail": model}
        lowered = model.lower()
        if any(tag in lowered for tag in ("awq", "gptq", "exl2")):
            return {"ok": False, "reason": "backend_specific_quantized_repo", "backend": backend.name, "detail": model}
        try:
            _torch_stack()
        except RuntimeError as exc:
            return {"ok": False, "reason": "missing_cuda_training_dependency", "backend": backend.name, "detail": str(exc)}
    return {"ok": True, "backend": backend.name}


@dataclass(frozen=True)
class TrainingPlan:
    effective_batch_size: int
    grad_accum: int
    steps: int
    memory_limit_bytes: int
    training_budget_bytes: int
    resident_model_bytes: int
    aggressive_checkpointing: bool
    busy: bool


class ResumeInvalidated(RuntimeError):
    pass


def _slice_steps(config, total_steps: int) -> int:
    if not config.get("_background"):
        return total_steps
    limit = max(1, int(config.get("background_slice_steps", 2)))
    return min(total_steps, limit)


class AdaptiveMemoryGuard:
    def __init__(self, backend, limit_bytes: int):
        self.backend = backend
        self.limit_bytes = max(int(limit_bytes), MIN_TRAINING_BUDGET)
        self.backend.apply_limits(self.limit_bytes)

    def check(self, label: str) -> None:
        used = self.backend.current_memory_bytes()
        if used <= self.limit_bytes:
            return
        self.backend.clear_all()
        self.backend.synchronize()
        used = self.backend.current_memory_bytes()
        if used <= self.limit_bytes:
            return
        raise MemoryError(
            f"memory pressure at {label}: {used / 1e9:.2f}GB > {self.limit_bytes / 1e9:.2f}GB"
        )

    def log_step(self, label: str, step: int | None = None) -> None:
        snap = self.backend.memory_snapshot()
        _log_event(
            "memory",
            label=label,
            step=step,
            limit_gb=round(self.limit_bytes / 1e9, 3),
            **{k: round(v, 3) for k, v in snap.items()},
        )


_TRANSIENT_RESOURCE_BLOCKS = {
    "high_cpu_load",
    "gpu_busy",
    "gpu_memory_busy",
    "host_memory_busy",
    "low_free_vram",
    "memory_busy",
}


def _pressure_retry_limit(config) -> int:
    return max(0, int(config.get("pressure_retry_limit", 2)))


def _pressure_cooldown_seconds(config) -> float:
    return max(0.0, float(config.get("pressure_cooldown_s", 3.0)))


def _pressure_cooldown(backend, config, step: int, retry: int, *, reason: str | None = None, detail: str | None = None) -> None:
    if retry:
        _log_event("pressure_retry", backend=backend.name, step=step, retry=retry, reason=reason, detail=detail)
    backend.clear_cache()
    backend.synchronize()
    time.sleep(_pressure_cooldown_seconds(config))


def _skip(reason: str, **fields) -> dict:
    return {"status": "skipped", "reason": reason, **fields}


def _hardware_gbs(hw):
    return (
        round((hw.available_memory_bytes or 0) / 1e9, 3),
        round((getattr(hw, "system_available_memory_bytes", 0) or 0) / 1e9, 3),
    )


def _conn_db_path(conn) -> str | None:
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
    except Exception:
        return None
    return str(row[2] if not isinstance(row, dict) else row.get("file") or "") or None


def _read_tail(fh, limit=1_000_000):
    fh.seek(0, os.SEEK_END)
    fh.seek(max(0, fh.tell() - limit))
    return fh.read()


def _fresh_process_train_retry(config, conn) -> dict | None:
    if config.get("_fresh_process_retry_done"):
        return None
    db_path = _conn_db_path(conn)
    if not db_path:
        return None
    _free_mlx()
    time.sleep(3)
    retry_token = secrets.token_hex(16)
    cfg = {**_sanitize_retry_config(config), "_fresh_process_retry_done": True, "_skip_lock_once": True, "_retry_token": retry_token}
    db.secure_private_dir(TRAIN_LOG_PATH.parent)
    cfg_path = TRAIN_LOG_PATH.parent / f"retry-{secrets.token_hex(16)}.json"
    proc = None
    try:
        fd = os.open(cfg_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in cfg.items() if k != "_retry_token"}, fh)
        cmd = [
            sys.executable,
            "-c",
            (
                "import json, os, sys; "
                "from pathlib import Path; "
                "from reinforceclaw import db, trainer; "
                "cfg = json.loads(Path(sys.argv[1]).read_text()); "
                "cfg['_retry_token'] = os.environ.get('REINFORCECLAW_INTERNAL_RETRY'); "
                "conn = db.connect(Path(sys.argv[2])); "
                "print('__REINFORCECLAW_RESULT__' + json.dumps(trainer.train_result(cfg, conn)))"
            ),
            str(cfg_path),
            db_path,
        ]
        stdout_path = TRAIN_LOG_PATH.parent / f"retry-{secrets.token_hex(16)}.out"
        stderr_path = TRAIN_LOG_PATH.parent / f"retry-{secrets.token_hex(16)}.err"
        stdout_fd = os.open(stdout_path, os.O_RDWR | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600)
        stderr_fd = os.open(stderr_path, os.O_RDWR | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600)
        with os.fdopen(stdout_fd, "w+", encoding="utf-8") as stdout, os.fdopen(stderr_fd, "w+", encoding="utf-8") as stderr:
            proc = subprocess.run(
                cmd,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=float(config.get("fresh_retry_timeout_s", 21600)),
                cwd=str(Path(__file__).resolve().parents[1]),
                env={**_child_env(keep_hf=True), "REINFORCECLAW_INTERNAL_RETRY": retry_token},
            )
            proc.stdout = _read_tail(stdout)
            proc.stderr = _read_tail(stderr)
    except subprocess.TimeoutExpired as exc:
        _log_event("fresh_process_retry_timeout", timeout_s=config.get("fresh_retry_timeout_s", 21600), stderr=_scrub_secret(exc.stderr or "")[-LOG_DETAIL_CHARS:])
        return None
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        _log_event("fresh_process_retry_failed", error=type(exc).__name__, detail=_detail(exc))
        return None
    finally:
        cfg_path.unlink(missing_ok=True)
        for retry_output in (locals().get("stdout_path"), locals().get("stderr_path")):
            if retry_output:
                retry_output.unlink(missing_ok=True)
    if proc is None:
        return None
    if proc.returncode != 0:
        _log_event("fresh_process_retry_failed", returncode=proc.returncode, stderr=_scrub_secret(proc.stderr)[-LOG_DETAIL_CHARS:])
        return None
    lines = [line[len("__REINFORCECLAW_RESULT__"):] for line in proc.stdout.splitlines() if line.startswith("__REINFORCECLAW_RESULT__")]
    if not lines:
        return None
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        _log_event("fresh_process_retry_bad_json", detail=_detail(exc))
        return None
    result.setdefault("retry_mode", "fresh_process")
    return result


def _cleanup_resume_checkpoint(config, checkpoint_path=None):
    root = _resume_dir(config).resolve()
    if checkpoint_path:
        try:
            path = Path(checkpoint_path).resolve()
        except OSError:
            return
        if root in path.parents:
            shutil.rmtree(path.parent, ignore_errors=True)
        return
    if not root.exists():
        return
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)


def _resume_state(conn, config):
    state = db.get_training_state(conn)
    if not state:
        _cleanup_resume_checkpoint(config)
        return None
    expected = {
        "model": config.get("model"),
        "backend": config.get("compute_backend") or ("mlx" if platform.system() == "Darwin" else "cuda"),
        "loss_fn": str(config.get("loss_fn", "mis-po")).lower(),
        "lora_rank": int(config.get("lora_rank", 8)),
        "lora_alpha": int(config.get("lora_alpha", config.get("lora_rank", 8))),
        "lora_target": _lora_target(config),
        "max_seq_len": int(config.get("max_seq_len", 2048)),
        "grad_accum": int(config.get("grad_accum", 4)),
        "requested_steps": int(config.get("steps", 8)),
        "lr": float(config.get("lr", 8e-6)),
        "kl_coeff": float(config.get("kl_coeff", 0.001)),
        "batch_size": int(config.get("batch_size", config.get("batch_min", 32))),
        "replay_ratio": float(config.get("replay_ratio", 0.0)),
        "traj_clip": [float(x) for x in config.get("traj_clip", [0.996, 1.001])],
        "token_clip": [float(x) for x in config.get("token_clip", [0.5, 2.0])],
    }
    for key, value in expected.items():
        current = state.get(key)
        if isinstance(value, float):
            ok = abs(float(current or 0.0) - value) <= max(1e-9, abs(value) * 1e-6)
        else:
            ok = current == value
        if not ok:
            db.clear_training_state(conn)
            _cleanup_resume_checkpoint(config, state.get("checkpoint_path"))
            return None
    path = state.get("checkpoint_path")
    if path and not Path(path).exists():
        db.clear_training_state(conn)
        _cleanup_resume_checkpoint(config)
        return None
    parent = state.get("parent_version")
    if parent is not None and not db.can_activate_adapter(conn, parent):
        db.clear_training_state(conn)
        _cleanup_resume_checkpoint(config, path)
        _log_event("resume_invalidated", reason="stale_parent_adapter", parent_version=parent)
        return None
    return state


def _load_batch(conn, config, plan, resume):
    model = config.get("model")
    if resume:
        batch_ids = list(resume["batch_ids"])
        batch = db.get_feedback_by_ids(conn, batch_ids)
        fresh_ids = list(resume["fresh_ids"])
        if len(batch) != len(batch_ids):
            db.clear_training_state(conn)
            _cleanup_resume_checkpoint(config, resume.get("checkpoint_path"))
            _log_event("resume_invalidated", reason="missing_batch_feedback")
            raise ResumeInvalidated("missing_batch_feedback")
        if any(item.get("model") != model for item in batch):
            db.clear_training_state(conn)
            _cleanup_resume_checkpoint(config, resume.get("checkpoint_path"))
            _log_event("resume_invalidated", reason="model_mismatch_feedback")
            raise ResumeInvalidated("model_mismatch_feedback")
        by_id = {item["id"]: item for item in batch}
        missing_fresh = [i for i in fresh_ids if i not in by_id]
        if missing_fresh:
            db.clear_training_state(conn)
            _cleanup_resume_checkpoint(config, resume.get("checkpoint_path"))
            _log_event("resume_invalidated", reason="missing_fresh_feedback", missing=missing_fresh[:10])
            raise ResumeInvalidated("missing_fresh_feedback")
        fresh = [by_id[i] for i in fresh_ids]
        return batch, fresh_ids, fresh
    return _build_batch(
        conn,
        plan.effective_batch_size,
        config.get("replay_ratio", 0.0),
        source=_feedback_source(config),
        model=model,
    )


def _paused_result(backend_name, cfg, batch, total_steps, remaining_steps, config, checkpoint_path, parent_version):
    return {
        "status": "paused",
        "reason": "resume_pending",
        "remaining_steps": remaining_steps,
        "batch_size": len(batch),
        "steps": total_steps,
        "checkpoint_path": checkpoint_path,
        "parent_version": parent_version,
        "backend": backend_name,
        "loss_fn": cfg["loss_fn"],
        "lora_target": _lora_target(config),
    }


def _update_ema_from_fresh(ema_mean, ema_count, fresh, decay):
    """EMA tracks newly rated feedback only; replay rows already affected it when first seen."""
    for item in fresh:
        if ema_count <= 0:
            ema_count, ema_mean = 1, float(item["rating"])
            continue
        ema_count += 1
        ema_mean = decay * ema_mean + (1 - decay) * item["rating"]
    return ema_mean, ema_count


def _finalize_training(conn, config, backend, cfg, batch, fresh_ids, adapter_path,
                       ema_mean, ema_count, total_loss, total_steps, new_v, parent_v,
                       resume_checkpoint=None):
    _verify_saved_adapter(config, adapter_path)
    metrics = {
        "status": "trained",
        "avg_loss": total_loss / max(total_steps, 1),
        "batch_size": len(batch),
        "steps": total_steps,
        "ema_mean": ema_mean,
        "ema_count": ema_count,
        "peak_memory_gb": round(backend.peak_memory_bytes() / 1e9, 3),
        "backend": backend.name,
        "model": config["model"],
        "loss_fn": cfg["loss_fn"],
        "lora_target": _lora_target(config),
        "version": new_v,
        "path": adapter_path,
        "feedback_ids": fresh_ids,
    }
    db.record_training_round(conn, ema_mean, ema_count, new_v, adapter_path, parent_v, metrics, fresh_ids, clear_state=True)
    _cleanup_resume_checkpoint(config, resume_checkpoint)
    if config.get("adapter_keep"):
        root = _adapter_dir(config).resolve()
        active_dir = Path(adapter_path).parent.resolve()
        for version, path in db.cleanup_adapters(conn, keep=config["adapter_keep"], model=config.get("model")):
            raw_dir = Path(path).expanduser().parent
            if raw_dir.is_symlink():
                _log_event("adapter_cleanup_rejected", version=version, path=str(raw_dir), reason="symlink")
                continue
            old_dir = raw_dir.resolve()
            if old_dir.parent != root or old_dir.name != f"v{version}":
                _log_event("adapter_cleanup_rejected", version=version, path=str(old_dir), reason="outside_adapter_root")
                continue
            if old_dir == active_dir:
                continue
            try:
                shutil.rmtree(old_dir)
            except FileNotFoundError:
                db.delete_adapter(conn, version)
            except OSError as exc:
                _log_event("adapter_cleanup_failed", version=version, path=str(old_dir), detail=_detail(exc))
            else:
                db.delete_adapter(conn, version)
    return metrics


def _training_state_payload(config, cfg, backend_name, batch, fresh_ids, resume, latest,
                            run_id, checkpoint_path, remaining_steps, total_target_steps):
    return {
        "run_id": run_id,
        "model": config["model"],
        "backend": backend_name,
        "loss_fn": str(cfg["loss_fn"]).lower(),
        "lora_rank": int(config.get("lora_rank", 8)),
        "lora_alpha": int(config.get("lora_alpha", config.get("lora_rank", 8))),
        "lora_target": _lora_target(config),
        "max_seq_len": int(config.get("max_seq_len", 2048)),
        "grad_accum": int(config.get("grad_accum", 4)),
        "requested_steps": int(config.get("steps", 8)),
        "lr": float(config.get("lr", 8e-6)),
        "kl_coeff": float(cfg.get("kl_coeff", 0.001)),
        "batch_size": int(config.get("batch_size", config.get("batch_min", 32))),
        "replay_ratio": float(config.get("replay_ratio", 0.0)),
        "traj_clip": [float(x) for x in cfg["traj_clip"]],
        "token_clip": [float(x) for x in cfg["token_clip"]],
        "checkpoint_path": checkpoint_path,
        "batch_ids": [item["id"] for item in batch],
        "fresh_ids": fresh_ids,
        "remaining_steps": remaining_steps,
        "total_steps": total_target_steps,
        "parent_version": resume.get("parent_version") if resume else (latest["version"] if latest else None),
    }


def _build_train_cfg(config, plan: TrainingPlan):
    defaults = {
        "loss_fn": "mis-po",
        "steps": plan.steps,
        "lr": 8e-6,
        "token_clip": [0.5, 2.0],
        "traj_clip": [0.996, 1.001],
        "kl_coeff": 0.001,
        "grad_accum": plan.grad_accum,
        "grad_clip": 1.0,
        "ema_decay": 0.99,
        "pos_weight": 1.0,
        "adv_clip": 2.0,
        "adv_norm": True,
        "max_passes": 1.0,
        "adapter_keep": 20,
    }
    return {key: config.get(key, value) for key, value in defaults.items()}


def _tighten_small_batch_cfg(cfg: dict, batch_size: int) -> dict:
    if batch_size <= 2:
        cfg = {**cfg, "grad_accum": 1, "steps": min(cfg["steps"], max(2, batch_size * 4))}
    elif batch_size <= 4:
        cfg = {**cfg, "grad_accum": min(cfg["grad_accum"], 2), "steps": min(cfg["steps"], batch_size * 4)}
    return cfg


def _trajectory_scale_mlx(delta_mean, cfg):
    low, high = cfg["traj_clip"]
    ratio = mx.maximum(delta_mean, mx.array(1e-6))
    return mx.minimum(mx.array(1.0), mx.minimum(ratio / mx.array(low), mx.array(high) / ratio))


def _trajectory_scale_torch(delta_mean, cfg, torch):
    low, high = cfg["traj_clip"]
    ratio = torch.clamp(delta_mean, min=1e-6)
    one = torch.tensor(1.0, device=delta_mean.device, dtype=delta_mean.dtype)
    lo = torch.tensor(low, device=delta_mean.device, dtype=delta_mean.dtype)
    hi = torch.tensor(high, device=delta_mean.device, dtype=delta_mean.dtype)
    return torch.minimum(one, torch.minimum(ratio / lo, hi / ratio))


def _safe_total_memory_bytes(hardware) -> int:
    cap = getattr(hardware, "recommended_working_set_bytes", None)
    return min(hardware.total_memory_bytes, cap) if cap else hardware.total_memory_bytes


def _preload_limit_bytes(hardware, reserve_bytes: int) -> int:
    if hardware.unified_memory:
        available = hardware.available_memory_bytes or hardware.total_memory_bytes
        safe_total = _safe_total_memory_bytes(hardware)
        safe_cap = min(
            max(available - reserve_bytes // 2, MIN_TRAINING_BUDGET),
            max(safe_total - reserve_bytes, MIN_TRAINING_BUDGET),
        )
        return int(safe_cap)
    available = hardware.available_memory_bytes or hardware.total_memory_bytes
    return max(MIN_TRAINING_BUDGET, min(int(hardware.total_memory_bytes * 0.85), int(available * 0.9)))


def _training_budget_bytes(hardware, reserve_bytes: int, model_bytes: int) -> int:
    if hardware.unified_memory:
        available = hardware.available_memory_bytes or hardware.total_memory_bytes
        free_budget = max(0, available - reserve_bytes)
        cap_budget = max(0, _safe_total_memory_bytes(hardware) - reserve_bytes - max(model_bytes, 0))
        return min(free_budget, cap_budget)

    device_reserve = max(1 * GB, int(hardware.total_memory_bytes * 0.10))
    device_available = hardware.available_memory_bytes or hardware.total_memory_bytes
    device_budget = max(0, device_available - device_reserve - max(model_bytes, 0))
    host_available = getattr(hardware, "system_available_memory_bytes", None) or getattr(hardware, "system_total_memory_bytes", 0)
    host_budget = max(0, host_available - reserve_bytes)
    return min(device_budget, host_budget) if host_budget else device_budget


def _next_adapter_version(conn) -> int:
    row = conn.execute("SELECT COALESCE(MAX(version), 0) AS version FROM adapters").fetchone()
    return int(row["version"]) + 1


def _plan_strategy(config, hardware, model_bytes: int) -> TrainingPlan | None:
    reserve_bytes = _os_reserve_bytes(hardware.unified_memory)
    busy = _load_ratio() > float(config.get("idle_load_threshold", BACKGROUND_IDLE_LOAD))
    available = hardware.available_memory_bytes
    if available is not None and available < int(reserve_bytes * 1.25):
        busy = True

    budget = _training_budget_bytes(hardware, reserve_bytes, model_bytes)
    if budget < MIN_TRAINING_BUDGET:
        return None

    batch_cap = max(1, int(config.get("batch_size", config.get("batch_min", 32))))
    base_accum = max(1, config.get("grad_accum", 4))
    base_steps = max(1, config.get("steps", 8))
    preload = _preload_limit_bytes(hardware, reserve_bytes)

    if budget < 4 * GB:
        batch_cap = min(batch_cap, 1)
        grad_accum = 1
        steps = min(base_steps, 2)
        aggressive = True
    elif budget < 8 * GB:
        batch_cap = min(batch_cap, 4)
        grad_accum = min(base_accum, 2)
        steps = min(base_steps, 3)
        aggressive = True
    else:
        batch_cap = min(batch_cap, 8 if budget < 12 * GB else batch_cap)
        grad_accum = base_accum
        steps = base_steps
        aggressive = False

    floor = model_bytes + min(budget, MIN_TRAINING_BUDGET)
    limit = max(floor, min(preload, model_bytes + max(MIN_TRAINING_BUDGET, int(budget * 0.9))))
    return TrainingPlan(
        effective_batch_size=max(1, batch_cap),
        grad_accum=max(1, grad_accum),
        steps=max(1, steps),
        memory_limit_bytes=max(limit, MIN_TRAINING_BUDGET),
        training_budget_bytes=budget,
        resident_model_bytes=max(0, model_bytes),
        aggressive_checkpointing=aggressive,
        busy=busy,
    )


def _degrade_plan(plan: TrainingPlan | None) -> TrainingPlan | None:
    if plan is None:
        return None
    if plan.effective_batch_size == 1 and plan.grad_accum == 1:
        return None
    floor = plan.resident_model_bytes + MIN_TRAINING_BUDGET
    return replace(
        plan,
        effective_batch_size=max(1, plan.effective_batch_size // 2),
        grad_accum=max(1, plan.grad_accum // 2),
        steps=max(1, min(plan.steps, 2)),
        memory_limit_bytes=max(floor, int(plan.memory_limit_bytes * 0.9)),
        aggressive_checkpointing=True,
        busy=True,
    )


def _build_batch(conn, batch_size: int, replay_ratio: float, source: str | None = None, model: str | None = None):
    batch_size = max(1, int(batch_size))
    replay_target = min(batch_size, max(0, int(round(batch_size * max(0.0, float(replay_ratio))))))
    all_fresh = db.get_untrained(conn, limit=batch_size, source=source, model=model)
    if all_fresh and replay_target >= batch_size:
        replay_target = batch_size - 1
    fresh_target = max(0, batch_size - replay_target)
    all_replay = db.get_replay(conn, limit=batch_size, source=source, model=model) if replay_target else []
    fresh = all_fresh[:fresh_target] if fresh_target else []
    fresh_seen = {item["id"] for item in fresh}
    replay = [item for item in all_replay if item["id"] not in fresh_seen][:replay_target] if replay_target else []
    missing = batch_size - (len(fresh) + len(replay))
    if missing > 0:
        fresh.extend(all_fresh[len(fresh):len(fresh) + missing])
        fresh_seen = {item["id"] for item in fresh}
    missing = batch_size - (len(fresh) + len(replay))
    if missing > 0:
        replay_seen = {item["id"] for item in replay}
        replay.extend(item for item in all_replay if item["id"] not in fresh_seen and item["id"] not in replay_seen)
        replay = replay[:batch_size - len(fresh)]
    return fresh + replay, [item["id"] for item in fresh], fresh


def _release_backend_memory(backend) -> None:
    backend.clear_all()
    gc.collect()
    try:
        backend.synchronize()
    except Exception:
        pass
    if backend.name == "mlx":
        time.sleep(0.25)


def _lora_target(config) -> str:
    return str(config.get("lora_target", "attention")).lower()


_ATTENTION_LEAVES = {"q_proj", "k_proj", "v_proj", "o_proj", "wq", "wk", "wv", "wo", "c_attn", "c_proj"}
_MLP_LEAVES = {"gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"}


def _strict_target_selection(target: str) -> bool:
    # This is intentionally strict for all non-"all" LoRA targets.
    # It matters most for MoE models because falling back to every linear layer can
    # accidentally LoRA expert/router-adjacent weights, but the same silent fallback
    # is also unsafe on dense models.
    return target != "all"


def _mlx_lora_keys(model, target: str):
    if target == "all":
        return None
    allowed = _ATTENTION_LEAVES if target == "attention" else _ATTENTION_LEAVES | _MLP_LEAVES
    keys = set()
    layers = getattr(getattr(model, "model", None), "layers", None) or getattr(model, "layers", None) or []
    for layer in layers:
        for path, _module in layer.named_modules():
            if path.rsplit(".", 1)[-1] in allowed:
                keys.add(path)
    return keys


def _apply_lora(model, rank: int, target: str = "attention"):
    _ensure_mlx()
    model.freeze()
    layers = getattr(getattr(model, "model", None), "layers", None) or getattr(model, "layers", None) or []
    cfg = {"rank": rank, "scale": 1.0, "dropout": 0.0}
    keys = _mlx_lora_keys(model, target)
    if _strict_target_selection(target) and not keys:
        raise RuntimeError(
            f"no_{target}_modules_found_for_lora:strict_targeting_enabled"
        )
    if keys is not None:
        cfg["keys"] = keys
    linear_to_lora_layers(
        model,
        len(layers),
        cfg,
    )
    return model


def _disable_lora(model):
    saved = {
        name: param
        for name, param in nn.utils.tree_flatten(model.trainable_parameters())
        if "lora" in name.lower()
    }
    if not saved:
        raise RuntimeError("lora_disable_failed:no_lora_weights")
    zeros = {key: mx.zeros_like(value) for key, value in saved.items()}
    model.load_weights(list(zeros.items()), strict=False)
    mx.eval(model.parameters())
    return saved


def _enable_lora(model, saved_weights):
    if saved_weights:
        model.load_weights(list(saved_weights.items()), strict=False)
        mx.eval(model.parameters())


def _enable_grad_checkpoint_mlx(model):
    if not hasattr(mx, "checkpoint"):
        return lambda: None
    try:
        layers = getattr(getattr(model, "model", None), "layers", None) or getattr(model, "layers", None)
        if not layers:
            return lambda: None
        layer_cls = type(layers[0])
        original_call = getattr(layer_cls, "__reinforceclaw_original_call__", layer_cls.__call__)
        if getattr(layer_cls, "_reinforceclaw_checkpointed", False):
            return lambda: None

        def checkpointed_call(self, *args, **kwargs):
            def inner(params, *inner_args, **inner_kwargs):
                self.update(params)
                return original_call(self, *inner_args, **inner_kwargs)

            return mx.checkpoint(inner)(self.trainable_parameters(), *args, **kwargs)

        layer_cls.__reinforceclaw_original_call__ = original_call
        layer_cls.__call__ = checkpointed_call
        layer_cls._reinforceclaw_checkpointed = True
        def restore():
            if getattr(layer_cls, "_reinforceclaw_checkpointed", False):
                layer_cls.__call__ = layer_cls.__reinforceclaw_original_call__
                layer_cls._reinforceclaw_checkpointed = False
        return restore
    except Exception:
        return lambda: None


def _safe_chat_messages(messages, *, limit=16):
    safe = []
    for msg in messages if isinstance(messages, list) else []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").lower()
        if role not in {"system", "user", "assistant"}:
            continue
        content = db.scrub_text(msg.get("content", "")).strip()
        if content:
            safe.append({"role": role, "content": content})
    return safe[-limit:]


def _build_chat_messages(ctx, prompt):
    msgs = _safe_chat_messages(ctx.get("messages")) if isinstance(ctx, dict) else []
    if msgs:
        return msgs
    msgs = []
    if isinstance(ctx, dict) and "system" in ctx:
        system = db.scrub_text(ctx["system"]).strip()
        if system:
            msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def _chat_text_pair(tokenizer, item):
    """Return (prompt_text, full_text) for an item, or None if tokenizer has no chat template."""
    prompt, response = item["prompt"], item["response"]
    ctx = _context_dict(item)
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = _build_chat_messages(ctx, prompt)
        prompt_text = _apply_chat_template(tokenizer, msgs, add_generation_prompt=True, tokenize=False)
        full_text = _apply_chat_template(
            tokenizer, msgs + [{"role": "assistant", "content": response}],
            add_generation_prompt=False, tokenize=False,
        )
        return prompt_text, full_text
    if isinstance(ctx, dict) and isinstance(ctx.get("messages"), list):
        messages = _build_chat_messages(ctx, prompt)
        prompt_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
        return prompt_text, prompt_text + "\nAssistant: " + response
    prompt_text = f"User: {prompt}\nAssistant:"
    return prompt_text, f"{prompt_text} {response}"


def _tokenize_mlx(tokenizer, item):
    max_seq_len = max(128, int(item.get("max_seq_len") or 2048))
    prompt_text, full_text = _chat_text_pair(tokenizer, item)
    prompt_ids = _plain_token_ids(tokenizer, prompt_text)
    full_ids_all = _plain_token_ids(tokenizer, full_text)
    offset = max(0, len(full_ids_all) - max_seq_len)
    full_ids = full_ids_all[offset:]
    out = {
        "input_ids": mx.array(full_ids),
        "response_start": _response_start_from_prefix(prompt_ids, full_ids_all, offset),
        "rating": item["rating"],
        "id": item["id"],
    }
    for key in ("context", "rollout_context"):
        if item.get(key):
            out[key] = item[key]
    return out


def _compute_logprobs_mlx(model, input_ids):
    logits = model(input_ids[None, :]).squeeze(0)
    lp = nn.log_softmax(logits, axis=-1)
    return mx.take_along_axis(lp[:-1], input_ids[1:, None], axis=-1).squeeze(-1)


def _fallback_chat_text(messages, *, add_generation_prompt=False):
    parts = [f"{str(m.get('role', 'user')).capitalize()}: {str(m.get('content', ''))}" for m in messages]
    if add_generation_prompt:
        parts.append("Assistant:")
    return "\n".join(parts)


def _apply_chat_template(tokenizer, messages, *, add_generation_prompt=False, tokenize=True):
    kwargs = {"add_generation_prompt": add_generation_prompt, "tokenize": tokenize}
    if add_generation_prompt:
        kwargs["enable_thinking"] = False
    for attempt in (kwargs, {k: v for k, v in kwargs.items() if k != "enable_thinking"}):
        try:
            return tokenizer.apply_chat_template(messages, **attempt)
        except TypeError:
            continue
        except ValueError as exc:
            if "chat_template is not set" not in str(exc):
                raise
        except ImportError:
            _log_event("chat_template_import_failed")
        break
    fallback = _fallback_chat_text(messages, add_generation_prompt=add_generation_prompt)
    return _plain_token_ids(tokenizer, fallback) if tokenize else fallback


def _context_dict(item):
    context = item.get("rollout_context") or item.get("context")
    if not context:
        return None
    try:
        return json.loads(context)
    except (json.JSONDecodeError, TypeError) as exc:
        _log_event("context_json_ignored", feedback_id=item.get("id"), detail=_detail(exc))
        return None


def _plain_token_ids(tokenizer, text):
    if isinstance(text, (list, tuple)):
        return [int(tok) for tok in text]
    if hasattr(tokenizer, "__call__"):
        try:
            encoded = tokenizer(text, add_special_tokens=False)
            ids = encoded["input_ids"] if isinstance(encoded, dict) else getattr(encoded, "input_ids", encoded)
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return [int(tok) for tok in ids]
        except Exception as exc:
            call_error = exc
    if hasattr(tokenizer, "encode"):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(text)
        return [int(tok) for tok in ids]
    detail = f"; __call__ failed: {type(call_error).__name__}: {call_error}" if "call_error" in locals() else ""
    raise TypeError(f"tokenizer cannot produce plain token ids{detail}")


def _response_start_from_prefix(prompt_ids, full_ids, offset=0) -> int:
    common = 0
    for prompt_tok, full_tok in zip(prompt_ids, full_ids):
        if int(prompt_tok) != int(full_tok):
            break
        common += 1
    return max(1, common - int(offset))


def _behavior_logprobs(item, length: int, xp_name: str, torch_module=None):
    ctx = _context_dict(item)
    if not isinstance(ctx, dict):
        return None
    values = ctx.get("behavior_logprobs")
    if not isinstance(values, list):
        return None
    if len(values) == length + 1:
        values = values[1:]
    elif len(values) != length:
        _log_event("behavior_logprobs_ignored", feedback_id=item.get("id"), expected=length, got=len(values))
        return None
    if xp_name == "mlx":
        return mx.array(values)
    device = item["input_ids"].device
    return torch_module.tensor(values, device=device, dtype=torch_module.float32)


def _scalar_advantage(rating: int, ema_mean: float, cfg) -> float:
    clip = float(cfg.get("adv_clip", 2.0))
    return max(-clip, min(clip, _raw_advantage(rating, ema_mean, cfg)))


def _raw_advantage(rating: int, ema_mean: float, cfg) -> float:
    adv = float(rating - ema_mean)
    return adv * float(cfg.get("pos_weight", 1.0)) if rating > 0 else adv


def _attach_advantages(items, ema_mean, cfg):
    vals = [_raw_advantage(item["rating"], ema_mean, cfg) for item in items]
    if len(vals) > 1 and len({item["rating"] for item in items}) > 1 and cfg.get("adv_norm", True):
        mean = sum(vals) / len(vals)
        std = statistics.stdev(vals) or 1.0
        vals = [(v - mean) / (std + 1e-6) for v in vals]
    clip = float(cfg.get("adv_clip", 2.0))
    for item, value in zip(items, vals):
        item["advantage"] = max(-clip, min(clip, value))
    return items


def _aggregate_diag(samples: list[dict]) -> dict:
    """Average per-sample diagnostics into per-step scalars for logging."""
    if not samples:
        return {}
    keys = set().union(*(sample.keys() for sample in samples))
    out = {}
    for k in keys:
        vals = [s[k] for s in samples if k in s]
        if vals:
            out[k] = round(statistics.mean(vals), 6)
    # advantage_std is meaningful only across a batch — override the per-sample 0s.
    adv = [s["advantage_mean"] for s in samples if "advantage_mean" in s]
    if len(adv) > 1:
        out["advantage_std"] = round(statistics.pstdev(adv), 6)
    return out


def _make_loss_fn_mlx(model, tokenized, ref_logprobs, cfg, ema_mean, diag_sink=None, behavior_logprobs=None):
    tc_lo, tc_hi = cfg["token_clip"]
    kl_c = cfg["kl_coeff"]
    loss_name = str(cfg.get("loss_fn", "mis-po")).lower()
    traj_lo, traj_hi = cfg["traj_clip"]

    def loss_fn(idx: int):
        item = tokenized[idx]
        ids, start, rating = item["input_ids"], item["response_start"], item["rating"]
        ref = ref_logprobs[idx]
        logits = model(ids[None, :]).squeeze(0)
        new_lp = mx.take_along_axis(
            nn.log_softmax(logits, axis=-1)[:-1],
            ids[1:, None],
            axis=-1,
        ).squeeze(-1)
        new_r, ref_r = new_lp[start - 1:], ref[start - 1:]
        if int(new_r.shape[0]) == 0 or int(ref_r.shape[0]) == 0:
            return mx.sum(logits) * 0.0
        adv_scalar = item.get("advantage", _scalar_advantage(rating, ema_mean, cfg))
        adv = mx.array(adv_scalar, dtype=new_r.dtype)

        # Diagnostics (no-grad, scalar floats) — captured via sink so the trainer
        # can aggregate them per step without changing loss_fn's signature.
        if diag_sink is not None:
            delta = new_r - ref_r
            traj_val = float(mx.exp(mx.mean(delta)).item())
            traj_pass = 1.0 if traj_lo <= traj_val <= traj_hi else 0.0
            # token_mask_rate = fraction of ratio values inside [tc_lo, tc_hi].
            ratio_for_diag = mx.exp(delta)
            inside = (ratio_for_diag >= tc_lo) & (ratio_for_diag <= tc_hi)
            token_mask_rate = float(mx.mean(inside.astype(new_r.dtype)).item())
            diag_sink.append({
                "policy_kl": float(mx.mean(delta).item()),
                "ratio_mean": float(mx.mean(ratio_for_diag).item()),
                "ratio_p5": float(mx.min(ratio_for_diag).item()),   # approx — mlx has no percentile
                "ratio_p95": float(mx.max(ratio_for_diag).item()),  # approx
                "advantage_mean": adv_scalar,
                "advantage_std": 0.0,  # scalar per sample — std computed at batch level
                "pct_zero_adv": 1.0 if adv_scalar == 0.0 else 0.0,
                "traj_gate_pass_rate": traj_pass,
                "token_mask_rate": token_mask_rate,
            })

        if loss_name == "reinforce++":
            per_token_kl = new_r - ref_r
            adjusted = mx.stop_gradient(adv - kl_c * per_token_kl)
            return -mx.mean(new_r * adjusted)
        behavior_r = _behavior_logprobs(item, int(new_r.shape[0]), "mlx") if loss_name == "mis-po" else ref_r
        if behavior_r is None:
            behavior_r = behavior_logprobs[idx][start - 1:] if behavior_logprobs is not None else ref_r

        ratio = mx.exp(new_r - behavior_r)
        clipped_ratio = mx.clip(ratio, tc_lo, tc_hi)
        surr_1 = ratio * adv
        surr_2 = clipped_ratio * adv
        actor = -mx.mean(mx.minimum(surr_1, surr_2))
        kl = mx.exp(ref_r - new_r) - (ref_r - new_r) - 1
        traj = mx.exp(mx.mean(new_r - ref_r))
        return actor * _trajectory_scale_mlx(traj, cfg) + kl_c * mx.mean(kl)

    return loss_fn


def _torch_stack():
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(f"missing_cuda_training_dependency:{exc}") from exc

    return torch, AutoModelForCausalLM, AutoTokenizer, LoraConfig, PeftModel, get_peft_model


def _optimizer_file(adapter_path: str | Path) -> Path:
    return Path(adapter_path).parent / "optimizer.safetensors"


def _save_mlx_optimizer(opt, adapter_path) -> None:
    mx.save_safetensors(str(_optimizer_file(adapter_path)), dict(nn.utils.tree_flatten(opt.state)))


def _load_mlx_optimizer(opt, adapter_path) -> None:
    path = _optimizer_file(adapter_path)
    if path.exists():
        try:
            opt.state = tree_unflatten(list(mx.load(str(path)).items()))
        except (RuntimeError, OSError, ValueError) as exc:
            _log_event("optimizer_resume_skipped", backend="mlx", detail=_detail(exc))


def _save_torch_optimizer(torch, opt, adapter_path) -> None:
    torch.save(opt.state_dict(), str(_optimizer_file(adapter_path).with_suffix(".pt")))


def _load_torch_optimizer(torch, opt, adapter_path) -> None:
    path = _optimizer_file(adapter_path).with_suffix(".pt")
    if not path.exists():
        return
    try:
        opt.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
    except (TypeError, RuntimeError, ValueError) as exc:
        _log_event("optimizer_resume_skipped", backend="cuda", detail=_detail(exc))


def _require_finite(value, reason="nonfinite_loss") -> float:
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(reason)
    return value


def _verify_saved_adapter(config, adapter_path: str) -> str:
    gate = publish_gate(config, adapter_path)
    if not gate.get("ok"):
        shutil.rmtree(Path(adapter_path).parent, ignore_errors=True)
        raise RuntimeError(gate.get("reason", "adapter_publish_gate_failed"))
    return adapter_path


def _save_partial_cuda(model, config, run_id: str, torch=None, optimizer=None) -> str:
    target_dir = _resume_checkpoint_dir(config, run_id)
    model.save_pretrained(target_dir, safe_serialization=True)
    adapter_path = next(target_dir.glob("*.safetensors"), None)
    if adapter_path is None:
        raise RuntimeError("partial_adapter_save_missing_safetensors")
    adapter = str(adapter_path)
    _verify_saved_adapter(config, adapter)
    if torch is not None and optimizer is not None:
        _save_torch_optimizer(torch, optimizer, adapter)
    return adapter


def _load_torch_adapter_weights(torch, model, adapter_path: str | Path) -> None:
    adapter_path = Path(adapter_path)
    if adapter_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(adapter_path), device="cpu")
    else:
        root = (Path.home() / ".reinforceclaw" / "adapters").resolve()
        if root not in adapter_path.resolve().parents:
            raise RuntimeError("Refusing non-safetensors adapter outside ~/.reinforceclaw/adapters")
        try:
            state_dict = torch.load(str(adapter_path), map_location="cpu", weights_only=True)
        except TypeError:
            raise RuntimeError("Non-safetensors adapters require PyTorch with weights_only=True")
    try:
        from peft.utils.save_and_load import set_peft_model_state_dict

        try:
            set_peft_model_state_dict(model, state_dict, adapter_name="default")
            return
        except TypeError:
            set_peft_model_state_dict(model, state_dict)
            return
        except RuntimeError as exc:
            _log_event("adapter_load_failed", detail=_detail(exc), path=str(adapter_path))
            raise
    except ImportError:
        pass
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    if not compatible:
        raise RuntimeError(f"adapter_no_compatible_weights:{adapter_path}")
    if len(compatible) != len(state_dict):
        _log_event("adapter_partial_load_blocked", path=str(adapter_path), loaded=len(compatible), total=len(state_dict))
        raise RuntimeError(f"adapter_incompatible_weights:{adapter_path}")
    model.load_state_dict(compatible, strict=False)


def _torch_target_modules(torch, model, target: str = "attention"):
    cache = getattr(model, _TORCH_TARGET_CACHE_ATTR, None)
    if isinstance(cache, dict) and target in cache:
        return cache[target]
    common = set()
    exact = set()
    exact_preferred = set()
    seen = set()
    # Handle "all" and "all_linear" to include ALL linear modules (including experts)
    if target in ("all", "all_linear"):
        preferred = None  # None means all linear modules
    elif target == "attention":
        preferred = _ATTENTION_LEAVES
    else:
        preferred = _ATTENTION_LEAVES | _MLP_LEAVES
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        parts = name.split(".")
        leaf = parts[-1]
        parent_leaf = parts[-2] if len(parts) >= 2 else ""
        seen.add(leaf)
        if any(part in _TORCH_EXCLUDED_MODULES for part in parts):
            continue
        # If preferred is None (target="all" or "all_linear"), accept all linear modules
        if preferred is None:
            exact.add(name)
        elif leaf in preferred:
            common.add(leaf)
            exact.add(name)
        elif leaf == "linear" and parent_leaf in preferred:
            exact_preferred.add(name)
    target_modules = exact_preferred or exact or common
    if _strict_target_selection(target) and not target_modules:
        raise RuntimeError(
            f"no_{target}_modules_found_for_lora:strict_targeting_enabled:{','.join(sorted(seen))}"
        )
    result = sorted(target_modules)
    try:
        cache = cache if isinstance(cache, dict) else {}
        cache[target] = result
        setattr(model, _TORCH_TARGET_CACHE_ATTR, cache)
    except Exception:
        pass
    return result


def _tokenize_torch(tokenizer, item, torch_device):
    max_seq_len = max(128, int(item.get("max_seq_len") or 2048))
    prompt_text, full_text = _chat_text_pair(tokenizer, item)
    prompt_batch = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    full_batch = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    prompt_ids = prompt_batch["input_ids"][0]
    full_ids_all = full_batch["input_ids"][0]
    offset = max(0, int(full_ids_all.shape[0]) - max_seq_len)
    full_ids = full_ids_all[offset:]
    tokenized = {
        "input_ids": full_ids,
        "response_start": _response_start_from_prefix(prompt_ids.tolist(), full_ids_all.tolist(), offset),
        "rating": item["rating"],
        "id": item["id"],
    }
    for key in ("context", "rollout_context"):
        if item.get(key):
            tokenized[key] = item[key]
    for key, value in full_batch.items():
        if key == "input_ids" or not hasattr(value, "shape"):
            continue
        tokenized[key] = value[0][offset:]
    return tokenized


def _response_token_count(item) -> int:
    return max(0, int(item["input_ids"].shape[0]) - int(item["response_start"]))


def _filter_tokenized_pairs(conn, batch, pairs, backend_name):
    kept = [(item, tok) for item, tok in pairs if _response_token_count(tok) > 0]
    dropped = [item["id"] for item, tok in pairs if _response_token_count(tok) <= 0]
    if dropped:
        db.ignore_feedback_ids(conn, dropped)
        _log_event("feedback_ignored", backend=backend_name, reason="no_response_tokens", count=len(dropped), ids=dropped[:10])
    if not kept:
        return None, _skip("no_response_tokens", backend=backend_name, ignored=len(dropped))
    return kept, None


def _micro_indices(n_items: int, n_steps: int, grad_accum: int) -> list[int]:
    total = max(0, int(n_steps)) * max(1, int(grad_accum))
    if n_items <= 0 or total <= 0:
        return []
    rng = random.Random(secrets.randbits(64))
    order, out = list(range(n_items)), []
    while len(out) < total:
        rng.shuffle(order)
        out.extend(order)
    return out[:total]


def _torch_move_item(item, device):
    return {
        key: value if getattr(value, "device", None) == device else (
            value.to(device, non_blocking=True) if hasattr(value, "to") else value
        )
        for key, value in item.items()
    }


def _torch_model_type(model):
    for cfg in (
        getattr(model, "config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(getattr(model, "base_model", None), "config", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None),
    ):
        if model_type := getattr(cfg, "model_type", None):
            return str(model_type).lower()
    return None


def _torch_forward_params(model):
    cached = getattr(model, "_reinforceclaw_forward_params", None)
    if cached is not None:
        return cached
    try:
        sig = inspect.signature(model.forward)
        names = set(sig.parameters)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except (TypeError, ValueError):
        names, accepts_kwargs = set(), True
    cached = (names, accepts_kwargs)
    try:
        setattr(model, "_reinforceclaw_forward_params", cached)
    except Exception:
        pass
    return cached


def _torch_accepts(model, key):
    names, accepts_kwargs = _torch_forward_params(model)
    return accepts_kwargs or key in names


def _torch_forward_kwargs(model, item, torch):
    ids = item["input_ids"].unsqueeze(0)
    kwargs = {"input_ids": ids}
    attention_mask = item.get("attention_mask")
    if attention_mask is not None and _torch_accepts(model, "attention_mask"):
        kwargs["attention_mask"] = attention_mask.unsqueeze(0)
    for key, value in item.items():
        if key.startswith("mm_") and key not in kwargs and hasattr(value, "unsqueeze") and _torch_accepts(model, key):
            kwargs[key] = value.unsqueeze(0)
    if (
        (_torch_model_type(model) or "") in {"gemma4", "gemma5"}
        and "mm_token_type_ids" not in kwargs
        and _torch_accepts(model, "mm_token_type_ids")
    ):
        kwargs["mm_token_type_ids"] = torch.zeros_like(ids)
    return kwargs


def _compute_logprobs_torch(model, item, torch):
    with torch.no_grad():
        device = next(model.parameters()).device
        device_item = _torch_move_item(item, device)
        logits = model(**_torch_forward_kwargs(model, device_item, torch)).logits.squeeze(0)
        lp = torch.log_softmax(logits, dim=-1)
        ids = device_item["input_ids"]
        return lp[:-1].gather(-1, ids[1:].unsqueeze(-1)).squeeze(-1)


@contextmanager
def _torch_adapter_disabled(model, torch):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
        return
    saved = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                saved.append((param, param.detach().clone()))
                param.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for param, value in saved:
                param.copy_(value)



_FLOAT_EXP_MASKS = {
    "F16": (2, 0x7C00),
    "BF16": (2, 0x7F80),
    "F32": (4, 0x7F800000),
    "F64": (8, 0x7FF0000000000000),
}


def _scan_tensor_bytes(fh, remaining: int, dtype: str):
    itemsize, exp_mask = _FLOAT_EXP_MASKS.get(str(dtype).upper(), (0, 0))
    if itemsize and remaining % itemsize:
        return False, False, "adapter_safetensors_bad_dtype_size"
    has_nonzero, carry = False, b""
    while remaining > 0:
        chunk = fh.read(min(65536, remaining))
        if not chunk:
            return has_nonzero, False, "adapter_safetensors_truncated"
        has_nonzero = has_nonzero or any(chunk)
        remaining -= len(chunk)
        if itemsize:
            data = carry + chunk
            usable = len(data) - (len(data) % itemsize)
            for pos in range(0, usable, itemsize):
                if int.from_bytes(data[pos:pos + itemsize], "little") & exp_mask == exp_mask:
                    return has_nonzero, False, "adapter_nonfinite_tensor"
            carry = data[usable:]
    return has_nonzero, True, ""


def publish_gate(config, adapter_path):
    path = Path(adapter_path)
    size = path.stat().st_size if path.exists() and not path.is_symlink() else 0
    if size <= 0:
        return {"ok": False, "reason": "adapter_missing_or_empty"}
    if path.suffix == ".safetensors":
        try:
            with path.open("rb") as fh:
                header_len = struct.unpack("<Q", fh.read(8))[0]
                if header_len <= 0 or header_len > min(size - 8, 1_000_000):
                    return {"ok": False, "reason": "adapter_safetensors_bad_header"}
                header = json.loads(fh.read(header_len))
                if not isinstance(header, dict):
                    return {"ok": False, "reason": "adapter_safetensors_invalid"}
            tensors = [key for key in header if key != "__metadata__"]
            if not tensors or size <= 8 + header_len:
                return {"ok": False, "reason": "adapter_safetensors_empty"}
            has_nonzero = False
            with path.open("rb") as fh:
                for key in tensors:
                    info = header.get(key) or {}
                    shape = info.get("shape") or []
                    offsets = info.get("data_offsets") or []
                    dtype = info.get("dtype", "")
                    if (
                        not isinstance(info, dict)
                        or not shape
                        or not isinstance(offsets, list)
                        or len(offsets) != 2
                        or not all(isinstance(x, int) for x in offsets)
                        or offsets[0] < 0
                        or offsets[1] <= offsets[0]
                        or 8 + header_len + offsets[1] > size
                    ):
                        return {"ok": False, "reason": "adapter_safetensors_invalid_tensor"}
                    fh.seek(8 + header_len + offsets[0])
                    tensor_nonzero, ok, reason = _scan_tensor_bytes(fh, offsets[1] - offsets[0], dtype)
                    if not ok:
                        return {"ok": False, "reason": reason}
                    has_nonzero = has_nonzero or tensor_nonzero
            if not has_nonzero:
                return {"ok": False, "reason": "adapter_all_zero"}
        except (OSError, struct.error, json.JSONDecodeError, UnicodeDecodeError, TypeError, AttributeError):
            return {"ok": False, "reason": "adapter_safetensors_invalid"}
    return {"ok": True, "reason": "basic_adapter_sanity"}


def _mlx_adapter_rank(adapter_path: str | Path) -> int | None:
    try:
        config_path = Path(adapter_path).parent / "adapter_config.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return int(data["r"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _mlx_rank_mismatch(adapter_path: str | Path, config) -> int | None:
    saved_rank = _mlx_adapter_rank(adapter_path)
    current_rank = int(config.get("lora_rank", 8))
    return saved_rank if saved_rank is not None and saved_rank != current_rank else None


def _write_mlx_adapter_dir(target_dir: Path, model, config, *, write_config: bool = True) -> str:
    _ensure_mlx()
    adapter_file = target_dir / "adapter_model.safetensors"
    lora_weights = {name: value for name, value in nn.utils.tree_flatten(model.trainable_parameters()) if "lora" in name.lower()}
    mx.save_safetensors(str(adapter_file), lora_weights)
    if not write_config:
        return str(adapter_file)
    _rank = config.get("lora_rank", 8)
    target = _lora_target(config)
    keys = _mlx_lora_keys(model, target)
    target_modules = "all-linear" if keys is None else sorted(keys)
    _write_private_text(target_dir / "adapter_config.json", json.dumps({
        "r": _rank,
        "lora_alpha": config.get("lora_alpha", _rank),
        "base_model_name_or_path": config["model"],
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "target_modules": target_modules,
        "base_model": config["model"],
    }))
    return str(adapter_file)


def _save_partial_mlx(model, config, run_id: str, optimizer=None) -> str:
    adapter = _write_mlx_adapter_dir(_resume_checkpoint_dir(config, run_id), model, config, write_config=False)
    _verify_saved_adapter(config, adapter)
    if optimizer is not None:
        _save_mlx_optimizer(optimizer, adapter)
    return adapter


def _make_loss_fn_torch(model, tokenized, ref_logprobs, cfg, ema_mean, torch, diag_sink=None, behavior_logprobs=None):
    tc_lo, tc_hi = cfg["token_clip"]
    kl_c = cfg["kl_coeff"]
    loss_name = str(cfg.get("loss_fn", "mis-po")).lower()
    traj_lo, traj_hi = cfg["traj_clip"]
    device = next(model.parameters()).device

    def loss_fn(idx: int):
        item = _torch_move_item(tokenized[idx], device)
        ids, start, rating = item["input_ids"], item["response_start"], item["rating"]
        ref = ref_logprobs[idx]
        logits = model(**_torch_forward_kwargs(model, item, torch)).logits.squeeze(0)
        new_lp = torch.log_softmax(logits, dim=-1)[:-1].gather(-1, ids[1:].unsqueeze(-1)).squeeze(-1)
        new_r, ref_r = new_lp[start - 1:], ref[start - 1:]
        if new_r.numel() == 0 or ref_r.numel() == 0:
            return logits.sum() * 0.0
        adv_scalar = item.get("advantage", _scalar_advantage(rating, ema_mean, cfg))
        adv = torch.tensor(adv_scalar, device=ids.device, dtype=new_r.dtype)

        if diag_sink is not None:
            with torch.no_grad():
                delta = new_r - ref_r
                ratio_for_diag = torch.exp(delta)
                traj_val = float(torch.exp(delta.mean()).item())
                traj_pass = 1.0 if traj_lo <= traj_val <= traj_hi else 0.0
                inside = ((ratio_for_diag >= tc_lo) & (ratio_for_diag <= tc_hi)).to(new_r.dtype)
                try:
                    p5 = float(torch.quantile(ratio_for_diag.float(), 0.05).item())
                    p95 = float(torch.quantile(ratio_for_diag.float(), 0.95).item())
                except Exception:
                    p5, p95 = float(ratio_for_diag.min().item()), float(ratio_for_diag.max().item())
                diag_sink.append({
                    "policy_kl": float(delta.mean().item()),
                    "ratio_mean": float(ratio_for_diag.mean().item()),
                    "ratio_p5": p5,
                    "ratio_p95": p95,
                    "advantage_mean": adv_scalar,
                    "advantage_std": 0.0,
                    "pct_zero_adv": 1.0 if adv_scalar == 0.0 else 0.0,
                    "traj_gate_pass_rate": traj_pass,
                    "token_mask_rate": float(inside.mean().item()),
                })

        if loss_name == "reinforce++":
            per_token_kl = new_r - ref_r
            adjusted = (adv - kl_c * per_token_kl).detach()
            return -(new_r * adjusted).mean()
        behavior_r = _behavior_logprobs(item, int(new_r.shape[0]), "torch", torch_module=torch) if loss_name == "mis-po" else ref_r
        if behavior_r is None:
            behavior_r = behavior_logprobs[idx][start - 1:] if behavior_logprobs is not None else ref_r

        ratio = torch.exp(new_r - behavior_r)
        clipped_ratio = ratio.clamp(tc_lo, tc_hi)
        surr_1 = ratio * adv
        surr_2 = clipped_ratio * adv
        actor = -torch.minimum(surr_1, surr_2).mean()
        kl = torch.exp(ref_r - new_r) - (ref_r - new_r) - 1
        traj = torch.exp((new_r - ref_r).mean())
        return actor * _trajectory_scale_torch(traj, cfg, torch) + kl_c * kl.mean()

    return loss_fn


def _attempt_train_mlx(config, conn, backend, hardware, attempt: int):
    # mlx_lm loads local converted weights; it does not execute HF repo Python via trust_remote_code.
    _ensure_mlx()
    reserve_bytes = _os_reserve_bytes(True)
    guard = AdaptiveMemoryGuard(backend, _preload_limit_bytes(hardware, reserve_bytes))
    latest = db.latest_adapter(conn, model=config.get("model"))
    resume = _resume_state(conn, config)
    guard.check("before_model_load")
    try:
        model = tokenizer = tokenized = ref_lps = saved_lora = None
        restore_checkpoint = lambda: None
        model, tokenizer = mlx_load(resolve_hf_model_source(config["model"]))
        model = _apply_lora(model, rank=config.get("lora_rank", 8), target=_lora_target(config))
        parent_path = latest["path"] if latest and Path(latest["path"]).exists() else None
        checkpoint_path = resume.get("checkpoint_path") if resume and resume.get("checkpoint_path") else None
        if parent_path:
            parent_path = Path(parent_path)
            if not parent_path.is_file() or parent_path.suffix != ".safetensors":
                return _skip("invalid_mlx_adapter_path", backend="mlx")
            mismatched_rank = _mlx_rank_mismatch(parent_path, config)
            if mismatched_rank is not None:
                current_rank = int(config.get("lora_rank", 8))
                _log_event("skip", backend="mlx", reason="adapter_rank_mismatch", saved_rank=mismatched_rank, config_rank=current_rank)
                return _skip("adapter_rank_mismatch", backend="mlx", saved_rank=mismatched_rank, config_rank=current_rank)
            model.load_weights(str(parent_path), strict=False)
        parent_v = resume.get("parent_version") if resume else (latest["version"] if parent_path else None)
        model.eval()

        model_bytes = max(backend.current_memory_bytes(), backend.active_memory_bytes())
        hardware = _settle_backend_hardware(backend, backend.hardware())
        plan = _plan_strategy(config, hardware, model_bytes)
        for _ in range(attempt):
            plan = _degrade_plan(plan)
        if plan is None:
            _log_event("skip", backend="mlx", reason="insufficient_budget", model_gb=round(model_bytes / 1e9, 3))
            return _skip("insufficient_budget", backend="mlx", model_gb=round(model_bytes / 1e9, 3))

        guard = AdaptiveMemoryGuard(backend, plan.memory_limit_bytes)
        guard.log_step("post_load")

        try:
            batch, fresh_ids, fresh = _load_batch(conn, config, plan, resume)
        except ResumeInvalidated as exc:
            return _skip("resume_invalidated", backend="mlx", detail=str(exc))
        if not batch:
            return _skip("empty_batch", backend="mlx")

        cfg = _tighten_small_batch_cfg(_build_train_cfg(config, plan), len(batch))
        ema_mean, ema_count = db.get_ema(conn, model=config.get("model"))
        pairs = [
            (item, _tokenize_mlx(tokenizer, {**item, "max_seq_len": config.get("max_seq_len", 2048)}))
            for item in batch
        ]
        pairs, skip = _filter_tokenized_pairs(conn, batch, pairs, "mlx")
        if skip:
            return skip
        batch = [item for item, _ in pairs]
        keep_ids = {item["id"] for item in batch}
        fresh = [item for item in fresh if item["id"] in keep_ids]
        fresh_ids = [item_id for item_id in fresh_ids if item_id in keep_ids]
        cfg = _tighten_small_batch_cfg(cfg, len(batch))
        tokenized = _attach_advantages([tok for _, tok in pairs], ema_mean, cfg)
        behavior_lps = None
        if str(cfg.get("loss_fn", "mis-po")).lower() == "mis-po" and parent_path:
            behavior_lps = [mx.stop_gradient(_compute_logprobs_mlx(model, item["input_ids"])) for item in tokenized]
            mx.eval(*behavior_lps)
        saved_lora = _disable_lora(model)
        try:
            ref_lps = [mx.stop_gradient(_compute_logprobs_mlx(model, item["input_ids"])) for item in tokenized]
            mx.eval(*ref_lps)
        finally:
            _enable_lora(model, saved_lora)
        if behavior_lps is None:
            behavior_lps = ref_lps
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.is_file() or checkpoint_path.suffix != ".safetensors":
                return _skip("invalid_mlx_adapter_path", backend="mlx")
            model.load_weights(str(checkpoint_path), strict=False)
        restore_checkpoint = _enable_grad_checkpoint_mlx(model) if plan.aggressive_checkpointing else (lambda: None)
        model.train()

        try:
            total_target_steps = int(resume["total_steps"]) if resume else int(cfg["steps"])
            remaining_steps = int(resume["remaining_steps"]) if resume else total_target_steps
            if remaining_steps <= 0:
                return _skip("nothing_to_train", backend="mlx")
            cfg["steps"] = _slice_steps(config, remaining_steps)
            diag_sink: list[dict] = []
            loss_fn = _make_loss_fn_mlx(model, tokenized, ref_lps, cfg, ema_mean, diag_sink=diag_sink, behavior_logprobs=behavior_lps)
            vg = nn.value_and_grad(model, loss_fn)
            opt = optim.Adam(learning_rate=cfg["lr"])
            if resume and resume.get("checkpoint_path"):
                _load_mlx_optimizer(opt, resume["checkpoint_path"])
            backend.reset_peak_memory()

            total_loss = 0.0
            total_steps = 0
            micro_indices = _micro_indices(len(tokenized), cfg["steps"], cfg["grad_accum"])
            for step in range(cfg["steps"]):
                for retry in range(_pressure_retry_limit(config) + 1):
                    try:
                        current = backend.hardware()
                        block = _background_block_reason(config, backend, current)
                        if block:
                            if block in _TRANSIENT_RESOURCE_BLOCKS and retry < _pressure_retry_limit(config):
                                _pressure_cooldown(backend, config, step, retry, reason=block)
                                continue
                            _log_event("skip", backend="mlx", reason=block, step=step)
                            return _skip(block, backend="mlx", step=step)
                        if not _has_minimum_headroom(current):
                            raise MemoryError("memory pressure while training")
                        guard.check(f"before_step_{step}")
                        acc_grads = None
                        step_loss = 0.0
                        diag_sink.clear()
                        for micro in range(cfg["grad_accum"]):
                            idx = micro_indices[step * cfg["grad_accum"] + micro]
                            loss, grads = vg(idx)
                            mx.eval(loss)
                            step_loss += _require_finite(loss.item())
                            acc_grads = grads if acc_grads is None else tree_map(lambda a, b: a + b, acc_grads, grads)

                        if acc_grads is None:
                            return _skip("no_trainable_gradients", backend="mlx")
                        flat = [grad for _, grad in nn.utils.tree_flatten(acc_grads)]
                        if not flat:
                            return _skip("no_trainable_gradients", backend="mlx")
                        inv_accum = 1.0 / cfg["grad_accum"]
                        norm = mx.sqrt(sum(mx.sum(grad * grad) for grad in flat)) * inv_accum
                        scale = mx.minimum(mx.array(cfg["grad_clip"]) / (norm + 1e-6), mx.array(1.0)) * inv_accum
                        acc_grads = tree_map(lambda grad: grad * scale, acc_grads)
                        grad_norm = _require_finite(norm.item(), "nonfinite_gradient")

                        opt.update(model, acc_grads)
                        mx.eval(model.parameters(), opt.state)
                        total_loss += step_loss / cfg["grad_accum"]
                        total_steps += 1
                        diag_fields = _aggregate_diag(diag_sink)
                        _log_event(
                            "opt_step",
                            backend="mlx",
                            step=step,
                            loss=round(step_loss / cfg["grad_accum"], 6),
                            grad_norm=round(grad_norm, 6),
                            **diag_fields,
                        )
                        backend.clear_cache()
                        guard.log_step("step", step)
                        guard.check(f"after_step_{step}")
                        break
                    except MemoryError as exc:
                        if retry >= _pressure_retry_limit(config):
                            raise
                        _pressure_cooldown(backend, config, step, retry, reason="memory_pressure", detail=str(exc))
        finally:
            restore_checkpoint()

        remaining_steps = max(0, remaining_steps - total_steps)
        if remaining_steps > 0:
            if total_steps == 0:
                return _skip("no_steps_completed", backend="mlx")
            run_id = resume["run_id"] if resume else f"{int(time.time())}-{os.getpid()}-{secrets.token_hex(4)}"
            checkpoint_path = _save_partial_mlx(model, config, run_id, opt)
            db.save_training_state(conn, _training_state_payload(
                config, cfg, backend.name, batch, fresh_ids, resume, latest if parent_path else None,
                run_id, checkpoint_path, remaining_steps, total_target_steps,
            ))
            return _paused_result("mlx", cfg, batch, total_steps, remaining_steps, config, checkpoint_path, parent_v)

        ema_mean, ema_count = _update_ema_from_fresh(ema_mean, ema_count, fresh, cfg["ema_decay"])
        new_v, temp_dir, save_dir = _stage_adapter_dir(_next_adapter_version(conn), config)
        try:
            _write_mlx_adapter_dir(temp_dir, model, config)
            _commit_adapter_dir(temp_dir, save_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        return _finalize_training(
            conn, config, backend, cfg, batch, fresh_ids,
            str(save_dir / "adapter_model.safetensors"),
            ema_mean, ema_count, total_loss, total_steps, new_v, parent_v,
            resume.get("checkpoint_path") if resume else None,
        )
    finally:
        model = tokenizer = tokenized = ref_lps = saved_lora = None
        _release_backend_memory(backend)


def _attempt_train_cuda(config, conn, backend, hardware, attempt: int):
    torch, AutoModelForCausalLM, AutoTokenizer, LoraConfig, PeftModel, get_peft_model = _torch_stack()
    reserve_bytes = _os_reserve_bytes(False)
    guard = AdaptiveMemoryGuard(backend, _preload_limit_bytes(hardware, reserve_bytes))
    latest = db.latest_adapter(conn, model=config.get("model"))
    resume = _resume_state(conn, config)

    guard.check("before_model_load")
    try:
        model = base = tokenizer = tokenized = ref_lps = None
        _warn_remote_code(config)
        _guard_remote_code(config)
        model_source = resolve_hf_model_source(config["model"], trust_remote_code=_trust_remote_code(config))
        local_only = model_source != config["model"]
        tokenizer_source = _prepare_tokenizer_source(model_source, config["model"])
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            **_hf_tokenizer_kwargs(config, local_files_only=local_only),
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=_trust_remote_code(config),
            torch_dtype=backend.preferred_dtype(),
            low_cpu_mem_usage=True,
            local_files_only=local_only,
        ).to(backend.device)
        if hasattr(base, "config") and hasattr(base.config, "use_cache"):
            base.config.use_cache = False
        target_modules = _torch_target_modules(torch, base, _lora_target(config))
        _rank = config.get("lora_rank", 8)
        model = get_peft_model(
            base,
            LoraConfig(
                r=_rank,
                lora_alpha=config.get("lora_alpha", _rank),
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
                **_lora_init_kwargs(config),
            ),
        )
        _maybe_apply_liger(model, config)
        parent_path = latest["path"] if latest and Path(latest["path"]).exists() else None
        checkpoint_path = resume.get("checkpoint_path") if resume and resume.get("checkpoint_path") and Path(resume["checkpoint_path"]).exists() else None
        if parent_path:
            _load_torch_adapter_weights(torch, model, parent_path)
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model_bytes = max(backend.current_memory_bytes(), backend.active_memory_bytes())
        plan = _plan_strategy(config, hardware, model_bytes)
        for _ in range(attempt):
            plan = _degrade_plan(plan)
        if plan is None:
            _log_event("skip", backend="cuda", reason="insufficient_budget", model_gb=round(model_bytes / 1e9, 3))
            return _skip("insufficient_budget", backend="cuda", model_gb=round(model_bytes / 1e9, 3))

        guard = AdaptiveMemoryGuard(backend, plan.memory_limit_bytes)
        if plan.aggressive_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
            except Exception as exc:
                _log_event("gradient_checkpointing_skipped", backend="cuda", detail=_detail(exc))

        try:
            batch, fresh_ids, fresh = _load_batch(conn, config, plan, resume)
        except ResumeInvalidated as exc:
            return _skip("resume_invalidated", backend="cuda", detail=str(exc))
        if not batch:
            return _skip("empty_batch", backend="cuda")

        cfg = _tighten_small_batch_cfg(_build_train_cfg(config, plan), len(batch))
        ema_mean, ema_count = db.get_ema(conn, model=config.get("model"))
        pairs = [
            (item, _tokenize_torch(tokenizer, {**item, "max_seq_len": config.get("max_seq_len", 2048)}, backend.device))
            for item in batch
        ]
        pairs, skip = _filter_tokenized_pairs(conn, batch, pairs, "cuda")
        if skip:
            return skip
        batch = [item for item, _ in pairs]
        keep_ids = {item["id"] for item in batch}
        fresh = [item for item in fresh if item["id"] in keep_ids]
        fresh_ids = [item_id for item_id in fresh_ids if item_id in keep_ids]
        cfg = _tighten_small_batch_cfg(cfg, len(batch))
        tokenized = _attach_advantages([tok for _, tok in pairs], ema_mean, cfg)
        # Reference logprobs: eval mode (no dropout) with LoRA disabled, so the
        # reference is the base-model distribution and is deterministic.
        model.eval()
        behavior_lps = [_compute_logprobs_torch(model, item, torch) for item in tokenized] if str(cfg.get("loss_fn", "mis-po")).lower() == "mis-po" and parent_path else None
        with _torch_adapter_disabled(model, torch):
            ref_lps = [_compute_logprobs_torch(model, item, torch) for item in tokenized]
        if behavior_lps is None:
            behavior_lps = ref_lps
        if checkpoint_path:
            _load_torch_adapter_weights(torch, model, checkpoint_path)
        model.train()

        save_model = model
        train_model = _maybe_torch_compile(model, config)
        total_target_steps = int(resume["total_steps"]) if resume else int(cfg["steps"])
        remaining_steps = int(resume["remaining_steps"]) if resume else total_target_steps
        if remaining_steps <= 0:
            return _skip("nothing_to_train", backend="cuda")
        cfg["steps"] = _slice_steps(config, remaining_steps)
        diag_sink: list[dict] = []
        loss_fn = _make_loss_fn_torch(train_model, tokenized, ref_lps, cfg, ema_mean, torch, diag_sink=diag_sink, behavior_logprobs=behavior_lps)
        optimizer = _build_optimizer_torch(torch, save_model, cfg, config)
        if resume and resume.get("checkpoint_path"):
            _load_torch_optimizer(torch, optimizer, resume["checkpoint_path"])
        backend.reset_peak_memory()

        total_loss = 0.0
        total_steps = 0
        micro_indices = _micro_indices(len(tokenized), cfg["steps"], cfg["grad_accum"])
        for step in range(cfg["steps"]):
            for retry in range(_pressure_retry_limit(config) + 1):
                try:
                    current = backend.hardware()
                    block = _background_block_reason(config, backend, current)
                    if block:
                        if block in _TRANSIENT_RESOURCE_BLOCKS and retry < _pressure_retry_limit(config):
                            _pressure_cooldown(backend, config, step, retry, reason=block)
                            continue
                        _log_event("skip", backend="cuda", reason=block, step=step)
                        return _skip(block, backend="cuda", step=step)
                    if not _has_minimum_headroom(current):
                        raise MemoryError("memory pressure while training")
                    guard.check(f"before_step_{step}")
                    optimizer.zero_grad(set_to_none=True)
                    step_loss = 0.0
                    diag_sink.clear()
                    for micro in range(cfg["grad_accum"]):
                        idx = micro_indices[step * cfg["grad_accum"] + micro]
                        loss = loss_fn(idx) / cfg["grad_accum"]
                        loss_value = _require_finite(loss.detach().item())
                        loss.backward()
                        step_loss += loss_value
                    trainable_params = [p for p in save_model.parameters() if p.requires_grad and p.grad is not None]
                    grad_norm = _require_finite(
                        torch.nn.utils.clip_grad_norm_(trainable_params, cfg["grad_clip"]).detach().item()
                        if trainable_params else 0.0,
                        "nonfinite_gradient",
                    )
                    optimizer.step()
                    backend.synchronize()
                    backend.clear_cache()
                    diag_fields = _aggregate_diag(diag_sink)
                    _log_event(
                        "opt_step",
                        backend="cuda",
                        step=step,
                        loss=round(step_loss, 6),
                        grad_norm=round(grad_norm, 6),
                        **diag_fields,
                    )
                    guard.log_step("step", step)
                    guard.check(f"after_step_{step}")
                    total_loss += step_loss
                    total_steps += 1
                    break
                except MemoryError as exc:
                    if retry >= _pressure_retry_limit(config):
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    _pressure_cooldown(backend, config, step, retry, reason="memory_pressure", detail=str(exc))

        remaining_steps = max(0, remaining_steps - total_steps)
        if remaining_steps > 0:
            if total_steps == 0:
                return _skip("no_steps_completed", backend="cuda")
            run_id = resume["run_id"] if resume else f"{int(time.time())}-{os.getpid()}-{secrets.token_hex(4)}"
            checkpoint_path = _save_partial_cuda(save_model, config, run_id, torch, optimizer)
            parent_v = resume.get("parent_version") if resume else (latest["version"] if parent_path else None)
            db.save_training_state(conn, _training_state_payload(
                config, cfg, backend.name, batch, fresh_ids, resume, latest if parent_path else None,
                run_id, checkpoint_path, remaining_steps, total_target_steps,
            ))
            return _paused_result("cuda", cfg, batch, total_steps, remaining_steps, config, checkpoint_path, parent_v)

        ema_mean, ema_count = _update_ema_from_fresh(ema_mean, ema_count, fresh, cfg["ema_decay"])
        parent_v = resume.get("parent_version") if resume else (latest["version"] if parent_path else None)
        new_v, temp_dir, save_dir = _stage_adapter_dir(_next_adapter_version(conn), config)
        try:
            save_model.save_pretrained(temp_dir, safe_serialization=True)
            _commit_adapter_dir(temp_dir, save_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        adapter_file = next((str(path) for path in save_dir.glob("*.safetensors")), None)
        if not adapter_file:
            raise RuntimeError("adapter_save_missing_safetensors")
        return _finalize_training(
            conn, config, backend, cfg, batch, fresh_ids, adapter_file,
            ema_mean, ema_count, total_loss, total_steps, new_v, parent_v,
            resume.get("checkpoint_path") if resume else None,
        )
    finally:
        model = base = tokenizer = tokenized = ref_lps = None
        _release_backend_memory(backend)


def _attempt_train(config, conn, backend, hardware, attempt: int):
    if backend.name == "mlx":
        return _attempt_train_mlx(config, conn, backend, hardware, attempt)
    return _attempt_train_cuda(config, conn, backend, hardware, attempt)


def _is_retryable_memory_error(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return any(
        token in name or token in msg
        for token in (
            "outofmemory",
            "cuda out of memory",
            "memory pressure",
            "out of memory",
            "insufficient memory",
            "unable to allocate",
            "working set",
        )
    )


def _interrupt_training(signum, _frame):
    raise KeyboardInterrupt(f"training interrupted by signal {signum}")


def train_result(config, conn):
    lock_fd = None
    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            old_handlers[sig] = signal.signal(sig, _interrupt_training)
        except (ValueError, OSError):
            pass
    internal_retry = (
        config.get("_skip_lock_once")
        and config.get("_fresh_process_retry_done")
        and config.get("_retry_token")
        and os.environ.get("REINFORCECLAW_INTERNAL_RETRY") == config.get("_retry_token")
    )
    if not internal_retry:
        lock_fd = _acquire_lock()
        if lock_fd is None:
            _log_event("skip", reason="train_lock_held")
            return _skip("train_lock_held")

    caffeinate = _keep_awake() if config.get("_background") else None
    try:
        if config.get("low_priority", True):
            _set_low_priority()

        resume = _resume_state(conn, config)
        trainable = db.count_trainable_untrained(conn, source=_feedback_source(config), model=config.get("model"))
        if not resume and trainable < config.get("batch_min", 32):
            return _skip("below_threshold", trainable=trainable, batch_min=config.get("batch_min", 32))

        compat = model_compatibility(config)
        if not compat["ok"]:
            _log_event("skip", reason=compat["reason"], detail=compat.get("detail"), backend=compat.get("backend"))
            return _skip(compat["reason"], detail=compat.get("detail"), backend=compat.get("backend"))

        try:
            backend = _select_backend(config)
        except Exception as exc:
            _log_event("skip", reason="backend_unavailable", error=type(exc).__name__, detail=str(exc))
            return _skip("backend_unavailable", detail=str(exc))
        hardware = _settle_backend_hardware(backend, backend.hardware())
        if not _has_minimum_headroom(hardware):
            avail_gb, _ = _hardware_gbs(hardware)
            _log_event("skip", backend=backend.name, reason="insufficient_headroom", available_gb=avail_gb)
            return _skip("insufficient_headroom", backend=backend.name, available_gb=avail_gb)
        block = _background_block_reason(config, backend, hardware)
        if block:
            avail_gb, host_gb = _hardware_gbs(hardware)
            _log_event("skip", backend=backend.name, reason=block, available_gb=avail_gb, host_available_gb=host_gb)
            return _skip(block, backend=backend.name, available_gb=avail_gb, host_available_gb=host_gb)
        avail_gb, host_gb = _hardware_gbs(hardware)
        _log_event(
            "train_start",
            backend=backend.name,
            device=getattr(hardware, "device_name", backend.name),
            total_gb=round(hardware.total_memory_bytes / 1e9, 3),
            available_gb=avail_gb,
            host_available_gb=host_gb,
        )

        max_attempts = max(2, int(config.get("cuda_degrade_attempts", 5)))
        for attempt in range(max_attempts):
            try:
                hardware = hardware if attempt == 0 else _settle_backend_hardware(backend, backend.hardware())
                if not _has_minimum_headroom(hardware):
                    avail_gb, _ = _hardware_gbs(hardware)
                    _log_event("skip", backend=backend.name, attempt=attempt, reason="insufficient_headroom", available_gb=avail_gb)
                    return _skip("insufficient_headroom", backend=backend.name, attempt=attempt)
                block = _background_block_reason(config, backend, hardware)
                if block:
                    avail_gb, host_gb = _hardware_gbs(hardware)
                    _log_event("skip", backend=backend.name, attempt=attempt, reason=block, available_gb=avail_gb, host_available_gb=host_gb)
                    return _skip(block, backend=backend.name, attempt=attempt)
                result = _attempt_train(config, conn, backend, hardware, attempt)
                if (
                    backend.name == "mlx"
                    and result.get("status") == "skipped"
                    and result.get("reason") == "insufficient_budget"
                ):
                    retried = _fresh_process_train_retry(config, conn)
                    if retried is not None:
                        return retried
                if result.get("status") == "trained":
                    _log_event("train_done", attempt=attempt, **{k: v for k, v in result.items() if k != "status"})
                    return result
                if (
                    result.get("status") == "skipped"
                    and result.get("reason") in {"insufficient_budget", "memory_pressure"}
                    and attempt + 1 < max_attempts
                ):
                    _release_backend_memory(backend)
                    continue
                return result
            except Exception as exc:
                _release_backend_memory(backend)
                detail = _scrub_secret(str(exc))
                _log_event("train_error", attempt=attempt, error=type(exc).__name__, detail=detail)
                if isinstance(exc, FloatingPointError):
                    return _skip(detail or "nonfinite_loss", backend=backend.name, attempt=attempt)
                if "missing_cuda_training_dependency:" in str(exc):
                    return _skip("missing_cuda_training_dependency", detail=detail)
                if not _is_retryable_memory_error(exc):
                    return _skip("train_error", detail=detail, backend=backend.name, attempt=attempt)
                if attempt == 0:
                    continue
                return _skip("memory_pressure", detail=detail, backend=backend.name, attempt=attempt)
        return _skip("no_training_result")
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if caffeinate is not None:
            caffeinate.terminate()
            try:
                caffeinate.wait(timeout=1)
            except subprocess.TimeoutExpired:
                caffeinate.kill()
                try:
                    caffeinate.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            CAFFEINATE_PID_PATH.unlink(missing_ok=True)
        _release_lock(lock_fd)


def train(config, conn):
    result = train_result(config, conn)
    if result.get("status") == "trained":
        return {k: v for k, v in result.items() if k != "status"}
    return None


def _adapter_dir(config=None):
    root = Path((config or {}).get("adapter_root", Path.home() / ".reinforceclaw" / "adapters")).expanduser()
    private_root = db.secure_private_dir(Path.home() / ".reinforceclaw").resolve()
    resolved = root.resolve(strict=False)
    if resolved == private_root or private_root not in resolved.parents:
        raise PermissionError(f"adapter_root must be a child of {private_root}")
    return db.secure_private_dir(root)


def _resume_dir(config=None):
    root = _adapter_dir(config) / ".resume"
    return db.secure_private_dir(root)


def _resume_checkpoint_dir(config, run_id: str) -> Path:
    path = _resume_dir(config) / run_id
    return db.secure_private_dir(path)


def _stage_adapter_dir(version: int, config=None) -> tuple[int, Path, Path]:
    root = _adapter_dir(config)
    current = version
    for _ in range(1000):
        final_dir = root / f"v{current}"
        temp_dir = root / f".v{current}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
        if not final_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir.mkdir(parents=True, exist_ok=False)
            db.secure_private_dir(temp_dir)
            return current, temp_dir, final_dir
        current += 1
    raise RuntimeError("adapter_version_space_exhausted")


def _commit_adapter_dir(temp_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise FileExistsError(f"adapter dir already exists: {final_dir}")
    temp_dir.replace(final_dir)


def _convert_to_gguf(adapter_dir):
    import subprocess

    safetensors_file = Path(adapter_dir) / "adapter.safetensors"
    if not safetensors_file.exists():
        safetensors_file = Path(adapter_dir) / "adapter_model.safetensors"
    gguf_file = Path(adapter_dir) / "adapter.gguf"
    if gguf_file.exists():
        return str(gguf_file)
    if not safetensors_file.exists():
        return None
    for cmd in (["convert-lora-to-gguf"], ["python3", "-m", "llama_cpp.convert_lora"]):
        try:
            result = subprocess.run(
                cmd + [str(safetensors_file), "--outfile", str(gguf_file)],
                capture_output=True,
                timeout=120,
                env=_child_env(),
            )
            if result.returncode == 0 and gguf_file.exists():
                return str(gguf_file)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def load_adapter(server_type, adapter_path, model_name, server_url=None):
    import requests
    import hashlib

    def local_server_url(value, default):
        base_url = str(value or default).rstrip("/")
        try:
            host = urlparse(base_url).hostname or ""
            if host == "localhost" or ip_address(host).is_loopback:
                return base_url
        except ValueError:
            pass
        return base_url if os.environ.get("REINFORCECLAW_ALLOW_REMOTE_SERVER") == "1" else ""

    def post_with_retry(url, **kwargs):
        last_exc = last_response = None
        for attempt in range(3):
            try:
                response = requests.post(url, **kwargs)
                if response.status_code == 200:
                    return response
                last_response = response
            except requests.RequestException as exc:
                last_exc = exc
            time.sleep(1)
        if last_response is not None:
            return last_response
        raise last_exc or requests.RequestException("request failed")

    def blob_digest(path):
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"

    def ensure_blob(base_url, path):
        digest = blob_digest(path)
        url = f"{base_url}/api/blobs/{digest}"
        try:
            if requests.head(url, timeout=30).status_code == 200:
                return digest
        except requests.RequestException:
            pass
        with path.open("rb") as fh:
            response = requests.post(url, data=fh, timeout=120)
        return digest if response.status_code in (200, 201) else ""

    adapter_file = Path(adapter_path)
    adapter_dir_path = adapter_file.parent if adapter_file.is_file() else adapter_file
    adapter_dir = str(adapter_dir_path)
    if server_type == "ollama":
        model_name = str(model_name or "").strip()
        if not model_name or not re.fullmatch(r"[A-Za-z0-9_.:-]+(?:/[A-Za-z0-9_.:-]+)*", model_name) or ".." in model_name.split("/"):
            return None
        if any(ch in adapter_dir for ch in "\r\n"):
            return None
        gguf = adapter_dir_path / "adapter.gguf"
        if not gguf.exists() and any(adapter_dir_path.glob("*.safetensors")):
            _convert_to_gguf(adapter_dir)
        adapter_ref = "./adapter.gguf" if gguf.exists() else "." if any(adapter_dir_path.glob("*.safetensors")) else ""
        if not adapter_ref:
            print(f"Ollama adapter not prepared; use the saved adapter manually: {adapter_path}")
            return None
        modelfile = f"FROM {model_name}\nADAPTER {adapter_ref}\n"
        out_name = f"{model_name.split('/')[-1]}-reinforceclaw"
        modelfile_path = Path(adapter_dir) / "Modelfile.reinforceclaw"
        try:
            _write_private_text(modelfile_path, modelfile)
            for attempt in range(3):
                result = subprocess.run(
                    ["ollama", "create", out_name, "-f", modelfile_path.name],
                    capture_output=True, text=True, timeout=120, env=_child_env(), cwd=adapter_dir,
                )
                if result.returncode == 0:
                    return True
                if attempt < 2:
                    time.sleep(1)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            modelfile_path.unlink(missing_ok=True)
        try:
            api_adapter_ref = str(gguf if gguf.exists() else adapter_dir_path)
            if any(ch in api_adapter_ref for ch in "\r\n"):
                return None
            base_url = local_server_url(server_url or os.environ.get("REINFORCECLAW_OLLAMA_URL"), "http://localhost:11434")
            if not base_url:
                return None
            adapter_blob = gguf if gguf.exists() else next(adapter_dir_path.glob("*.safetensors"), None)
            if adapter_blob and adapter_blob.is_file():
                digest = ensure_blob(base_url, adapter_blob)
                if digest:
                    response = post_with_retry(
                        f"{base_url}/api/create",
                        json={"model": out_name, "from": model_name, "adapters": {adapter_blob.name: digest}, "stream": False},
                        timeout=60,
                    )
                    return response.status_code == 200
            response = post_with_retry(
                f"{base_url}/api/create",
                json={"model": out_name, "modelfile": f"FROM {model_name}\nADAPTER {api_adapter_ref}\n", "stream": False},
                timeout=60,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    if server_type == "lmstudio":
        print(f"Load adapter in LM Studio manually if your LM Studio build supports LoRA: {adapter_dir}")
        return None
    if server_type == "vllm":
        base_url = local_server_url(server_url or os.environ.get("REINFORCECLAW_VLLM_URL"), "http://localhost:8000")
        if not base_url:
            return None
        try:
            post_with_retry(
                f"{base_url}/v1/unload_lora_adapter",
                json={"lora_name": "reinforceclaw"},
                timeout=30,
            )
            response = post_with_retry(
                f"{base_url}/v1/load_lora_adapter",
                json={"lora_name": "reinforceclaw", "lora_path": adapter_dir, "load_inplace": True},
                timeout=30,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    print(f"Restart your server with adapter: {adapter_path}")
    return None
