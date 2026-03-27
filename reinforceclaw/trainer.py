"""Adaptive LoRA trainer with MLX and CUDA backends."""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import shutil
import sys
import time
import fcntl
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from . import db
from .backend_cuda import CUDABackend
from .backend_mlx import MLXBackend, mlx_drain

GB = 1024 ** 3
MIN_TRAINING_BUDGET = 2 * GB
MAC_OS_RESERVE = 8 * GB
LINUX_OS_RESERVE = 4 * GB
TRAIN_LOG_PATH = Path.home() / ".reinforceclaw" / "train.log"
TRAIN_LOCK_PATH = Path.home() / ".reinforceclaw" / "train.lock"
BACKGROUND_IDLE_LOAD = 0.85
BACKGROUND_WINDOW_MINUTES = 180

# lazy MLX imports
mx = nn = optim = mlx_load = linear_to_lora_layers = tree_map = None


def _ensure_mlx():
    global mx, nn, optim, mlx_load, linear_to_lora_layers, tree_map
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
    tree_map = mlx.utils.tree_map


def _log_event(event: str, **fields) -> None:
    TRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
    with TRAIN_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_lock() -> int | None:
    TRAIN_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(TRAIN_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode("utf-8"))
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
    gpu_ok = available is None or available >= MIN_TRAINING_BUDGET
    host_ok = host_available is None or host_available - reserve >= MIN_TRAINING_BUDGET
    return gpu_ok and host_ok


def _settle_backend_hardware(backend, hardware=None, rounds: int = 3, delay: float = 1.0):
    hardware = hardware or backend.hardware()
    if backend.name != "mlx":
        return hardware
    best = hardware
    best_available = hardware.available_memory_bytes or 0
    stable = 0
    for _ in range(max(1, rounds)):
        backend.clear_all()
        backend.synchronize()
        time.sleep(delay)
        current = backend.hardware()
        current_available = current.available_memory_bytes or 0
        if current_available > best_available:
            best = current
            best_available = current_available
            stable = 0
            continue
        stable += 1
        if stable >= 2:
            break
    return best


def _scheduled_window_open(config) -> bool:
    schedule = config.get("train_schedule", "03:00")
    if schedule in ("manual", "auto") or not config.get("_background"):
        return True
    try:
        hour, minute = (int(part) for part in schedule.split(":", 1))
    except ValueError:
        return False
    now = datetime.now().astimezone()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now < target:
        return False
    return now < target + timedelta(minutes=float(config.get("schedule_window_minutes", BACKGROUND_WINDOW_MINUTES)))


def _is_scheduled_background(config) -> bool:
    return bool(config.get("_background")) and config.get("train_schedule", "03:00") not in ("manual", "auto")


def _cuda_activity(backend) -> dict | None:
    device_index = getattr(getattr(backend, "device", None), "index", 0) or 0
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_util": int(util.gpu),
            "mem_used": int(mem.used),
            "mem_total": int(mem.total),
            "source": "nvml",
        }
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
            text=True,
            timeout=2,
        ).strip()
        if not out:
            return None
        util, used, total = (int(part.strip()) for part in out.split(",", 2))
        mib = 1024 * 1024
        return {"gpu_util": util, "mem_used": used * mib, "mem_total": total * mib, "source": "nvidia-smi"}
    except Exception:
        pass
    try:
        hardware = backend.hardware()
        if hardware.available_memory_bytes is None:
            return None
        used = max(0, hardware.total_memory_bytes - hardware.available_memory_bytes)
        return {"gpu_util": 0, "mem_used": used, "mem_total": hardware.total_memory_bytes, "source": "fallback"}
    except Exception:
        return None


def _background_block_reason(config, backend, hardware) -> str | None:
    if not config.get("_background"):
        return None
    if not _scheduled_window_open(config):
        return "outside_schedule_window"
    nightly = _is_scheduled_background(config)
    if not nightly and _load_ratio() > float(config.get("idle_load_threshold", BACKGROUND_IDLE_LOAD)):
        return "high_cpu_load"
    reserve = _os_reserve_bytes(hardware.unified_memory)
    if hardware.unified_memory:
        available = hardware.available_memory_bytes
        if available is not None and available < reserve + int(1.5 * MIN_TRAINING_BUDGET):
            return "memory_busy"
        return None
    activity = _cuda_activity(backend)
    if activity is None and not nightly:
        return "missing_cuda_idle_telemetry"
    if activity is not None and not nightly and activity.get("source") != "fallback":
        if activity["gpu_util"] > int(config.get("cuda_idle_gpu_util", 80)):
            return "gpu_busy"
        mem_threshold = float(
            config.get(
                "cuda_idle_mem_fraction_fallback" if activity.get("source") == "fallback" else "cuda_idle_mem_fraction",
                0.60 if activity.get("source") == "fallback" else 0.80,
            )
        )
        if activity["mem_total"] and activity["mem_used"] / activity["mem_total"] > mem_threshold:
            return "gpu_memory_busy"
    host_available = getattr(hardware, "system_available_memory_bytes", None)
    gpu_available = hardware.available_memory_bytes
    host_busy = host_available is not None and host_available < reserve + MIN_TRAINING_BUDGET
    gpu_busy = gpu_available is not None and gpu_available < max(MIN_TRAINING_BUDGET, int(hardware.total_memory_bytes * 0.20))
    if host_busy:
        return "host_memory_busy"
    if gpu_busy:
        return "low_free_vram"
    return None


def _background_should_wait(config, backend, hardware) -> bool:
    return _background_block_reason(config, backend, hardware) is not None


def smoke_status(config, conn):
    cfg = {**config, "_background": True}
    trainable = db.count_trainable_untrained(conn)
    batch_min = cfg.get("batch_min", 24)
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
    return {
        "would_train": block is None,
        "reason": block or "ready",
        "backend": backend.name,
        "trainable": trainable,
        "batch_min": batch_min,
        "schedule": cfg.get("train_schedule", "03:00"),
        "available_gb": round((hardware.available_memory_bytes or 0) / 1e9, 3),
        "host_available_gb": round((getattr(hardware, "system_available_memory_bytes", 0) or 0) / 1e9, 3),
    }


def model_compatibility(config):
    try:
        backend = _select_backend(config)
    except Exception as exc:
        return {"ok": False, "reason": "backend_unavailable", "detail": str(exc)}

    model = str(config.get("model", ""))
    lowered = model.lower()
    if "gguf" in lowered:
        return {"ok": False, "reason": "gguf_models_not_trainable", "backend": backend.name, "detail": model}
    if backend.name == "cuda":
        if model.startswith("mlx-community/"):
            return {"ok": False, "reason": "mlx_model_on_cuda_backend", "backend": backend.name, "detail": model}
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
    _log_event(
        "pressure_retry",
        backend=backend.name,
        step=step,
        retry=retry,
        reason=reason,
        detail=detail,
    )
    backend.clear_cache()
    backend.synchronize()
    time.sleep(_pressure_cooldown_seconds(config))


def _skip(reason: str, **fields) -> dict:
    return {"status": "skipped", "reason": reason, **fields}


def _conn_db_path(conn) -> str | None:
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
    except Exception:
        return None
    for row in rows:
        name = row[1] if not isinstance(row, dict) else row.get("name")
        file = row[2] if not isinstance(row, dict) else row.get("file")
        if name == "main" and file:
            return str(file)
    return None


def _fresh_process_train_retry(config, conn) -> dict | None:
    if config.get("_fresh_process_retry_done"):
        return None
    db_path = _conn_db_path(conn)
    if not db_path:
        return None
    _free_mlx()
    time.sleep(3)
    cfg = {**config, "_fresh_process_retry_done": True, "_skip_lock_once": True}
    cmd = [
        sys.executable,
        "-c",
        (
            "import json, sys; "
            "from pathlib import Path; "
            "from reinforceclaw import db, trainer; "
            "cfg = json.loads(sys.argv[1]); "
            "conn = db.connect(Path(sys.argv[2])); "
            "print(json.dumps(trainer.train_result(cfg, conn)))"
        ),
        json.dumps(cfg),
        db_path,
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    if proc.returncode != 0:
        _log_event("fresh_process_retry_failed", returncode=proc.returncode, stderr=proc.stderr[-500:])
        return None
    lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        return None
    result = json.loads(lines[-1])
    result.setdefault("retry_mode", "fresh_process")
    return result


def _resume_state(conn, config):
    state = db.get_training_state(conn)
    if not state:
        return None
    if state.get("model") != config.get("model"):
        return None
    if state.get("loss_fn") != str(config.get("loss_fn", "mis-po")).lower():
        return None
    path = state.get("checkpoint_path")
    if path and not Path(path).exists():
        db.clear_training_state(conn)
        return None
    return state


def _build_train_cfg(config, plan: TrainingPlan):
    defaults = {
        "loss_fn": "mis-po",
        "steps": plan.steps,
        "lr": 8e-6,
        "token_clip": [0.5, 2.0],
        "traj_clip": [0.996, 1.001],
        "kl_coeff": 0.08,
        "traj_penalty": 0.05,
        "grad_accum": plan.grad_accum,
        "grad_clip": 1.0,
        "ema_decay": 0.99,
        "pos_weight": 1.0,
        "adv_clip": 2.0,
        "max_passes": 1.0,
        "adapter_keep": 20,
    }
    return {key: config.get(key, value) for key, value in defaults.items()}


def _tighten_small_batch_cfg(cfg: dict, batch_size: int) -> dict:
    if batch_size <= 2:
        cfg = {**cfg, "grad_accum": 1}
    elif batch_size <= 4:
        cfg = {**cfg, "grad_accum": min(cfg["grad_accum"], 2)}
    elif batch_size <= 8:
        cfg = dict(cfg)
    return cfg


def _trajectory_scale_mlx(delta_mean, cfg):
    low, high = cfg["traj_clip"]
    low_v = mx.maximum(mx.array(low), mx.array(1e-6))
    high_v = mx.maximum(mx.array(high), low_v)
    violation = mx.maximum(low_v - delta_mean, mx.array(0.0)) + mx.maximum(delta_mean - high_v, mx.array(0.0))
    return 1.0 / (1.0 + float(cfg.get("traj_penalty", 0.05)) * violation * violation)


def _trajectory_scale_torch(delta_mean, cfg, torch):
    low, high = cfg["traj_clip"]
    low_v = torch.clamp(torch.tensor(low, device=delta_mean.device, dtype=delta_mean.dtype), min=1e-6)
    high_v = torch.maximum(torch.tensor(high, device=delta_mean.device, dtype=delta_mean.dtype), low_v)
    zero = torch.tensor(0.0, device=delta_mean.device, dtype=delta_mean.dtype)
    violation = torch.maximum(low_v - delta_mean, zero) + torch.maximum(delta_mean - high_v, zero)
    return 1.0 / (1.0 + float(cfg.get("traj_penalty", 0.05)) * violation * violation)


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

    batch_cap = max(1, int(config.get("batch_size", config.get("batch_min", 24))))
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


def _build_batch(conn, batch_size: int, replay_ratio: float):
    batch_size = max(1, int(batch_size))
    replay_target = min(batch_size, max(0, int(round(batch_size * max(0.0, float(replay_ratio))))))
    fresh_target = max(0, batch_size - replay_target)
    all_fresh = db.get_untrained(conn, limit=batch_size)
    all_replay = db.get_replay(conn, limit=batch_size)
    fresh = all_fresh[:fresh_target] if fresh_target else []
    replay = all_replay[:replay_target] if replay_target else []
    missing = batch_size - (len(fresh) + len(replay))
    if missing > 0:
        fresh.extend(all_fresh[len(fresh):len(fresh) + missing])
    missing = batch_size - (len(fresh) + len(replay))
    if missing > 0:
        replay.extend(all_replay[len(replay):len(replay) + missing])
    return fresh + replay, [item["id"] for item in fresh], fresh


def _release_backend_memory(backend) -> None:
    backend.clear_all()
    gc.collect()
    try:
        backend.synchronize()
    except Exception:
        pass
    if backend.name == "mlx":
        time.sleep(2)


def _lora_target(config) -> str:
    return str(config.get("lora_target", "attention")).lower()


def _strict_target_selection(target: str) -> bool:
    # This is intentionally strict for all non-"all" LoRA targets.
    # It matters most for MoE models because falling back to every linear layer can
    # accidentally LoRA expert/router-adjacent weights, but the same silent fallback
    # is also unsafe on dense models.
    return target != "all"


def _mlx_lora_keys(model, target: str):
    if target == "all":
        return None
    attention = {"q_proj", "k_proj", "v_proj", "o_proj", "wq", "wk", "wv", "wo", "c_attn", "c_proj"}
    mlp = {"gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"}
    allowed = attention if target == "attention" else attention | mlp
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
    cfg = {"rank": rank, "scale": rank, "dropout": 0.0}
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
    saved = {}
    for name, param in nn.utils.tree_flatten(model.parameters()):
        if "lora_b" in name:
            saved[name] = param
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


def _tokenize_mlx(tokenizer, item):
    prompt, response = item["prompt"], item["response"]
    max_seq_len = max(128, int(item.get("max_seq_len") or 2048))
    ctx = None
    if item.get("context"):
        try:
            ctx = json.loads(item["context"])
        except (json.JSONDecodeError, TypeError):
            ctx = None
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = ctx.get("messages") if isinstance(ctx, dict) and isinstance(ctx.get("messages"), list) else None
        if msgs is None:
            msgs = []
            if isinstance(ctx, dict) and "system" in ctx:
                msgs.append({"role": "system", "content": ctx["system"]})
            msgs.append({"role": "user", "content": prompt})
        prompt_ids = _apply_chat_template(tokenizer, msgs, add_generation_prompt=True)
        full_ids = _apply_chat_template(
            tokenizer,
            msgs + [{"role": "assistant", "content": response}],
            add_generation_prompt=False,
        )
    elif isinstance(ctx, dict) and isinstance(ctx.get("messages"), list):
        prompt_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in ctx["messages"])
        prompt_ids = tokenizer.encode(prompt_text)
        full_ids = tokenizer.encode(prompt_text + "\nAssistant: " + response)
    else:
        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(prompt + response)
    offset = max(0, len(full_ids) - max_seq_len)
    full_ids = full_ids[offset:]
    return {
        "input_ids": mx.array(full_ids),
        "response_start": max(1, len(prompt_ids) - offset),
        "rating": item["rating"],
        "id": item["id"],
    }


def _compute_logprobs_mlx(model, input_ids):
    logits = model(input_ids[None, :]).squeeze(0)
    lp = nn.log_softmax(logits, axis=-1)
    tok_lp = mx.take_along_axis(lp[:-1], input_ids[1:, None], axis=-1).squeeze(-1)
    mx.eval(tok_lp)
    return tok_lp


def _apply_chat_template(tokenizer, messages, *, add_generation_prompt=False, tokenize=True):
    kwargs = {"add_generation_prompt": add_generation_prompt, "tokenize": tokenize}
    if add_generation_prompt:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _context_dict(item):
    if not item.get("context"):
        return None
    try:
        return json.loads(item["context"])
    except (json.JSONDecodeError, TypeError):
        return None


def _behavior_logprobs(item, length: int, xp_name: str, torch_module=None):
    ctx = _context_dict(item)
    if not isinstance(ctx, dict):
        return None
    values = ctx.get("behavior_logprobs")
    if not isinstance(values, list):
        return None
    if len(values) != length:
        return None
    if xp_name == "mlx":
        return mx.array(values)
    device = item["input_ids"].device
    return torch_module.tensor(values, device=device, dtype=torch_module.float32)


def _effective_steps(planned_steps: int, batch_size: int, grad_accum: int, max_passes: float) -> int:
    budgeted = math.ceil(max(1, batch_size) * max(max_passes, 0.25) / max(1, grad_accum))
    return max(1, min(planned_steps, budgeted))


def _scalar_advantage(rating: int, ema_mean: float, cfg) -> float:
    adv = float(rating - ema_mean)
    if rating > 0:
        adv *= float(cfg.get("pos_weight", 1.0))
    clip = float(cfg.get("adv_clip", 2.0))
    return max(-clip, min(clip, adv))


def _traj_gate(value, lo, hi, xp):
    one = xp.array(1.0) if hasattr(xp, "array") else 1.0
    if hasattr(xp, "where"):
        below = xp.where(value < lo, xp.square(value / lo), one)
        above = xp.where(value > hi, xp.square(hi / value), one)
        return below * above
    if value < lo:
        return (value / lo) ** 2
    if value > hi:
        return (hi / value) ** 2
    return one


def _make_loss_fn_mlx(model, tokenized, ref_logprobs, cfg, ema_mean):
    tc_lo, tc_hi = cfg["token_clip"]
    kl_c = cfg["kl_coeff"]
    loss_name = str(cfg.get("loss_fn", "mis-po")).lower()

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
        adv = mx.array(_scalar_advantage(rating, ema_mean, cfg), dtype=new_r.dtype)
        if loss_name == "reinforce++":
            per_token_kl = new_r - ref_r
            adjusted = mx.stop_gradient(adv - kl_c * per_token_kl)
            return -mx.mean(new_r * adjusted)
        behavior_r = _behavior_logprobs(item, int(new_r.shape[0]), "mlx") if loss_name == "mis-po" else ref_r
        if behavior_r is None:
            behavior_r = ref_r

        ratio = mx.exp(new_r - behavior_r)
        clipped_ratio = mx.clip(ratio, tc_lo, tc_hi)
        surr_1 = ratio * adv
        surr_2 = clipped_ratio * adv
        actor = -mx.mean(mx.minimum(surr_1, surr_2) if rating > 0 else mx.maximum(surr_1, surr_2))
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


def _save_partial_cuda(model, config, run_id: str) -> str:
    target_dir = _resume_checkpoint_dir(config, run_id)
    model.save_pretrained(target_dir, safe_serialization=True)
    return str(next((path for path in target_dir.glob("*.safetensors")), target_dir / "adapter_model.bin"))


def _torch_target_modules(torch, model, target: str = "attention"):
    common = []
    seen = set()
    attention = {"q_proj", "k_proj", "v_proj", "o_proj", "wq", "wk", "wv", "wo", "c_attn", "c_proj"}
    mlp = {"gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"}
    preferred = attention if target == "attention" else attention | mlp
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in seen:
                seen.add(leaf)
                if leaf in preferred:
                    common.append(leaf)
    if _strict_target_selection(target) and not common:
        raise RuntimeError(
            f"no_{target}_modules_found_for_lora:strict_targeting_enabled:{','.join(sorted(seen))}"
        )
    return common or sorted(seen)


def _tokenize_torch(tokenizer, item, torch_device):
    prompt, response = item["prompt"], item["response"]
    max_seq_len = max(128, int(item.get("max_seq_len") or 2048))
    ctx = None
    if item.get("context"):
        try:
            ctx = json.loads(item["context"])
        except (json.JSONDecodeError, TypeError):
            ctx = None
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = ctx.get("messages") if isinstance(ctx, dict) and isinstance(ctx.get("messages"), list) else None
        if msgs is None:
            msgs = []
            if isinstance(ctx, dict) and "system" in ctx:
                msgs.append({"role": "system", "content": ctx["system"]})
            msgs.append({"role": "user", "content": prompt})
        prompt_text = _apply_chat_template(tokenizer, msgs, tokenize=False, add_generation_prompt=True)
        full_text = _apply_chat_template(
            tokenizer,
            msgs + [{"role": "assistant", "content": response}],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    elif isinstance(ctx, dict) and isinstance(ctx.get("messages"), list):
        prompt_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in ctx["messages"])
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        full_ids = tokenizer(
            prompt_text + "\nAssistant: " + response,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]
    else:
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        full_ids = tokenizer(prompt + response, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    offset = max(0, int(full_ids.shape[0]) - max_seq_len)
    full_ids = full_ids[offset:]
    return {
        "input_ids": full_ids.to(torch_device),
        "response_start": max(1, int(prompt_ids.shape[0]) - offset),
        "rating": item["rating"],
        "id": item["id"],
    }


def _compute_logprobs_torch(model, input_ids, torch):
    with torch.no_grad():
        logits = model(input_ids=input_ids.unsqueeze(0)).logits.squeeze(0)
        lp = torch.log_softmax(logits, dim=-1)
        return lp[:-1].gather(-1, input_ids[1:].unsqueeze(-1)).squeeze(-1)



def load_model(model_name, lora_rank=16, adapter_path=None):
    _ensure_mlx()
    model, tokenizer = mlx_load(model_name)
    target = "attention"
    if adapter_path:
        cfg_path = Path(adapter_path).with_name("adapter_config.json")
        if cfg_path.exists():
            try:
                target = json.loads(cfg_path.read_text()).get("target_modules", target)
            except Exception:
                pass
    model = _apply_lora(model, rank=lora_rank, target=target)
    if adapter_path and Path(adapter_path).exists():
        model.load_weights(adapter_path, strict=False)
    model.eval()
    return model, tokenizer


_PUBLISH_CANARY = [
    ("Reply with only YES if 11 is prime, otherwise NO.", lambda r: r == "YES"),
    ("Reply with only YES if 15 is prime, otherwise NO.", lambda r: r == "NO"),
    ("Write HELLO in all caps and nothing else.", lambda r: r == "HELLO"),
    ('Reply with exactly {"ok":true} and nothing else.', lambda r: r == '{"ok":true}'),
    ("Return only the number 12. What is 3 * 4?", lambda r: r == "12"),
    ("Reply with one word only: red, blue, or yellow. Name a primary color.", lambda r: r in {"red", "blue", "yellow"}),
]


def _normalize_canary_output(text: str) -> str:
    return " ".join(text.strip().split())


def _publish_canary_score(outputs) -> int:
    return sum(int(check(_normalize_canary_output(text))) for (__, check), text in zip(_PUBLISH_CANARY, outputs))


def _publish_gate_mlx(config, adapter_path=None):
    from mlx_lm import generate

    model, tok = load_model(config["model"], lora_rank=config.get("lora_rank", 16), adapter_path=adapter_path)
    try:
        outputs = []
        for prompt, _ in _PUBLISH_CANARY:
            formatted = _apply_chat_template(tok, [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
            out = generate(model, tok, prompt=formatted, max_tokens=24, verbose=False)
            outputs.append(out)
            mlx_drain()
        return _publish_canary_score(outputs)
    finally:
        del model, tok
        _free_mlx()


def _publish_gate_cuda(config, adapter_path=None):
    torch, AutoModelForCausalLM, AutoTokenizer, _, PeftModel, _ = _torch_stack()
    tok = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        trust_remote_code=True,
        torch_dtype=CUDABackend().preferred_dtype(),
        low_cpu_mem_usage=True,
    ).to("cuda")
    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, str(Path(adapter_path).parent), is_trainable=False)
    try:
        outputs = []
        for prompt, _ in _PUBLISH_CANARY:
            formatted = _apply_chat_template(tok, [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
            ids = tok(formatted, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
            out = model.generate(ids, max_new_tokens=24, pad_token_id=tok.eos_token_id, do_sample=False)
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            outputs.append(text)
        return _publish_canary_score(outputs)
    finally:
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()


def publish_gate(config, adapter_path):
    if not config.get("publish_gate_enabled", True):
        return {"ok": True, "reason": "disabled"}
    try:
        backend = _select_backend(config)
        scorer = _publish_gate_mlx if backend.name == "mlx" else _publish_gate_cuda
        base = scorer(config, None)
        candidate = scorer(config, adapter_path)
    except Exception as exc:
        return {"ok": False, "reason": "gate_error", "detail": str(exc)}
    max_drop = int(config.get("publish_gate_max_drop", 0))
    return {
        "ok": candidate + max_drop >= base,
        "base_score": base,
        "candidate_score": candidate,
        "reason": "score_ok" if candidate + max_drop >= base else "score_drop",
    }


def _save_partial_mlx(model, config, run_id: str) -> str:
    _ensure_mlx()
    target_dir = _resume_checkpoint_dir(config, run_id)
    adapter_file = target_dir / "adapter.safetensors"
    portable_file = target_dir / "adapter_model.safetensors"
    lora_weights = {name: value for name, value in nn.utils.tree_flatten(model.trainable_parameters()) if "lora" in name.lower()}
    mx.save_safetensors(str(adapter_file), lora_weights)
    mx.save_safetensors(str(portable_file), lora_weights)
    (target_dir / "adapter_config.json").write_text(json.dumps({
        "r": config.get("lora_rank", 8),
        "lora_alpha": config.get("lora_rank", 8),
        "base_model_name_or_path": config["model"],
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "target_modules": _lora_target(config),
        "base_model": config["model"],
    }))
    return str(adapter_file)


def _make_loss_fn_torch(model, tokenized, ref_logprobs, cfg, ema_mean, torch):
    tc_lo, tc_hi = cfg["token_clip"]
    kl_c = cfg["kl_coeff"]
    loss_name = str(cfg.get("loss_fn", "mis-po")).lower()

    def loss_fn(idx: int):
        item = tokenized[idx]
        ids, start, rating = item["input_ids"], item["response_start"], item["rating"]
        ref = ref_logprobs[idx]
        logits = model(input_ids=ids.unsqueeze(0)).logits.squeeze(0)
        new_lp = torch.log_softmax(logits, dim=-1)[:-1].gather(-1, ids[1:].unsqueeze(-1)).squeeze(-1)
        new_r, ref_r = new_lp[start - 1:], ref[start - 1:]
        adv = torch.tensor(_scalar_advantage(rating, ema_mean, cfg), device=ids.device, dtype=new_r.dtype)
        if loss_name == "reinforce++":
            per_token_kl = new_r - ref_r
            adjusted = (adv - kl_c * per_token_kl).detach()
            return -(new_r * adjusted).mean()
        behavior_r = _behavior_logprobs(item, int(new_r.shape[0]), "torch", torch_module=torch) if loss_name == "mis-po" else ref_r
        if behavior_r is None:
            behavior_r = ref_r

        ratio = torch.exp(new_r - behavior_r)
        clipped_ratio = ratio.clamp(tc_lo, tc_hi)
        surr_1 = ratio * adv
        surr_2 = clipped_ratio * adv
        actor = -(torch.minimum(surr_1, surr_2) if rating > 0 else torch.maximum(surr_1, surr_2)).mean()
        kl = torch.exp(ref_r - new_r) - (ref_r - new_r) - 1
        traj = torch.exp((new_r - ref_r).mean())
        return actor * _trajectory_scale_torch(traj, cfg, torch) + kl_c * kl.mean()

    return loss_fn


def _attempt_train_mlx(config, conn, backend, hardware, attempt: int):
    _ensure_mlx()
    reserve_bytes = _os_reserve_bytes(True)
    guard = AdaptiveMemoryGuard(backend, _preload_limit_bytes(hardware, reserve_bytes))
    latest = db.latest_adapter(conn)
    resume = _resume_state(conn, config)
    guard.check("before_model_load")
    try:
        model = tokenizer = tokenized = ref_lps = saved_lora = None
        restore_checkpoint = lambda: None
        model, tokenizer = mlx_load(config["model"])
        model = _apply_lora(model, rank=config.get("lora_rank", 8), target=_lora_target(config))
        if resume and resume.get("checkpoint_path"):
            model.load_weights(resume["checkpoint_path"], strict=False)
        elif latest and Path(latest["path"]).exists():
            model.load_weights(latest["path"], strict=False)
        model.eval()

        model_bytes = max(backend.current_memory_bytes(), backend.active_memory_bytes())
        hardware = _settle_backend_hardware(backend, backend.hardware(), rounds=2, delay=0.5)
        plan = _plan_strategy(config, hardware, model_bytes)
        for _ in range(attempt):
            plan = _degrade_plan(plan)
        if plan is None:
            _log_event("skip", backend="mlx", reason="insufficient_budget", model_gb=round(model_bytes / 1e9, 3))
            return _skip("insufficient_budget", backend="mlx", model_gb=round(model_bytes / 1e9, 3))

        guard = AdaptiveMemoryGuard(backend, plan.memory_limit_bytes)
        guard.log_step("post_load")

        if resume:
            batch = db.get_feedback_by_ids(conn, resume["batch_ids"])
            fresh_ids = list(resume["fresh_ids"])
            fresh = db.get_feedback_by_ids(conn, fresh_ids)
        else:
            batch, fresh_ids, fresh = _build_batch(conn, plan.effective_batch_size, config.get("replay_ratio", 0.25))
        if not batch:
            return _skip("empty_batch", backend="mlx")

        tokenized = [_tokenize_mlx(tokenizer, {**item, "max_seq_len": config.get("max_seq_len", 2048)}) for item in batch]
        saved_lora = _disable_lora(model)
        ref_lps = [_compute_logprobs_mlx(model, item["input_ids"]) for item in tokenized]
        mx.eval(*ref_lps)
        _enable_lora(model, saved_lora)
        restore_checkpoint = _enable_grad_checkpoint_mlx(model) if plan.aggressive_checkpointing else (lambda: None)
        model.train()

        cfg = _tighten_small_batch_cfg(_build_train_cfg(config, plan), len(batch))
        try:
            ema_mean, ema_count = db.get_ema(conn)
            total_target_steps = int(resume["total_steps"]) if resume else int(cfg["steps"])
            remaining_steps = int(resume["remaining_steps"]) if resume else total_target_steps
            cfg["steps"] = min(remaining_steps, _slice_steps(config, remaining_steps))
            loss_fn = _make_loss_fn_mlx(model, tokenized, ref_lps, cfg, ema_mean)
            vg = nn.value_and_grad(model, loss_fn)
            opt = optim.Adam(learning_rate=cfg["lr"])
            backend.reset_peak_memory()

            total_loss = 0.0
            total_steps = 0
            for step in range(cfg["steps"]):
                for retry in range(_pressure_retry_limit(config) + 1):
                    try:
                        block = _background_block_reason(config, backend, backend.hardware())
                        if block:
                            if block in _TRANSIENT_RESOURCE_BLOCKS and retry < _pressure_retry_limit(config):
                                _pressure_cooldown(backend, config, step, retry, reason=block)
                                continue
                            _log_event("skip", backend="mlx", reason=block, step=step)
                            return _skip(block, backend="mlx", step=step)
                        if not _has_minimum_headroom(backend.hardware()):
                            raise MemoryError("memory pressure while training")
                        guard.check(f"before_step_{step}")
                        acc_grads = None
                        step_loss = 0.0
                        for micro in range(cfg["grad_accum"]):
                            idx = (step * cfg["grad_accum"] + micro) % len(tokenized)
                            loss, grads = vg(idx)
                            mx.eval(loss)
                            step_loss += loss.item()
                            acc_grads = grads if acc_grads is None else tree_map(lambda a, b: a + b, acc_grads, grads)

                        acc_grads = tree_map(lambda grad: grad / cfg["grad_accum"], acc_grads)
                        flat = [grad for _, grad in nn.utils.tree_flatten(acc_grads)]
                        if flat:
                            norm = mx.sqrt(sum(mx.sum(grad * grad) for grad in flat))
                            scale = mx.minimum(mx.array(cfg["grad_clip"]) / (norm + 1e-6), mx.array(1.0))
                            acc_grads = tree_map(lambda grad: grad * scale, acc_grads)
                            grad_norm = float(norm.item())
                        else:
                            grad_norm = 0.0

                        opt.update(model, acc_grads)
                        mx.eval(model.parameters(), opt.state)
                        total_loss += step_loss / cfg["grad_accum"]
                        total_steps += 1
                        _log_event("opt_step", backend="mlx", step=step, loss=round(step_loss / cfg["grad_accum"], 6), grad_norm=round(grad_norm, 6))
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

        remaining_steps -= total_steps
        if remaining_steps > 0:
            run_id = resume["run_id"] if resume else f"{int(time.time())}-{os.getpid()}"
            checkpoint_path = _save_partial_mlx(model, config, run_id)
            db.save_training_state(conn, {
                "run_id": run_id,
                "model": config["model"],
                "loss_fn": str(cfg["loss_fn"]).lower(),
                "checkpoint_path": checkpoint_path,
                "batch_ids": [item["id"] for item in batch],
                "fresh_ids": fresh_ids,
                "remaining_steps": remaining_steps,
                "total_steps": total_target_steps,
                "parent_version": resume.get("parent_version") if resume else (latest["version"] if latest else None),
            })
            return {
                "status": "paused",
                "reason": "resume_pending",
                "remaining_steps": remaining_steps,
                "batch_size": len(batch),
                "steps": total_steps,
                "backend": "mlx",
                "loss_fn": cfg["loss_fn"],
                "lora_target": _lora_target(config),
            }

        for item in fresh:
            ema_count += 1
            ema_mean = cfg["ema_decay"] * ema_mean + (1 - cfg["ema_decay"]) * item["rating"]
        parent_v = resume.get("parent_version") if resume else (latest["version"] if latest else None)
        new_v, temp_dir, save_dir = _stage_adapter_dir(_next_adapter_version(conn), config)
        adapter_file = str(temp_dir / "adapter.safetensors")
        portable_adapter = str(temp_dir / "adapter_model.safetensors")
        lora_weights = {name: value for name, value in nn.utils.tree_flatten(model.trainable_parameters()) if "lora" in name.lower()}
        try:
            mx.save_safetensors(adapter_file, lora_weights)
            mx.save_safetensors(portable_adapter, lora_weights)
            (temp_dir / "adapter_config.json").write_text(json.dumps({
                "r": config.get("lora_rank", 8),
                "lora_alpha": config.get("lora_rank", 8),
                "base_model_name_or_path": config["model"],
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "target_modules": _lora_target(config),
                "base_model": config["model"],
            }))
            _commit_adapter_dir(temp_dir, save_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        metrics = {
            "status": "trained",
            "avg_loss": total_loss / max(total_steps, 1),
            "batch_size": len(batch),
            "steps": cfg["steps"],
            "ema_mean": ema_mean,
            "ema_count": ema_count,
            "peak_memory_gb": round(backend.peak_memory_bytes() / 1e9, 3),
            "backend": "mlx",
            "loss_fn": cfg["loss_fn"],
            "lora_target": _lora_target(config),
            "version": new_v,
            "path": str(save_dir / "adapter.safetensors"),
            "feedback_ids": fresh_ids,
        }
        db.record_training_round(conn, ema_mean, ema_count, new_v, str(save_dir / "adapter.safetensors"), parent_v, metrics, fresh_ids)
        db.clear_training_state(conn)
        if config.get("adapter_keep"):
            for path in db.cleanup_adapters(conn, keep=config["adapter_keep"]):
                shutil.rmtree(Path(path).parent, ignore_errors=True)
        return metrics
    finally:
        model = tokenizer = tokenized = ref_lps = saved_lora = None
        _release_backend_memory(backend)


def _attempt_train_cuda(config, conn, backend, hardware, attempt: int):
    torch, AutoModelForCausalLM, AutoTokenizer, LoraConfig, PeftModel, get_peft_model = _torch_stack()
    reserve_bytes = _os_reserve_bytes(False)
    guard = AdaptiveMemoryGuard(backend, _preload_limit_bytes(hardware, reserve_bytes))
    latest = db.latest_adapter(conn)
    resume = _resume_state(conn, config)

    guard.check("before_model_load")
    try:
        model = base = tokenizer = tokenized = ref_lps = None
        tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            config["model"],
            trust_remote_code=True,
            torch_dtype=backend.preferred_dtype(),
            low_cpu_mem_usage=True,
        ).to(backend.device)

        if resume and resume.get("checkpoint_path") and Path(resume["checkpoint_path"]).exists():
            model = PeftModel.from_pretrained(base, str(Path(resume["checkpoint_path"]).parent), is_trainable=True)
        elif latest and Path(latest["path"]).exists():
            model = PeftModel.from_pretrained(base, str(Path(latest["path"]).parent), is_trainable=True)
        else:
            target_modules = _torch_target_modules(torch, base, _lora_target(config))
            model = get_peft_model(
                base,
                LoraConfig(
                    r=config.get("lora_rank", 8),
                    lora_alpha=config.get("lora_rank", 8),
                    target_modules=target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )

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
            except Exception:
                pass
        model.train()

        if resume:
            batch = db.get_feedback_by_ids(conn, resume["batch_ids"])
            fresh_ids = list(resume["fresh_ids"])
            fresh = db.get_feedback_by_ids(conn, fresh_ids)
        else:
            batch, fresh_ids, fresh = _build_batch(conn, plan.effective_batch_size, config.get("replay_ratio", 0.25))
        if not batch:
            return _skip("empty_batch", backend="cuda")

        tokenized = [_tokenize_torch(tokenizer, {**item, "max_seq_len": config.get("max_seq_len", 2048)}, backend.device) for item in batch]
        ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
        with ctx:
            ref_lps = [_compute_logprobs_torch(model, item["input_ids"], torch) for item in tokenized]

        cfg = _tighten_small_batch_cfg(_build_train_cfg(config, plan), len(batch))
        ema_mean, ema_count = db.get_ema(conn)
        total_target_steps = int(resume["total_steps"]) if resume else int(cfg["steps"])
        remaining_steps = int(resume["remaining_steps"]) if resume else total_target_steps
        cfg["steps"] = min(remaining_steps, _slice_steps(config, remaining_steps))
        loss_fn = _make_loss_fn_torch(model, tokenized, ref_lps, cfg, ema_mean, torch)
        optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=cfg["lr"])
        backend.reset_peak_memory()

        total_loss = 0.0
        total_steps = 0
        for step in range(cfg["steps"]):
            for retry in range(_pressure_retry_limit(config) + 1):
                try:
                    block = _background_block_reason(config, backend, backend.hardware())
                    if block:
                        if block in _TRANSIENT_RESOURCE_BLOCKS and retry < _pressure_retry_limit(config):
                            _pressure_cooldown(backend, config, step, retry, reason=block)
                            continue
                        _log_event("skip", backend="cuda", reason=block, step=step)
                        return _skip(block, backend="cuda", step=step)
                    if not _has_minimum_headroom(backend.hardware()):
                        raise MemoryError("memory pressure while training")
                    guard.check(f"before_step_{step}")
                    optimizer.zero_grad(set_to_none=True)
                    step_loss = 0.0
                    for micro in range(cfg["grad_accum"]):
                        idx = (step * cfg["grad_accum"] + micro) % len(tokenized)
                        loss = loss_fn(idx) / cfg["grad_accum"]
                        loss.backward()
                        step_loss += float(loss.detach().item())
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]).detach().item())
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    backend.synchronize()
                    backend.clear_cache()
                    _log_event("opt_step", backend="cuda", step=step, loss=round(step_loss, 6), grad_norm=round(grad_norm, 6))
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

        remaining_steps -= total_steps
        if remaining_steps > 0:
            run_id = resume["run_id"] if resume else f"{int(time.time())}-{os.getpid()}"
            checkpoint_path = _save_partial_cuda(model, config, run_id)
            db.save_training_state(conn, {
                "run_id": run_id,
                "model": config["model"],
                "loss_fn": str(cfg["loss_fn"]).lower(),
                "checkpoint_path": checkpoint_path,
                "batch_ids": [item["id"] for item in batch],
                "fresh_ids": fresh_ids,
                "remaining_steps": remaining_steps,
                "total_steps": total_target_steps,
                "parent_version": resume.get("parent_version") if resume else (latest["version"] if latest else None),
            })
            return {
                "status": "paused",
                "reason": "resume_pending",
                "remaining_steps": remaining_steps,
                "batch_size": len(batch),
                "steps": total_steps,
                "backend": "cuda",
                "loss_fn": cfg["loss_fn"],
                "lora_target": _lora_target(config),
            }

        for item in fresh:
            ema_count += 1
            ema_mean = cfg["ema_decay"] * ema_mean + (1 - cfg["ema_decay"]) * item["rating"]
        parent_v = resume.get("parent_version") if resume else (latest["version"] if latest else None)
        new_v, temp_dir, save_dir = _stage_adapter_dir(_next_adapter_version(conn), config)
        try:
            model.save_pretrained(temp_dir, safe_serialization=True)
            _commit_adapter_dir(temp_dir, save_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        adapter_file = next((str(path) for path in save_dir.glob("*.safetensors")), str(save_dir / "adapter_model.bin"))

        metrics = {
            "status": "trained",
            "avg_loss": total_loss / max(total_steps, 1),
            "batch_size": len(batch),
            "steps": cfg["steps"],
            "ema_mean": ema_mean,
            "ema_count": ema_count,
            "peak_memory_gb": round(backend.peak_memory_bytes() / 1e9, 3),
            "backend": "cuda",
            "loss_fn": cfg["loss_fn"],
            "lora_target": _lora_target(config),
            "version": new_v,
            "path": adapter_file,
            "feedback_ids": fresh_ids,
        }
        db.record_training_round(conn, ema_mean, ema_count, new_v, adapter_file, parent_v, metrics, fresh_ids)
        db.clear_training_state(conn)
        if config.get("adapter_keep"):
            for path in db.cleanup_adapters(conn, keep=config["adapter_keep"]):
                shutil.rmtree(Path(path).parent, ignore_errors=True)
        return metrics
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


def train_result(config, conn):
    lock_fd = None
    if not config.get("_skip_lock_once"):
        lock_fd = _acquire_lock()
        if lock_fd is None:
            _log_event("skip", reason="train_lock_held")
            return _skip("train_lock_held")

    try:
        if config.get("low_priority", True):
            _set_low_priority()

        resume = _resume_state(conn, config)
        if not resume and db.count_trainable_untrained(conn) < config.get("batch_min", 24):
            return _skip("below_threshold", trainable=db.count_trainable_untrained(conn), batch_min=config.get("batch_min", 24))

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
            _log_event(
                "skip",
                backend=backend.name,
                reason="insufficient_headroom",
                available_gb=round((hardware.available_memory_bytes or 0) / 1e9, 3),
            )
            return _skip("insufficient_headroom", backend=backend.name, available_gb=round((hardware.available_memory_bytes or 0) / 1e9, 3))
        block = _background_block_reason(config, backend, hardware)
        if block:
            _log_event(
                "skip",
                backend=backend.name,
                reason=block,
                available_gb=round((hardware.available_memory_bytes or 0) / 1e9, 3),
                host_available_gb=round((getattr(hardware, "system_available_memory_bytes", 0) or 0) / 1e9, 3),
            )
            return _skip(
                block,
                backend=backend.name,
                available_gb=round((hardware.available_memory_bytes or 0) / 1e9, 3),
                host_available_gb=round((getattr(hardware, "system_available_memory_bytes", 0) or 0) / 1e9, 3),
            )
        _log_event(
            "train_start",
            backend=backend.name,
            device=getattr(hardware, "device_name", backend.name),
            total_gb=round(hardware.total_memory_bytes / 1e9, 3),
            available_gb=round((hardware.available_memory_bytes or 0) / 1e9, 3),
            host_available_gb=round((getattr(hardware, "system_available_memory_bytes", 0) or 0) / 1e9, 3),
        )

        for attempt in range(2):
            try:
                hardware = _settle_backend_hardware(backend, backend.hardware())
                if not _has_minimum_headroom(hardware):
                    _log_event(
                        "skip",
                        backend=backend.name,
                        attempt=attempt,
                        reason="insufficient_headroom",
                        available_gb=round((hardware.available_memory_bytes or 0) / 1e9, 3),
                    )
                    return _skip("insufficient_headroom", backend=backend.name, attempt=attempt)
                block = _background_block_reason(config, backend, hardware)
                if block:
                    _log_event(
                        "skip",
                        backend=backend.name,
                        attempt=attempt,
                        reason=block,
                        available_gb=round((hardware.available_memory_bytes or 0) / 1e9, 3),
                        host_available_gb=round((getattr(hardware, "system_available_memory_bytes", 0) or 0) / 1e9, 3),
                    )
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
            except Exception as exc:
                _release_backend_memory(backend)
                _log_event("train_error", attempt=attempt, error=type(exc).__name__, detail=str(exc))
                if "missing_cuda_training_dependency:" in str(exc):
                    return _skip("missing_cuda_training_dependency", detail=str(exc))
                if attempt == 0 and _is_retryable_memory_error(exc):
                    continue
                if _is_retryable_memory_error(exc):
                    return _skip("memory_pressure", detail=str(exc), backend=backend.name, attempt=attempt)
                raise
        return _skip("no_training_result")
    finally:
        _release_lock(lock_fd)


def train(config, conn):
    result = train_result(config, conn)
    if result.get("status") == "trained":
        return {k: v for k, v in result.items() if k != "status"}
    return None


def _adapter_dir(config=None):
    root = Path((config or {}).get("adapter_root", Path.home() / ".reinforceclaw" / "adapters"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resume_dir(config=None):
    root = _adapter_dir(config) / ".resume"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resume_checkpoint_dir(config, run_id: str) -> Path:
    path = _resume_dir(config) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stage_adapter_dir(version: int, config=None) -> tuple[int, Path, Path]:
    root = _adapter_dir(config)
    current = version
    while True:
        final_dir = root / f"v{current}"
        temp_dir = root / f".v{current}.tmp-{os.getpid()}-{int(time.time() * 1000)}"
        if not final_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir.mkdir(parents=True, exist_ok=True)
            return current, temp_dir, final_dir
        current += 1


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
    for cmd in ["convert-lora-to-gguf", "python3 -m llama_cpp.convert_lora"]:
        try:
            result = subprocess.run(
                cmd.split() + [str(safetensors_file), "--outfile", str(gguf_file)],
                capture_output=True,
                timeout=120,
            )
            if result.returncode == 0 and gguf_file.exists():
                return str(gguf_file)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def hot_swap(server_type, adapter_path, model_name):
    import requests

    adapter_dir = str(Path(adapter_path).parent)
    if server_type == "ollama":
        gguf = _convert_to_gguf(adapter_dir)
        adapter_ref = gguf if gguf else adapter_dir
        modelfile = f"FROM {model_name}\nADAPTER {adapter_ref}\n"
        try:
            response = requests.post(
                "http://localhost:11434/api/create",
                json={"name": f"{model_name}-reinforceclaw", "modelfile": modelfile},
                timeout=60,
            )
            return response.status_code == 200
        except requests.ConnectionError:
            return False
    if server_type == "lmstudio":
        try:
            response = requests.post(
                "http://localhost:1234/v1/lora/load",
                json={"path": adapter_dir},
                timeout=30,
            )
            return response.status_code == 200
        except requests.ConnectionError:
            print(f"Load adapter in LM Studio from: {adapter_dir}")
            return None
    if server_type == "vllm":
        try:
            response = requests.post(
                "http://localhost:8000/v1/load_lora_adapter",
                json={"lora_name": "reinforceclaw", "lora_path": adapter_dir},
                timeout=30,
            )
            return response.status_code == 200
        except requests.ConnectionError:
            return False
    print(f"Restart your server with adapter: {adapter_path}")
    return None
