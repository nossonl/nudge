"""CLI. argparse. No Click, no Typer. Setup wizard + all commands."""
# two entry points: `reinforceclaw <cmd>` from terminal, `/rl <cmd>` from inside agents.
# both hit the same functions. wizard is `reinforceclaw init`.

import argparse
import copy
import secrets
import json
import math
import os
import platform
import re
import shlex
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
try:
    import termios
    import tty
except ImportError:  # Windows fallback: typed prompts still work.
    termios = tty = None

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table

from . import __version__, db, trainer, presets, profile
from .hooks._common import pop_pending, reset_pending, restore_pending

console = Console()
CONFIG_PATH = Path.home() / ".reinforceclaw" / "config.json"
ADAPTER_ROOT = Path.home() / ".reinforceclaw" / "adapters"
OPENCLAW_BRIDGE_CONFIG = Path.home() / ".reinforceclaw" / "openclaw_bridge.json"
TRAIN_RETRY_PATH = Path.home() / ".reinforceclaw" / "train.retry"
RESET_MARK_PATH = Path.home() / ".reinforceclaw" / "reset.marker"

PRESETS = {
    "careful": "Safest updates. Highest KL, slowest drift.",
    "balanced": "Stable MIS-PO default. Sweep winner.",
    "aggressive": "Faster drift. More overfit risk.",
}

DEFAULTS = {
    "loss_fn": "mis-po",
    "lora_target": "attention",
    "token_clip": [0.5, 2.0], "kl_coeff": 0.001, "lora_rank": 16,
    "grad_accum": 2, "grad_clip": 1.0, "batch_min": 32, "batch_size": 4,
    "replay_ratio": 0.0, "ema_decay": 0.99, "pos_weight": 1.2,
    "adv_clip": 2.0, "adv_norm": True, "max_passes": 1.0,
    "pressure_retry_limit": 2, "pressure_cooldown_s": 3.0,
    "background_slice_steps": 2,
    "adapter_keep": 20,
    "train_schedule": "auto",
    "schedule_window_minutes": 180,
    # speed-up knobs (opt-in; 0/None = disabled so validated baseline stands).
    "lora_plus_ratio": 0.0,    # B-matrix LR multiplier; ~16 roughly doubles convergence
    "use_liger": False,        # Liger-Kernel fused CE + RMSNorm (CUDA)
    "compile_backend": "none", # "reduce-overhead" | "max-autotune" | "default" (CUDA)
    "lora_init": "default",    # "pissa" | "olora" | "loftq" | "eva" — PEFT init recipe
    "sdpa_backend": "auto",    # PyTorch SDPA hint; "auto" lets torch pick FA4 on Blackwell
    "trust_remote_code": False,
    "agent_admin_commands": False,
    "feedback_keep_trained_rows": 100000,
    "feedback_keep_untrained_rows": 100000,
    "feedback_keep_days": 0,
}
_PROFILE_CACHE = {}
_PROFILE_CACHE_MAX = 64

from .models import MODELS  # model catalog lives in models.py

LOGO = r"""
[bold green]╭────────────────────────────╮
│        ReinforceClaw       │
│  local RL for your agents  │
╰────────────────────────────╯[/bold green]"""
WIZARD_BACK = object()


def _read_json(path, default=None, *, follow_symlink=True):
    if path.is_symlink() and not follow_symlink:
        raise PermissionError(f"refusing symlink JSON path: {path}")
    if not path.exists():
        return {} if default is None else default
    try:
        target = path.resolve(strict=True) if path.is_symlink() else path
        db.secure_private_file(target)
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}") from exc
    except OSError:
        return {} if default is None else default


_TUNED_KEYS = ("lr", "traj_clip", "steps", "kl_coeff", "lora_rank", "lora_alpha",
               "lora_target", "batch_min", "batch_size", "grad_accum", "pos_weight",
               "replay_ratio", "token_clip")


def _auto_tuned_values(cfg: dict) -> dict:
    model, preset = cfg.get("model", ""), cfg.get("preset", "balanced")
    mp = cfg.get("model_profile") if cfg.get("model_profile_model") == model else None
    if isinstance(mp, dict):
        try:
            prof = profile.ModelProfile(**mp)
        except TypeError:
            prof = None
    else:
        prof = None
    if prof is None:
        prof = _PROFILE_CACHE.pop(model, None) or profile.detect(model)
        if len(_PROFILE_CACHE) >= _PROFILE_CACHE_MAX:
            _PROFILE_CACHE.pop(next(iter(_PROFILE_CACHE)))
        _PROFILE_CACHE[model] = prof
    tuned = presets.pick(prof, preset)
    tuned["model_profile_model"] = model
    return tuned


def _resolve_config(cfg: dict) -> dict:
    if not cfg or cfg.get("tuning_mode") == "custom" or not cfg.get("model"):
        return cfg
    tuned = _auto_tuned_values(cfg)
    resolved = dict(cfg)
    for key in _TUNED_KEYS:
        if key in tuned:
            resolved[key] = tuned[key]
    resolved["model_profile"] = tuned["model_profile"]
    resolved["model_profile_model"] = tuned["model_profile_model"]
    resolved["tuning_mode"] = "auto"
    return resolved


def load_config(*, persist=True):
    cfg = _read_json(CONFIG_PATH, follow_symlink=False)
    resolved = _resolve_config(cfg)
    if persist and resolved.get("tuning_mode") == "auto" and not _same_json(resolved, cfg):
        save_config(resolved)
    return resolved


def save_config(cfg):
    _write_json_atomic(CONFIG_PATH, cfg, follow_symlink=False)


def _same_json(a, b):
    return json.dumps(a, sort_keys=True, separators=(",", ":")) == json.dumps(b, sort_keys=True, separators=(",", ":"))


def _clamp(val, lo, hi):
    return max(lo, min(val, hi))


def _load_model_cfg():
    cfg = load_config()
    if cfg.get("model"):
        return cfg
    console.print("[red]Run 'reinforceclaw init' first.[/red]")
    return None


def _swap_latest(cfg, conn):
    latest = db.latest_adapter(conn, model=cfg.get("model"))
    if not latest:
        return "none"
    ok = _load_adapter_status(cfg, latest["path"])
    return "loaded_ollama" if ok is True and cfg.get("server", "ollama") == "ollama" else "loaded" if ok is True else "failed" if ok is False else "manual"


def _load_adapter_status(cfg, path):
    return trainer.load_adapter(cfg.get("server", "ollama"), path, cfg.get("serve_model") or cfg["model"], cfg.get("server_url") or cfg.get("vllm_url"))


def _activate_candidate(conn, cfg, metrics):
    if not db.can_activate_adapter(conn, metrics["version"]):
        return None, "invalid"
    load_status = _load_adapter_status(cfg, metrics["path"])
    if load_status is False:
        return None, load_status
    active = db.activate_training_round(
        conn, metrics["version"], metrics["ema_mean"], metrics["ema_count"], metrics["feedback_ids"],
        max_rows=_keep_trained_rows(cfg),
        max_age_days=int(cfg.get("feedback_keep_days", 0)),
        max_untrained_rows=int(cfg.get("feedback_keep_untrained_rows", 100000)),
        model=cfg.get("model"),
    )
    return active, load_status


def _ollama_reinforced_name(cfg):
    return f"{str(cfg.get('serve_model') or cfg.get('model', '')).split('/')[-1]}-reinforceclaw"


def _pop_newest_pending(*sources):
    items = [item for item in (pop_pending(source) for source in sources) if item]
    if not items:
        return None
    items.sort(key=lambda item: float(item.get("ts") or 0), reverse=True)
    for item in items[1:]:
        restore_pending(item)
    return items[0]


def _keep_trained_rows(cfg):
    return int(cfg.get("feedback_keep_trained_rows", cfg.get("feedback_keep_rows", 100000)))


def _table(title, columns, rows=(), style="cyan"):
    t = Table(title=title, border_style=style)
    for col in columns:
        t.add_column(col[0], style=col[1]) if isinstance(col, tuple) else t.add_column(col)
    for row in rows:
        t.add_row(*(str(v) for v in row))
    return t


def _choice_parts(choice):
    return choice if isinstance(choice, tuple) else (choice, "")


def _key():
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        return {"[A": "up", "[B": "down", "[D": "back"}.get(sys.stdin.read(2), "esc")
    if ch in ("\r", "\n"):
        return "enter"
    if ch == " ":
        return "space"
    if ch in ("\x7f", "\b"):
        return "back"
    if ch in ("\x03", "\x04"):
        raise KeyboardInterrupt
    return ch


def _menu(title, choices, *, default=0, multi=False, selected=None, allow_back=False):
    if not (termios and tty and sys.stdin.isatty() and sys.stdout.isatty()):
        if multi:
            return [i for i, c in enumerate(choices) if Confirm.ask(_choice_parts(c)[0], default=i in (selected or []))]
        for i, c in enumerate(choices, 1):
            name, desc = _choice_parts(c)
            console.print(f"  [green]{i}[/green]. [bold]{name}[/bold]" + (f" - {desc}" if desc else ""))
        return _clamp(IntPrompt.ask(title, default=default + 1), 1, len(choices)) - 1
    pos = _clamp(default, 0, len(choices) - 1)
    picked = set(selected or [])
    fd, old = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())
    try:
        tty.setcbreak(fd)
        while True:
            console.clear()
            console.print(LOGO)
            console.print(f"[bold]{title}[/bold]")
            help_text = "Use ↑/↓, Space to select, Enter to continue." if multi else "Use ↑/↓ and Enter."
            if allow_back:
                help_text += " Backspace or ← goes back."
            console.print(f"[dim]{help_text}[/dim]")
            for i, c in enumerate(choices):
                name, desc = _choice_parts(c)
                cursor = "[green]›[/green]" if i == pos else " "
                mark = ("[green]●[/green]" if i in picked else "○") if multi else ("[green]●[/green]" if i == pos else "○")
                console.print(f"{cursor} {mark} [bold]{name}[/bold]" + (f"\n    [dim]{desc}[/dim]" if desc else ""))
            k = _key()
            if k == "up":
                pos = (pos - 1) % len(choices)
            elif k == "down":
                pos = (pos + 1) % len(choices)
            elif multi and k == "space":
                picked.symmetric_difference_update({pos})
            elif allow_back and k == "back":
                console.clear()
                return WIZARD_BACK
            elif k == "enter":
                console.clear()
                return sorted(picked or {pos}) if multi else pos
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _next_retry_delay(conn, base_delay=900):
    history = db.background_history(conn)
    if not history:
        return base_delay
    now = datetime.now()

    def option(hours_ahead):
        candidate = (now + timedelta(hours=hours_ahead)).replace(minute=0, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(hours=1)
        delay = max(base_delay, int((candidate - now).total_seconds()))
        stats = history.get(candidate.hour, {"pressure_count": 0, "success_count": 0})
        return stats["pressure_count"] - (0.5 * stats["success_count"]), delay

    return min((option(hour) for hour in range(24)), key=lambda item: item[0])[1]


def _set_panel(enabled):
    cfg = load_config()
    cfg["panel_enabled"] = enabled
    save_config(cfg)
    console.print("[green]Panel on.[/green]" if enabled else "[yellow]Panel off.[/yellow] Use /rl good or /rl bad.")


def _write_atomic(path, writer, *, backup=False, follow_symlink=True):
    path = Path(path).expanduser()
    if path.is_symlink() and not follow_symlink:
        raise PermissionError(f"refusing symlink write path: {path}")
    target = path.resolve(strict=False) if path.is_symlink() else path
    if target == path:
        db.secure_private_dir(path.parent)
    if backup and target.exists():
        backup_path = target.with_name(f"{target.name}.bak")
        if backup_path.is_symlink():
            raise PermissionError(f"refusing symlink backup path: {backup_path}")
        shutil.copy2(target, backup_path)
        db.secure_private_file(backup_path)
    tmp = target.with_name(f".{target.name}.{os.getpid()}-{secrets.token_hex(8)}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            writer(fh)
        os.replace(tmp, target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _write_json_atomic(path, payload, *, backup=False, follow_symlink=True):
    _write_atomic(path, lambda fh: (json.dump(payload, fh, indent=2), fh.write("\n")), backup=backup, follow_symlink=follow_symlink)


def _write_text_atomic(path, text, *, backup=False, follow_symlink=True):
    _write_atomic(path, lambda fh: fh.write(text), backup=backup, follow_symlink=follow_symlink)


def reset_state():
    lock_fd = trainer._acquire_lock()
    if lock_fd is None:
        raise RuntimeError("training is running; try again when the current update finishes")
    try:
        conn = db.connect()
        try:
            db.reset_all(conn)
        finally:
            conn.close()
        roots = {ADAPTER_ROOT}
        try:
            roots.add(trainer._adapter_dir(load_config(persist=False)))
        except (OSError, PermissionError, TypeError, ValueError):
            pass
        for root in roots:
            shutil.rmtree(root, ignore_errors=True)
        reset_pending()
        TRAIN_RETRY_PATH.unlink(missing_ok=True)
        _write_json_atomic(RESET_MARK_PATH, {"reset_at": datetime.now().isoformat()})
    finally:
        trainer._release_lock(lock_fd)


def rollback_adapter(conn, cfg, version=None):
    lock_fd = trainer._acquire_lock()
    if lock_fd is None:
        return None, "training is running; try again when the current update finishes"
    try:
        target = _rollback_target(conn, cfg, version)
        if not target:
            return None, "no previous adapter" if version is None else "no such adapter"
        status = _load_adapter_status(cfg, target["path"])
        if status is False:
            return target, "server reload failed; keeping current adapter active"
        prev = db.rollback_to(conn, version, model=cfg.get("model")) if version is not None else db.rollback(conn, model=cfg.get("model"))
        if prev:
            if status is None:
                return prev, "adapter rolled back; manual server reload may still be needed"
        return prev, None
    finally:
        trainer._release_lock(lock_fd)


def _rollback_target(conn, cfg, version=None):
    model = cfg.get("model")
    adapters = [a for a in db.list_adapters(conn, model=model) if a["status"] in {"active", "inactive", "rolled_back"}]
    if version is not None:
        return next((a for a in adapters if a["version"] == version), None)
    latest = db.latest_adapter(conn, model=model)
    return next((a for a in adapters if latest and a["version"] < latest["version"]), None)


def _ollama_tags():
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5, env=trainer._child_env())
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [line.split()[0] for line in result.stdout.splitlines()[1:] if line.split()]


def _default_ollama_tag(model_name):
    tags = _ollama_tags()
    if not tags:
        return model_name
    low_name = model_name.lower().rsplit("/", 1)[-1].replace("_", "-")
    return next((tag for tag in tags if tag.lower() == model_name.lower()), None) or \
        next((tag for tag in tags if tag.lower().split(":", 1)[0] in low_name or low_name in tag.lower()), tags[0])


def _backend_extra():
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return "mlx"
    if platform.system() == "Linux":
        return "cuda"
    return None


def _backend_ready(extra):
    import importlib.util
    needed = {"mlx": ("mlx", "mlx_lm"), "cuda": ("torch", "peft", "transformers")}.get(extra, ())
    return bool(needed) and all(importlib.util.find_spec(name) for name in needed)


def _install_backend(extra):
    if not extra or _backend_ready(extra):
        return True
    if not Confirm.ask(f"Install {extra.upper()} training dependencies now?", default=True):
        return False
    import subprocess
    source = Path(__file__).resolve().parent.parent
    target = f"{source}[{extra}]" if (source / "pyproject.toml").exists() else f"reinforceclaw[{extra}]"
    with console.status(f"[bold green]Preparing the {extra.upper()} training stack[/bold green]", spinner="aesthetic"):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", target],
            capture_output=True, text=True, timeout=1800, env=trainer._child_env(),
        )
    if result.returncode == 0:
        console.print(f"[green]{extra.upper()} training dependencies installed.[/green]")
        return True
    detail = "\n".join((result.stderr or result.stdout).splitlines()[-12:])
    console.print(f"[yellow]Could not install {extra.upper()} dependencies. Training can still be configured, but will not run until dependencies are installed.[/yellow]")
    if detail:
        console.print(f"[dim]{escape(detail)}[/dim]")
    return False


# -- setup wizard --

def cmd_init(_args):
    console.print(LOGO)
    console.print(Panel(
        "Choose your agent, choose the local model to improve, then rate answers Good or Bad. "
        "ReinforceClaw handles the background training.",
        title="ReinforceClaw", border_style="green"))

    state, data = 0, {}
    agent_choices = [
        ("OpenClaw", "Telegram, WhatsApp, Slack, Discord, and OpenClaw chat through your OpenClaw setup."),
        ("Codex", "Codex CLI/Desktop responses."),
        ("Claude Code", "Claude Code responses."),
    ]
    servers = [("Ollama", "Local Ollama model/tag."), ("LM Studio", "Local OpenAI-compatible server."),
               ("vLLM", "Local or remote vLLM server."), ("Other", "Manual adapter/server reload.")]
    feedback_choices = [
        ("Good/Bad prompt after each response", "Primary mode. OpenClaw sends a small Good/Bad poll or buttons after the AI answers."),
        ("Prompt plus reactions", "The prompt stays on. Held-message thumbs reactions also count when the platform forwards them."),
        ("/rl only", "Manual fallback. Type /rl good or /rl bad yourself."),
    ]
    while state < 5:
        if state == 0:
            chosen = _menu("Where will you rate responses?", agent_choices, multi=True,
                           selected=data.get("agent_indexes", []))
            data["agent_indexes"] = chosen
            data["agents"] = [["openclaw", "codex", "claude_code"][i] for i in chosen] or ["openclaw"]
            state += 1
        elif state == 1:
            companies = list(MODELS.keys()) + ["Other (HuggingFace ID)"]
            choice = _menu("Which local model should ReinforceClaw train?", companies,
                           default=data.get("company_index", 0), allow_back=True)
            if choice is WIZARD_BACK:
                state -= 1
                continue
            data["company_index"] = choice
            if choice < len(MODELS):
                company = list(MODELS.keys())[choice]
                mc = _menu(f"{company}: choose the model to improve", MODELS[company],
                           default=data.get("model_index", 0), allow_back=True)
                if mc is WIZARD_BACK:
                    continue
                data["model_index"] = mc
                data["model_name"] = MODELS[company][mc]
            else:
                model = Prompt.ask("HuggingFace model ID, or type 'back'").strip()
                if model.lower() == "back":
                    continue
                data["model_name"] = model
            state += 1
        elif state == 2:
            preset_choices = list(PRESETS.items()) + [("custom", "Set your own learning rate and steps.")]
            pc = _menu("How cautious should training be?", preset_choices,
                       default=data.get("preset_index", 1), allow_back=True)
            if pc is WIZARD_BACK:
                state -= 1
                continue
            data["preset_index"] = pc
            if pc <= 2:
                data.update({"preset_name": list(PRESETS.keys())[pc], "tuning_mode": "auto", "custom_overrides": {}})
            else:
                lr = Prompt.ask("Learning rate, or type 'back'", default=str(data.get("lr", "5e-6"))).strip()
                if lr.lower() == "back":
                    continue
                steps = Prompt.ask("Steps per round, or type 'back'", default=str(data.get("steps", 32))).strip()
                if steps.lower() == "back":
                    continue
                data.update({"preset_name": "balanced", "tuning_mode": "custom", "lr": lr, "steps": int(steps),
                             "custom_overrides": {"lr": float(lr), "steps": int(steps), "traj_clip": [0.996, 1.001]}})
            state += 1
        elif state == 3:
            sc = _menu("Where does the improved model run?", servers, default=data.get("server_index", 0), allow_back=True)
            if sc is WIZARD_BACK:
                state -= 1
                continue
            data["server_index"] = sc
            data["server"] = ["ollama", "lmstudio", "vllm", "other"][sc]
            data.pop("serve_model", None)
            data.pop("server_url", None)
            if data["server"] == "ollama":
                value = Prompt.ask("Ollama base model/tag, or type 'back'", default=_default_ollama_tag(data["model_name"])).strip()
                if value.lower() == "back":
                    continue
                data["serve_model"] = value
            elif data["server"] == "vllm":
                value = Prompt.ask("vLLM server URL, or type 'back'", default=data.get("server_url", "http://localhost:8000")).strip()
                if value.lower() == "back":
                    continue
                data["server_url"] = value
            state += 1
        elif state == 4:
            if "openclaw" not in data["agents"]:
                data["openclaw_feedback_mode"] = None
                state += 1
                continue
            fc = _menu("How should OpenClaw show the Good/Bad rating?", feedback_choices,
                       default=data.get("feedback_index", 0), allow_back=True)
            if fc is WIZARD_BACK:
                state -= 1
                continue
            data["feedback_index"] = fc
            data["openclaw_feedback_mode"] = ["slash_reactions_prompt", "slash_reactions_prompt", "slash_only"][fc]
            state += 1

    agents = data["agents"]
    model_name = data["model_name"]
    preset_name = data["preset_name"]
    server = data["server"]
    tuning_mode = data["tuning_mode"]
    custom_overrides = data.get("custom_overrides", {})
    serve_model = data.get("serve_model")
    server_url = data.get("server_url")
    extra = _backend_extra()
    _install_backend(extra)
    if extra:
        custom_overrides.setdefault("compute_backend", extra)

    cfg = {"model": model_name, "server": server, "preset": preset_name,
           "agents": agents, "panel_enabled": True,
           "tuning_mode": tuning_mode, **DEFAULTS, **custom_overrides}
    if serve_model:
        cfg["serve_model"] = serve_model
    if server_url:
        cfg["vllm_url"] = server_url
    if "openclaw" in agents:
        cfg["openclaw_secret"] = secrets.token_urlsafe(32)
        cfg["openclaw_feedback_mode"] = data.get("openclaw_feedback_mode") or "slash_reactions_prompt"
    resolved_cfg = _resolve_config(cfg)
    save_config(resolved_cfg)
    db.init()
    hook_results = _install_hooks(cfg)
    installed_agents = [name for name in agents if hook_results.get(name, False)]
    if installed_agents != resolved_cfg.get("agents"):
        resolved_cfg["agents"] = installed_agents
        save_config(resolved_cfg)
    if "openclaw" in agents and "openclaw" not in installed_agents:
        resolved_cfg.pop("openclaw_secret", None)
        save_config(resolved_cfg)
        OPENCLAW_BRIDGE_CONFIG.unlink(missing_ok=True)

    # set up training schedule
    from reinforceclaw import scheduler
    schedule = cfg.get("train_schedule", "auto")
    if cfg.get("training_enabled") is False:
        scheduler.uninstall()
    elif scheduler.install(schedule, cfg.get("schedule_window_minutes", 180)):
        if schedule not in ("manual", "auto"):
            console.print(f"[green]Training scheduled daily at {schedule}[/green]")
    elif schedule == "auto":
        detail = f" {escape(scheduler.LAST_ERROR)}" if getattr(scheduler, "LAST_ERROR", "") else ""
        console.print(f"[yellow]Could not remove an old system scheduler, but hook-triggered auto training remains enabled.{detail}[/yellow]")
    else:
        resolved_cfg["train_schedule"] = "manual"
        save_config(resolved_cfg)
        detail = f" {escape(scheduler.LAST_ERROR)}" if getattr(scheduler, "LAST_ERROR", "") else ""
        console.print(f"[yellow]Scheduler install failed. Training left in manual mode.{detail}[/yellow]")

    console.print("\n")
    console.print(Panel(
        "[bold red]Rate BOTH good AND bad responses.[/bold red]\n\n"
        "If you only rate bad, your model will get WORSE — it learns what to avoid\n"
        "but has no idea what you actually want. It needs both signals to improve.\n"
        "Bad only = broken model. Good only = weak model. Both = the goal.\n\n"
        "[dim]Your adapter only works on this exact model. Switching models = fresh start.[/dim]",
        title="Important", border_style="red"))
    console.print(Panel(
        f"Model: [bold]{model_name}[/bold] | Preset: [bold]{preset_name}[/bold] | Server: [bold]{server}[/bold]\n"
        f"Config: [dim]{CONFIG_PATH}[/dim]\n\n"
        f"1. Use your AI agent normally\n"
        f"2. Rate responses ([bold]/rl good[/bold], [bold]/rl bad[/bold], or ignore)\n"
        f"3. " + ("Ratings are saved, but local training is off until you enable it\n" if resolved_cfg.get("training_enabled") is False else f"Once you hit {resolved_cfg['batch_min']} ratings, training runs automatically in the background\n") +
        f"4. [bold]reinforceclaw status[/bold] to check progress | [bold]reinforceclaw history[/bold] to fix ratings",
        title="Setup complete", border_style="green"))


def _install_hooks(cfg):
    """Install hooks for each selected agent."""
    hook_dir = Path(__file__).parent / "hooks"
    results = {}

    installers = {
        "claude_code": lambda: (_install_claude_code_hooks(hook_dir), True)[1],
        "codex": lambda: (_install_codex_hooks(hook_dir), True)[1],
        "openclaw": lambda: _install_openclaw_plugin(cfg),
    }
    for name in cfg.get("agents", []):
        if name not in installers:
            continue
        try:
            results[name] = installers[name]()
        except Exception as exc:
            results[name] = False
            console.print(f"[yellow]{name} hook install failed: {escape(str(exc))}[/yellow]")
    return results


def _is_reinforceclaw_hook(entry, script_name):
    if isinstance(entry, dict) and entry.get("_reinforceclaw") is True:
        return True
    for hook in entry.get("hooks", []) if isinstance(entry, dict) else []:
        command = hook.get("command", "") if isinstance(hook, dict) else ""
        if "reinforceclaw" in command and script_name in command:
            return True
    return False


def _replace_reinforceclaw_hook(existing, entry, script_name):
    existing = existing if isinstance(existing, list) else []
    kept = [item for item in existing if not _is_reinforceclaw_hook(item, script_name)]
    kept.append(entry)
    return kept


def _install_json_hooks(path, script_name, command):
    cfg = _read_json(path)
    hooks = cfg.get("hooks") if isinstance(cfg.get("hooks"), dict) else {}
    for event, arg in (("Stop", "stop"), ("UserPromptSubmit", "prompt")):
        existing = hooks.get(event, [])
        entry = {"_reinforceclaw": True, "hooks": [{"type": "command", "command": f"{command} {arg}", "timeout": 30}]}
        hooks[event] = _replace_reinforceclaw_hook(existing, entry, script_name)
    cfg["hooks"] = hooks
    _write_json_atomic(path, cfg, backup=True)


def _install_claude_code_hooks(hook_dir):
    settings_path = Path.home() / ".claude" / "settings.json"
    script = str(hook_dir / "claude_code.py")
    _install_json_hooks(settings_path, "claude_code.py", f"{shlex.quote(sys.executable)} {shlex.quote(script)}")
    console.print(f"[green]Claude Code hooks installed:[/green] {settings_path}")


def _install_codex_hooks(hook_dir):
    """Install hooks into Codex CLI's hooks.json."""
    # codex hooks live at ~/.codex/hooks.json — same protocol as claude code
    hooks_path = Path.home() / ".codex" / "hooks.json"
    script = str(hook_dir / "codex.py")
    _install_json_hooks(hooks_path, "codex.py", f"{shlex.quote(sys.executable)} {shlex.quote(script)}")
    _enable_codex_hooks_feature()
    console.print(f"[green]Codex hooks installed:[/green] {hooks_path}")


def _enable_codex_hooks_feature():
    config_path = Path.home() / ".codex" / "config.toml"
    db.secure_private_dir(config_path.parent)
    text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    try:
        import tomlkit
        doc = tomlkit.parse(text or "")
        features = doc.setdefault("features", tomlkit.table())
        if features.get("codex_hooks") is True:
            return
        features["codex_hooks"] = True
        _write_text_atomic(config_path, tomlkit.dumps(doc), backup=True)
        return
    except ModuleNotFoundError:
        pass
    except Exception:
        console.print(f"[yellow]Could not edit Codex TOML with tomlkit: {config_path}[/yellow]")
        return
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = None
    if tomllib:
        try:
            parsed = tomllib.loads(text or "")
        except tomllib.TOMLDecodeError:
            console.print(f"[yellow]Could not edit invalid Codex TOML: {config_path}[/yellow]")
            return
        if parsed.get("features", {}).get("codex_hooks") is True:
            return
    lines = text.splitlines()
    in_features = False
    in_multiline = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        for marker in ('"""', "'''"):
            if stripped.count(marker) % 2:
                in_multiline = None if in_multiline == marker else marker
        if not in_multiline and re.match(r"^\[.*\]\s*(#.*)?$", stripped):
            if in_features:
                lines.insert(idx, "codex_hooks = true")
                _write_text_atomic(config_path, "\n".join(lines).rstrip() + "\n", backup=True)
                return
            in_features = re.match(r"^\[features\]\s*(#.*)?$", stripped) is not None
            continue
        if in_features and re.match(r"^codex_hooks\s*=", stripped):
            lines[idx] = "codex_hooks = true"
            _write_text_atomic(config_path, "\n".join(lines).rstrip() + "\n", backup=True)
            return
    if in_features:
        lines.append("codex_hooks = true")
        _write_text_atomic(config_path, "\n".join(lines).rstrip() + "\n", backup=True)
        return
    _write_text_atomic(config_path, text.rstrip() + ("\n\n" if text.strip() else "") + "[features]\ncodex_hooks = true\n", backup=True)


def _install_openclaw_plugin(cfg):
    """Install the OpenClaw plugin into the user's existing gateway."""
    plugin_dir = Path(__file__).parent / "openclaw_plugin"
    if not plugin_dir.exists():
        console.print("[yellow]Bundled OpenClaw plugin not found. Skipping.[/yellow]")
        return False
    secret = cfg.get("openclaw_secret")
    feedback_mode = cfg.get("openclaw_feedback_mode", "slash_reactions_prompt")
    host = "http://127.0.0.1:8420"
    import subprocess
    env = trainer._child_env()
    try:
        result = subprocess.run(
            ["openclaw", "plugins", "install", str(plugin_dir)],
            capture_output=True, text=True, timeout=30, env=env,
        )
        if result.returncode == 0:
            configured = True
            for path, value in (
                ("plugins.entries.reinforceclaw-feedback.config.reinforceclawHost", host),
                ("plugins.entries.reinforceclaw-feedback.config.reinforceclawSecret", secret),
                ("plugins.entries.reinforceclaw-feedback.config.reinforceclawFeedbackMode", feedback_mode),
            ):
                if not value:
                    continue
                configured = subprocess.run(
                    ["openclaw", "config", "set", path, value],
                    capture_output=True, text=True, timeout=15, env=env,
                ).returncode == 0 and configured
            enabled = subprocess.run(
                ["openclaw", "plugins", "enable", "reinforceclaw-feedback"],
                capture_output=True, text=True, timeout=15, env=env,
            ).returncode == 0
            from reinforceclaw import scheduler
            bridge_started = scheduler.install_openclaw_bridge()
            OPENCLAW_BRIDGE_CONFIG.unlink(missing_ok=True)
            if configured and enabled and bridge_started:
                extra = " Real thumbs/check reactions and Good/Bad prompts are enabled when OpenClaw forwards/supports them." if feedback_mode != "slash_only" else ""
                console.print(f"[green]OpenClaw plugin installed — use /rl good, /rl bad, /rl undo, or /rl status in any connected channel.{extra}[/green]")
                return True
            if configured and enabled and not bridge_started:
                detail = f" {escape(scheduler.LAST_ERROR)}" if getattr(scheduler, "LAST_ERROR", "") else ""
                console.print(f"[yellow]OpenClaw plugin installed, but the local ReinforceClaw bridge service could not start.{detail} Run 'python -m reinforceclaw.hooks.openclaw' until the service is fixed.[/yellow]")
                return False
            console.print("[yellow]OpenClaw plugin installed but could not be fully enabled/configured. Run reinforceclaw init again after checking OpenClaw.[/yellow]")
            OPENCLAW_BRIDGE_CONFIG.unlink(missing_ok=True)
            return False
        else:
            console.print(f"[yellow]OpenClaw plugin install failed: {result.stderr.strip()}[/yellow]")
        return False
    except FileNotFoundError:
        console.print("[yellow]openclaw not found. Install it first, then run reinforceclaw init again.[/yellow]")
        return False
    except subprocess.TimeoutExpired:
        console.print("[yellow]OpenClaw plugin install timed out.[/yellow]")
        return False


# -- rating commands --

def cmd_rate(_args, rating=None):
    # rate the last response
    cfg = _load_model_cfg()
    if not cfg:
        return
    conn = db.connect()
    try:
        pending = _pop_newest_pending("claude_code", "codex")
        if not pending:
            console.print(
                "[yellow]No captured response to rate. Use /rl good or /rl bad in your agent, "
                "or keep the panel on.[/yellow]"
            )
            return
        if not all(pending.get(key) for key in ("model", "prompt", "response")):
            console.print("[yellow]Captured response was incomplete and could not be rated.[/yellow]")
            return
        added = False
        try:
            db.add_feedback(
                conn, pending["model"], pending["prompt"], pending["response"], rating,
                context=pending.get("context"), source=pending.get("source", "cli"),
                event_id=pending.get("key"), rollout_context=pending.get("rollout_context"),
            )
            added = True
        except Exception:
            if not added:
                restore_pending(pending)
            raise
        label = "[green]good[/green]" if rating == 1 else "[red]bad[/red]"
        console.print(f"Rated: {label}")
        _maybe_train(cfg, conn)
    finally:
        conn.close()



def cmd_undo(_args):
    cfg = load_config()
    conn = db.connect()
    try:
        removed = db.remove_last(conn, model=cfg.get("model"))
        if removed:
            console.print(f"[yellow]Removed last rating ({'good' if removed['rating']==1 else 'bad'})[/yellow]")
        else:
            console.print("[dim]Nothing to undo.[/dim]")
    finally:
        conn.close()


def cmd_train(_args):
    cfg = _load_model_cfg()
    if not cfg:
        return
    if cfg.get("training_enabled") is False:
        console.print("[yellow]Training is off in this config. Run 'reinforceclaw init' again to enable local training.[/yellow]")
        return
    background = bool(getattr(_args, "background", False))
    cfg = {**copy.deepcopy(cfg), "_background": background}
    conn = db.connect()
    try:
        n = db.count_trainable_untrained(conn, model=cfg.get("model"))
        batch_min = cfg.get("batch_min", 8)
        resume = trainer._resume_state(conn, cfg)
        if background:
            TRAIN_RETRY_PATH.unlink(missing_ok=True)
        if n < batch_min and not resume:
            console.print(f"[yellow]Only {n} rated responses. Need at least {batch_min}.[/yellow]")
            return
        # only ask for confirmation when human is at the keyboard
        if sys.stdin.isatty() and not background and not Confirm.ask(f"{n} ratings ready. Train now?"):
            return
        console.print("[bold]Training...[/bold]" if not background else "[dim]Background training...[/dim]")
        result = trainer.train_result(cfg, conn)
        if result.get("status") == "trained":
            metrics = {k: v for k, v in result.items() if k != "status"}
            gate = (
                {"ok": False, "reason": "nonfinite_loss"}
                if not math.isfinite(float(metrics.get("avg_loss", float("nan"))))
                else trainer.publish_gate(cfg, metrics["path"])
            )
            if gate.get("ok"):
                _active, load_status = _activate_candidate(conn, cfg, metrics)
                if not _active:
                    console.print("[yellow]Trained candidate kept pending; server load or adapter lineage check failed.[/yellow]")
                    if load_status == "invalid":
                        console.print("[dim]Gate: stale parent adapter[/dim]")
                    elif load_status is False:
                        console.print("[dim]Server load failed; current active adapter was not changed.[/dim]")
                    return
                console.print(f"[green]Done![/green] Loss: {metrics['avg_loss']:.4f}, "
                               f"Batch: {metrics['batch_size']}, EMA: {metrics['ema_mean']:.3f}")
                if background:
                    db.record_background_event(conn, "success", datetime.now().hour)
                if load_status == "loaded_ollama":
                    console.print(f"[green]Prepared Ollama model {_ollama_reinforced_name(cfg)}. Point your agent at that tag.[/green]")
                elif load_status == "loaded":
                    console.print("[green]Prepared adapter/model loaded.[/green]")
                elif load_status == "failed":
                    console.print("[yellow]Could not load adapter automatically. Restart or repoint your server manually.[/yellow]")
                elif load_status == "manual":
                    console.print("[yellow]Adapter saved. Manual server reload may still be needed.[/yellow]")
            else:
                db.reject_adapter(conn, metrics["version"])
                console.print("[yellow]Trained candidate rejected by publish gate.[/yellow]")
                if gate.get("reason"):
                    console.print(f"[dim]Gate: {gate['reason']}[/dim]")
        else:
            reason = result.get("reason", "unknown")
            transient = trainer._TRANSIENT_RESOURCE_BLOCKS | {"outside_schedule_window", "missing_cuda_idle_telemetry", "memory_pressure", "insufficient_headroom", "resume_invalidated"}
            if background and reason == "resume_pending":
                from reinforceclaw.hooks._common import queue_training
                queue_training(delay_seconds=1)
            elif background and reason in transient:
                db.record_background_event(conn, "pressure", datetime.now().hour)
                from reinforceclaw.hooks._common import queue_training
                queue_training(delay_seconds=_next_retry_delay(conn))
            console.print(
                "[yellow]Skipped for now.[/yellow]"
                if background and reason not in ("ready", "backend_unavailable", "below_threshold")
                else "[yellow]Training failed.[/yellow]"
            )
            if background or reason != "ready":
                console.print(f"[dim]Reason: {reason}[/dim]")
            if result.get("detail"):
                console.print(f"[dim]{result['detail']}[/dim]")
    finally:
        conn.close()


def cmd_smoke(_args):
    cfg = _load_model_cfg()
    if not cfg:
        return
    if cfg.get("training_enabled") is False:
        console.print("[yellow]Feedback is connected, but local training is off.[/yellow]")
        return
    conn = db.connect()
    try:
        status = trainer.smoke_status(cfg, conn)
    finally:
        conn.close()
    title = "[green]Would train[/green]" if status["would_train"] else "[yellow]Would skip[/yellow]"
    keys = ("reason", "backend", "schedule", "trainable", "batch_min", "available_gb", "host_available_gb", "detail")
    console.print(_table(f"Background Smoke: {title}", [("Key", "bold"), "Value"], ((k, status[k]) for k in keys if status.get(k) is not None)))


def cmd_status(_args):
    cfg = load_config()
    conn = db.connect()
    try:
        counts = db.count(conn, model=cfg.get("model"))
        ema_mean, ema_count = db.get_ema(conn, model=cfg.get("model"))
        latest = db.latest_adapter(conn, model=cfg.get("model"))

        mp = cfg.get("model_profile")
        if not mp and cfg.get("training_enabled") is not False:
            _p = profile.detect(cfg.get("model", ""))
            mp = {"kind": _p.kind, "size_bucket": _p.size_bucket}
        mp = mp or {"kind": "feedback", "size_bucket": "training off"}
        console.print(_table("ReinforceClaw Status", [("", "bold"), ""], (
            ("Model", cfg.get("model", "not set")),
            ("Preset", cfg.get("preset", "balanced")),
            ("Profile", f"{mp.get('kind', '?')} / {mp.get('size_bucket', mp.get('scale', '?'))}"),
            ("Tuning", cfg.get("tuning_mode", "auto")),
            ("Training", "off" if cfg.get("training_enabled") is False else "on"),
            ("Server", cfg.get("server", "ollama")),
            ("Adapter", f"v{latest['version']}" if latest else "none (base model)"),
            ("Ratings", f"{counts['total']} total ({counts['good']}+ {counts['bad']}-)"),
            ("Untrained", db.count_trainable_untrained(conn, model=cfg.get("model"))),
            ("EMA", f"{ema_mean:.3f} ({ema_count} updates)"),
            ("Panel", "on" if cfg.get("panel_enabled", True) else "off"),
        )))
    finally:
        conn.close()


def cmd_rollback(_args):
    cfg = load_config()
    conn = db.connect()
    try:
        adapters = [a for a in db.list_adapters(conn, model=cfg.get("model")) if a["status"] not in {"rejected", "stale"}]
        if not adapters:
            console.print("[yellow]No usable adapters to roll back to.[/yellow]")
            return
        console.print(_table("Adapters", [("#", "dim"), "Status", "Created"], (
            (f"v{a['version']}", "[green]active[/green]" if a["status"] == "active" else f"[dim]{a['status']}[/dim]", a["created_at"])
            for a in adapters
        ), style="green"))
        pick = IntPrompt.ask("Roll back to version", default=adapters[0]["version"])
        prev, error = rollback_adapter(conn, cfg, pick)
        if prev:
            console.print(f"[green]Now on v{prev['version']}[/green]")
            if error:
                console.print(f"[yellow]{error}[/yellow]")
        elif error:
            console.print(f"[yellow]{error}[/yellow]")
    finally:
        conn.close()


def cmd_reset(_args):
    if not Confirm.ask("[red]Delete all ratings, adapters, and start fresh?[/red]"):
        return
    try:
        reset_state()
        console.print("[green]Reset. Clean slate.[/green]")
        console.print("[yellow]Restart or repoint your model server to stop serving any previously loaded adapter.[/yellow]")
    except RuntimeError as exc:
        console.print(f"[yellow]{exc}[/yellow]")


def _maybe_train(cfg, conn):
    if cfg.get("training_enabled") is False:
        return
    if cfg.get("train_schedule", "auto") != "auto":
        return
    if db.count_trainable_untrained(conn, model=cfg.get("model")) >= cfg.get("batch_min", 8):
        from reinforceclaw.hooks._common import queue_training
        console.print("[dim]Batch ready, training queued in background...[/dim]")
        queue_training()


def cmd_history(_args):
    """Show recent ratings. Use 'reinforceclaw rate <id> good|bad|delete' to edit one."""
    conn = db.connect()
    try:
        rows = db.recent(conn)
        if not rows:
            console.print("[dim]No ratings yet.[/dim]")
            return
        label = {"1": "[green]good[/green]", "-1": "[red]bad[/red]", "0": "[dim]unrated[/dim]"}
        console.print(_table("Recent Ratings", [("ID", "dim"), "Prompt", "Rating", "Source", "When"], (
            (r["id"], escape(r["prompt"][:47] + "..." if len(r["prompt"]) > 50 else r["prompt"]),
             label.get(str(r["rating"]), "?"), escape(str(r["source"])), r["created_at"])
            for r in rows
        ), style="green"))
        console.print("[dim]To change a rating: reinforceclaw rate <id> <good|bad|delete>[/dim]")
    finally:
        conn.close()


def cmd_rerate(_args):
    """Change or delete a specific rating: reinforceclaw rate 42 good|bad|delete"""
    rating = {"good": 1, "yes": 1, "bad": -1, "no": -1, "delete": 0, "ignore": 0}.get(_args.value)
    if rating is None:
        console.print("[red]Rating must be good, bad, delete, or ignore.[/red]")
        return
    conn = db.connect()
    try:
        changed = db.revise_feedback_rating(conn, _args.id, rating)
        if changed:
            label = "[green]good[/green]" if rating == 1 else "[red]bad[/red]" if rating == -1 else "[dim]deleted[/dim]"
            console.print(f"Rating #{_args.id} changed to {label}")
            cfg = _load_model_cfg()
            if cfg:
                _maybe_train(cfg, conn)
        else:
            console.print(f"[yellow]No rating found with ID {_args.id}.[/yellow]")
    finally:
        conn.close()


def cmd_prune(_args):
    cfg = _load_model_cfg()
    conn = db.connect()
    try:
        max_rows = _args.max_rows if _args.max_rows is not None else _keep_trained_rows(cfg or {})
        max_age_days = _args.max_age_days if _args.max_age_days is not None else int((cfg or {}).get("feedback_keep_days", 0))
        max_untrained = _args.max_untrained_rows if _args.max_untrained_rows is not None else int((cfg or {}).get("feedback_keep_untrained_rows", 100000))
        deleted = db.prune_feedback(conn, max_rows=max_rows, max_age_days=max_age_days, max_untrained_rows=max_untrained)
        console.print(f"[green]Pruned {deleted} old feedback rows.[/green]" if deleted else "[dim]Nothing to prune.[/dim]")
    finally:
        conn.close()


def cmd_schedule(_args):
    """Set training schedule: reinforceclaw schedule 03:00 / reinforceclaw schedule auto / reinforceclaw schedule manual"""
    from reinforceclaw import scheduler
    cfg = load_config()
    if hasattr(_args, 'time') and _args.time:
        val = _args.time
        try:
            if val not in ("auto", "manual"):
                scheduler._parse_time(val)
        except ValueError:
            console.print("[red]Use HH:MM format, 'auto', or 'manual'[/red]")
            return
        cfg["train_schedule"] = val
        if scheduler.install(val, cfg.get("schedule_window_minutes", 180)) or val == "auto":
            save_config(cfg)
            console.print(f"[green]Schedule set: {val}[/green]")
        else:
            detail = f" {escape(scheduler.LAST_ERROR)}" if getattr(scheduler, "LAST_ERROR", "") else ""
            console.print(f"[red]Could not install scheduler. Leaving existing schedule unchanged.{detail}[/red]")
    else:
        console.print(f"Current: {cfg.get('train_schedule', 'auto')}")
        console.print("[dim]reinforceclaw schedule 03:00 / auto / manual[/dim]")


COMMANDS = {
    "init": cmd_init, "good": lambda a: cmd_rate(a, 1), "yes": lambda a: cmd_rate(a, 1),
    "bad": lambda a: cmd_rate(a, -1), "no": lambda a: cmd_rate(a, -1),
    "undo": cmd_undo, "train": cmd_train, "status": cmd_status,
    "rollback": cmd_rollback, "reset": cmd_reset, "on": lambda a: _set_panel(True), "off": lambda a: _set_panel(False),
    "history": cmd_history, "schedule": cmd_schedule, "smoke": cmd_smoke,
    "prune": cmd_prune,
}


def main():
    parser = argparse.ArgumentParser(prog="reinforceclaw", description="Personal RL for AI agents")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")
    for name in COMMANDS:
        if name in ("rate", "schedule", "train", "prune"):
            continue  # these have custom args, added below
        sub.add_parser(name)

    train_p = sub.add_parser("train", help="Run training now")
    train_p.add_argument("--background", action="store_true")

    # reinforceclaw rate <id> <good|bad|delete>
    rate_p = sub.add_parser("rate", help="Change/delete a rating: reinforceclaw rate 42 good|bad|delete")
    rate_p.add_argument("id", type=int)
    rate_p.add_argument("value", choices=["good", "yes", "bad", "no", "delete", "ignore"])

    # reinforceclaw schedule <time>
    sched_p = sub.add_parser("schedule", help="Set training schedule")
    sched_p.add_argument("time", nargs="?")

    prune_p = sub.add_parser("prune", help="Prune old feedback rows")
    prune_p.add_argument("--max-rows", type=int)
    prune_p.add_argument("--max-untrained-rows", type=int)
    prune_p.add_argument("--max-age-days", type=int)

    args = parser.parse_args()
    if args.command == "rate":
        cmd_rerate(args)
        return 0
    elif args.command in COMMANDS:
        COMMANDS[args.command](args)
        return 0
    else:
        parser.print_help()
        return 2


if __name__ == "__main__":
    sys.exit(main())
