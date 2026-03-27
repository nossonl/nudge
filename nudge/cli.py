"""CLI. argparse. No Click, no Typer. Setup wizard + all commands."""
# two entry points: `nudge <cmd>` from terminal, `/rl <cmd>` from inside agents.
# both hit the same functions. wizard is `nudge init`.

import argparse
import json
import os
import shlex
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table

from . import db, trainer
from . import collect

console = Console()
CONFIG_PATH = Path.home() / ".nudge" / "config.json"
ADAPTER_ROOT = Path.home() / ".nudge" / "adapters"

PRESETS = {
    "careful":    {"lr": 6e-6, "traj_clip": [0.996, 1.001], "steps": 3,
                   "desc": "Small safest updates."},
    "balanced":   {"lr": 8e-6, "traj_clip": [0.996, 1.001], "steps": 4,
                   "desc": "Best current stable MIS-PO default."},
    "aggressive": {"lr": 9e-6, "traj_clip": [0.994, 1.003], "steps": 5,
                   "desc": "Faster drift. More overfit risk."},
}

DEFAULTS = {
    "loss_fn": "mis-po",
    "lora_target": "attention",
    "token_clip": [0.5, 2.0], "kl_coeff": 0.08, "lora_rank": 16,
    "grad_accum": 1, "grad_clip": 1.0, "batch_min": 24, "batch_size": 4,
    "replay_ratio": 0.25, "ema_decay": 0.99, "pos_weight": 1.0,
    "adv_clip": 2.0, "max_passes": 1.0,
    "pressure_retry_limit": 2, "pressure_cooldown_s": 3.0,
    "background_slice_steps": 2,
    "publish_gate_enabled": True, "publish_gate_max_drop": 0,
    "adapter_keep": 0,  # 0 = keep all adapters forever
    "train_schedule": "03:00",
    "schedule_window_minutes": 180,
}

from .models import MODELS  # model catalog lives in models.py

LOGO = r"""
[bold green]
    _   __          __
   / | / /__  ____/ /___ ____
  /  |/ / / / / __  / __ `/ _ \
 / /|  / /_/ / /_/ / /_/ /  __/
/_/ |_/\__,_/\__,_/\__, /\___/
                  /____/
[/bold green]
[dim]# ai designed this lol[/dim]"""


def load_config():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def save_config(cfg):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n")


def _clamp(val, lo, hi):
    return max(lo, min(val, hi))


def _load_model_cfg():
    cfg = load_config()
    if cfg.get("model"):
        return cfg
    console.print("[red]Run 'nudge init' first.[/red]")
    return None


def _trainable_untrained(conn):
    return db.count_trainable_untrained(conn)


def _server_base_url(server, cfg, prefix="server"):
    key = f"{prefix}_base_url"
    if cfg.get(key):
        return cfg[key]
    env_key = f"NUDGE_{prefix.upper()}_BASE_URL"
    if os.environ.get(env_key):
        return os.environ[env_key]
    if prefix == "server" and cfg.get("base_url"):
        return cfg["base_url"]
    defaults = {
        "ollama": "http://localhost:11434",
        "lmstudio": "http://localhost:1234/v1",
        "vllm": "http://localhost:8000/v1",
        "other": "",
        "openai": "http://localhost:8000/v1",
    }
    return defaults.get(server, "")


def _api_key(value, prefix):
    if value:
        return value
    return os.environ.get(f"NUDGE_{prefix.upper()}_API_KEY")


def _ollama_target_model(model_name, latest):
    return f"{model_name}-nudge" if latest else model_name


def _swap_latest(cfg, conn):
    latest = db.latest_adapter(conn)
    return latest and trainer.hot_swap(cfg.get("server", "ollama"), latest["path"], cfg["model"])


def _record_background_event(conn, kind):
    db.record_background_event(conn, kind, datetime.now().hour)


def _next_retry_delay(conn, base_delay=900):
    history = db.background_history(conn)
    if not history:
        return base_delay
    now = datetime.now()
    best = base_delay
    best_score = None
    for hours_ahead in range(0, 24):
        candidate = now + timedelta(hours=hours_ahead)
        delay = max(base_delay, int((candidate.replace(minute=0, second=0, microsecond=0) - now).total_seconds()))
        hour = candidate.hour
        stats = history.get(hour, {"pressure_count": 0, "success_count": 0})
        score = stats["pressure_count"] - (0.5 * stats["success_count"])
        if best_score is None or score < best_score:
            best_score, best = score, delay
    return best


def _set_panel(enabled):
    cfg = load_config()
    cfg["panel_enabled"] = enabled
    save_config(cfg)
    console.print("[green]Panel on.[/green]" if enabled else "[yellow]Panel off.[/yellow] Use /rl good or /rl bad.")


def reset_state():
    conn = db.connect()
    try:
        db.reset_all(conn)
    finally:
        conn.close()
    shutil.rmtree(ADAPTER_ROOT, ignore_errors=True)


# -- setup wizard --

def cmd_init(_args):
    console.print(LOGO)
    console.print(Panel("Hands-off reinforcement learning. Rate responses, your model learns the rest.",
                        title="Welcome to Nudge", border_style="green"))

    # agents
    agents = []
    console.print("\n[bold]Which agents do you use?[/bold]")
    for name in ["Claude Code", "Codex", "OpenClaw"]:
        if Confirm.ask(f"  {name}", default=(name == "Claude Code")):
            agents.append(name.lower().replace(" ", "_"))
    agents = agents or ["claude_code"]

    # model
    console.print("\n[bold]Pick your local model:[/bold]")
    companies = list(MODELS.keys()) + ["Other (HuggingFace ID)"]
    for i, c in enumerate(companies, 1):
        console.print(f"  [green]{i}[/green]. {c}")

    choice = _clamp(IntPrompt.ask("Company", default=1), 1, len(companies))
    if choice <= len(MODELS):
        company = list(MODELS.keys())[choice - 1]
        models = MODELS[company]
        console.print(f"\n[bold]{company} models:[/bold]")
        for i, m in enumerate(models, 1):
            console.print(f"  [green]{i}[/green]. {m}")
        mc = _clamp(IntPrompt.ask("Model", default=1), 1, len(models))
        model_name = models[mc - 1]
    else:
        model_name = Prompt.ask("HuggingFace model ID")

    # MoE models work fine, no warning needed — user picked what they want

    # preset
    console.print("\n[bold]Training preset:[/bold]")
    for i, (name, info) in enumerate(PRESETS.items(), 1):
        console.print(f"  [green]{i}[/green]. [bold]{name}[/bold] — {info['desc']}")
    console.print(f"  [green]4[/green]. [bold]custom[/bold] — set your own learning rate and steps")
    pc = _clamp(IntPrompt.ask("Preset", default=2), 1, 4)
    if pc <= 3:
        preset_name = list(PRESETS.keys())[pc - 1]
        preset = PRESETS[preset_name]
    else:
        preset_name = "custom"
        lr = float(Prompt.ask("Learning rate", default="3e-6"))
        steps = IntPrompt.ask("Steps per round", default=3)
        preset = {"lr": lr, "traj_clip": [0.996, 1.001], "steps": steps}

    # server
    servers = ["Ollama", "LM Studio", "vLLM", "Other"]
    console.print("\n[bold]Inference server:[/bold]")
    for i, s in enumerate(servers, 1):
        console.print(f"  [green]{i}[/green]. {s}")
    sc = _clamp(IntPrompt.ask("Server", default=1), 1, len(servers))
    server = ["ollama", "lmstudio", "vllm", "other"][sc - 1]

    cfg = {"model": model_name, "server": server, "preset": preset_name,
           "agents": agents, "panel_enabled": True,
           "lr": preset["lr"], "traj_clip": preset["traj_clip"],
           "steps": preset["steps"], **DEFAULTS}
    save_config(cfg)
    db.connect().close()  # init tables
    _install_hooks(cfg)

    # set up training schedule
    from nudge import scheduler
    schedule = cfg.get("train_schedule", "03:00")
    if schedule not in ("manual", "auto"):
        if scheduler.install(schedule, cfg.get("schedule_window_minutes", 180)):
            console.print(f"[green]Training scheduled daily at {schedule}[/green]")
        else:
            cfg["train_schedule"] = "manual"
            save_config(cfg)
            console.print("[yellow]Scheduler install failed. Training left in manual mode.[/yellow]")

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
        f"3. Once you hit {cfg['batch_min']} ratings, training becomes eligible and runs at {cfg.get('train_schedule', '03:00')} by default (or [bold]nudge train[/bold] anytime)\n"
        f"4. [bold]nudge status[/bold] to check progress | [bold]nudge history[/bold] to fix ratings",
        title="Setup complete", border_style="green"))


def _install_hooks(cfg):
    """Install hooks for each selected agent."""
    hook_dir = Path(__file__).parent / "hooks"

    if "claude_code" in cfg.get("agents", []):
        _install_claude_code_hooks(hook_dir)
    if "codex" in cfg.get("agents", []):
        _install_codex_hooks(hook_dir)
    if "openclaw" in cfg.get("agents", []):
        _install_openclaw_plugin()


def _install_claude_code_hooks(hook_dir):
    settings_path = Path.home() / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    hooks = settings.get("hooks", {})
    script = str(hook_dir / "claude_code.py")
    python = shlex.quote(sys.executable)
    command = f"{python} {shlex.quote(script)}"

    # Stop hook — fires after each assistant turn
    stop_hooks = hooks.get("Stop", [])
    entry = {"type": "command", "command": f"{command} stop"}
    if not any(script in str(h.get("command", "")) for h in stop_hooks):
        stop_hooks.append(entry)

    # UserPromptSubmit — intercepts /rl commands
    prompt_hooks = hooks.get("UserPromptSubmit", [])
    entry2 = {"type": "command", "command": f"{command} prompt"}
    if not any(script in str(h.get("command", "")) for h in prompt_hooks):
        prompt_hooks.append(entry2)

    hooks["Stop"] = stop_hooks
    hooks["UserPromptSubmit"] = prompt_hooks
    settings["hooks"] = hooks
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    console.print(f"[green]Claude Code hooks installed:[/green] {settings_path}")


def _install_codex_hooks(hook_dir):
    """Install hooks into Codex CLI's hooks.json."""
    # codex hooks live at ~/.codex/hooks.json — same protocol as claude code
    hooks_path = Path.home() / ".codex" / "hooks.json"
    hooks_cfg = json.loads(hooks_path.read_text()) if hooks_path.exists() else {}
    hooks = hooks_cfg.get("hooks", {})
    script = str(hook_dir / "codex.py")
    python = shlex.quote(sys.executable)

    stop_hooks = hooks.get("Stop", [])
    entry = {"hooks": [{"type": "command", "command": f"{python} {shlex.quote(script)} stop", "timeout": 30}]}
    if not any(script in json.dumps(h) for h in stop_hooks):
        stop_hooks.append(entry)

    prompt_hooks = hooks.get("UserPromptSubmit", [])
    entry2 = {"hooks": [{"type": "command", "command": f"{python} {shlex.quote(script)} prompt", "timeout": 30}]}
    if not any(script in json.dumps(h) for h in prompt_hooks):
        prompt_hooks.append(entry2)

    hooks["Stop"] = stop_hooks
    hooks["UserPromptSubmit"] = prompt_hooks
    hooks_cfg["hooks"] = hooks
    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    hooks_path.write_text(json.dumps(hooks_cfg, indent=2) + "\n")
    console.print(f"[green]Codex hooks installed:[/green] {hooks_path}")


def _install_openclaw_plugin():
    """Auto-install the nudge plugin into openclaw. Detects platforms automatically."""
    plugin_dir = Path(__file__).parent.parent / "openclaw-plugin"
    if not plugin_dir.exists():
        console.print("[yellow]openclaw-plugin/ not found. Skipping.[/yellow]")
        return
    # try to install — if openclaw isn't installed it'll just fail quietly
    import subprocess
    try:
        result = subprocess.run(
            ["openclaw", "plugins", "install", str(plugin_dir)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            console.print("[green]OpenClaw plugin installed — all platforms connected[/green]")
        else:
            console.print(f"[yellow]OpenClaw plugin install failed: {result.stderr.strip()}[/yellow]")
    except FileNotFoundError:
        console.print("[yellow]openclaw not found. Install it first, then run nudge init again.[/yellow]")
    except subprocess.TimeoutExpired:
        console.print("[yellow]OpenClaw plugin install timed out.[/yellow]")


# -- rating commands --

def cmd_rate(_args, rating=None):
    # rate the last response
    cfg = _load_model_cfg()
    if not cfg:
        return
    conn = db.connect()
    pending = db.latest_pending(conn)  # find any unrated response, regardless of source
    if pending:
        db.update_feedback_rating(conn, pending["id"], rating)
    else:
        # external CLI doesn't have the actual conversation — store what we have
        # the hook path stores real prompt/response text, this is the fallback
        db.add_feedback(conn, cfg["model"], "(cli)", "(cli)", rating, source="cli")
    label = "[green]good[/green]" if rating == 1 else "[red]bad[/red]"
    console.print(f"Rated: {label}")
    _maybe_train(cfg, conn)
    conn.close()



def cmd_undo(_args):
    conn = db.connect()
    removed = db.remove_last(conn)
    if removed:
        console.print(f"[yellow]Removed last rating ({'good' if removed['rating']==1 else 'bad'})[/yellow]")
    else:
        console.print("[dim]Nothing to undo.[/dim]")
    conn.close()


def cmd_train(_args):
    cfg = _load_model_cfg()
    if not cfg:
        return
    background = bool(getattr(_args, "background", False))
    cfg = {**cfg, "_background": background}
    compat = trainer.model_compatibility(cfg)
    if not compat["ok"]:
        console.print(f"[red]Model/backend incompatibility:[/red] {compat['reason']}")
        if compat.get("detail"):
            console.print(f"[dim]{compat['detail']}[/dim]")
        return
    conn = db.connect()
    n = _trainable_untrained(conn)
    batch_min = cfg.get("batch_min", 8)
    if n < batch_min:
        console.print(f"[yellow]Only {n} rated responses. Need at least {batch_min}.[/yellow]")
        conn.close()
        return
    # only ask for confirmation when human is at the keyboard
    if sys.stdin.isatty() and not background:
        msg = (f"[yellow]{n} ratings — might be weak with so few. Train anyway?[/yellow]"
               if n < batch_min * 2 else f"{n} ratings ready. Train now?")
        if not Confirm.ask(msg):
            conn.close()
            return
    console.print("[bold]Training...[/bold]" if not background else "[dim]Background training...[/dim]")
    result = trainer.train_result(cfg, conn)
    if result.get("status") == "trained":
        metrics = {k: v for k, v in result.items() if k != "status"}
        gate_ready = all(key in metrics for key in ("path", "version", "ema_mean", "ema_count", "feedback_ids"))
        gate = {"ok": True, "reason": "legacy_train_result"} if not gate_ready else trainer.publish_gate(cfg, metrics["path"])
        if gate.get("ok"):
            if gate_ready:
                db.activate_training_round(conn, metrics["version"], metrics["ema_mean"], metrics["ema_count"], metrics["feedback_ids"])
            console.print(f"[green]Done![/green] Loss: {metrics['avg_loss']:.4f}, "
                           f"Batch: {metrics['batch_size']}, EMA: {metrics['ema_mean']:.3f}")
            if background:
                _record_background_event(conn, "success")
            ok = _swap_latest(cfg, conn)
            if ok is True:
                console.print("[green]Adapter loaded.[/green]")
            elif ok is False:
                console.print("[yellow]Hot-swap failed. Restart server manually.[/yellow]")
            elif ok is None:
                console.print("[yellow]Adapter saved. Manual server reload may still be needed.[/yellow]")
        else:
            if gate_ready:
                db.reject_adapter(conn, metrics["version"])
            console.print("[yellow]Trained candidate rejected by publish gate.[/yellow]")
            if gate.get("reason"):
                console.print(f"[dim]Gate: {gate['reason']}[/dim]")
            if gate.get("base_score") is not None:
                console.print(f"[dim]Canary: {gate['candidate_score']}/{len(trainer._PUBLISH_CANARY)} vs base {gate['base_score']}/{len(trainer._PUBLISH_CANARY)}[/dim]")
    else:
        reason = result.get("reason", "unknown")
        if background and reason in {
            "resume_pending", "high_cpu_load", "memory_busy", "gpu_busy", "gpu_memory_busy",
            "host_memory_busy", "low_free_vram", "outside_schedule_window",
            "missing_cuda_idle_telemetry", "memory_pressure", "insufficient_headroom",
        }:
            _record_background_event(conn, "pressure")
        if background and reason in {
            "resume_pending",
            "high_cpu_load", "memory_busy", "gpu_busy", "gpu_memory_busy",
            "host_memory_busy", "low_free_vram", "outside_schedule_window",
            "missing_cuda_idle_telemetry", "memory_pressure", "insufficient_headroom",
        }:
            from nudge.hooks._common import queue_training
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
    conn.close()


def cmd_smoke(_args):
    cfg = _load_model_cfg()
    if not cfg:
        return
    conn = db.connect()
    try:
        status = trainer.smoke_status(cfg, conn)
    finally:
        conn.close()
    title = "[green]Would train[/green]" if status["would_train"] else "[yellow]Would skip[/yellow]"
    t = Table(title=f"Background Smoke: {title}", border_style="cyan")
    t.add_column("Key", style="bold")
    t.add_column("Value")
    for key in ("reason", "backend", "schedule", "trainable", "batch_min", "available_gb", "host_available_gb", "detail"):
        if key in status and status[key] is not None:
            t.add_row(key, str(status[key]))
    console.print(t)


def cmd_status(_args):
    cfg = load_config()
    conn = db.connect()
    counts = db.count(conn)
    ema_mean, ema_count = db.get_ema(conn)
    latest = db.latest_adapter(conn)

    t = Table(title="Nudge Status", border_style="cyan")
    t.add_column("", style="bold")
    t.add_column("")
    t.add_row("Model", cfg.get("model", "not set"))
    t.add_row("Preset", cfg.get("preset", "balanced"))
    t.add_row("Server", cfg.get("server", "ollama"))
    t.add_row("Adapter", f"v{latest['version']}" if latest else "none (base model)")
    t.add_row("Ratings", f"{counts['total']} total ({counts['good']}+ {counts['bad']}-)")
    t.add_row("Untrained", str(_trainable_untrained(conn)))
    t.add_row("EMA", f"{ema_mean:.3f} ({ema_count} updates)")
    t.add_row("Panel", "on" if cfg.get("panel_enabled", True) else "off")
    console.print(t)
    conn.close()


def cmd_rollback(_args):
    cfg = load_config()
    conn = db.connect()
    adapters = db.list_adapters(conn)
    if not adapters:
        console.print("[yellow]No adapters to roll back to.[/yellow]")
        conn.close()
        return
    t = Table(title="Adapters", border_style="green")
    t.add_column("#", style="dim")
    t.add_column("Status")
    t.add_column("Created")
    for a in adapters:
        status = "[green]active[/green]" if a["status"] == "active" else "[dim]rolled back[/dim]"
        t.add_row(f"v{a['version']}", status, a["created_at"])
    console.print(t)
    pick = IntPrompt.ask("Roll back to version", default=adapters[0]["version"])
    prev = db.rollback_to(conn, pick)
    if prev:
        console.print(f"[green]Now on v{prev['version']}[/green]")
        trainer.hot_swap(cfg.get("server", "ollama"), prev["path"], cfg.get("model", ""))
    conn.close()


def cmd_reset(_args):
    if not Confirm.ask("[red]Delete all ratings, adapters, and start fresh?[/red]"):
        return
    reset_state()
    console.print("[green]Reset. Clean slate.[/green]")


def cmd_on(_a):
    _set_panel(True)

def cmd_off(_a):
    _set_panel(False)


def _maybe_train(cfg, conn):
    if cfg.get("train_schedule", "03:00") != "auto":
        return
    if _trainable_untrained(conn) >= cfg.get("batch_min", 8):
        from nudge.hooks._common import queue_training
        console.print("[dim]Batch ready, training queued in background...[/dim]")
        queue_training()


def cmd_history(_args):
    """Show recent ratings. Use 'nudge rate <id> good/bad' to change one."""
    conn = db.connect()
    rows = db.recent(conn)
    if not rows:
        console.print("[dim]No ratings yet.[/dim]")
        conn.close()
        return
    t = Table(title="Recent Ratings", border_style="green")
    t.add_column("ID", style="dim")
    t.add_column("Prompt")
    t.add_column("Rating")
    t.add_column("Source")
    t.add_column("When")
    for r in rows:
        label = {"1": "[green]good[/green]", "-1": "[red]bad[/red]", "0": "[dim]unrated[/dim]"}
        t.add_row(str(r["id"]), r["prompt"][:50], label.get(str(r["rating"]), "?"),
                  r["source"], r["created_at"])
    console.print(t)
    console.print("[dim]To change a rating: nudge rate <id> <good|bad>[/dim]")
    conn.close()


def cmd_rerate(_args):
    """Change a specific rating: nudge rate 42 good"""
    if not hasattr(_args, 'id') or not hasattr(_args, 'value'):
        console.print("Usage: nudge rate <id> <good|bad>")
        return
    rating = 1 if _args.value == "good" else -1
    conn = db.connect()
    db.update_feedback_rating(conn, int(_args.id), rating)
    console.print(f"Rating #{_args.id} changed to {'[green]good[/green]' if rating == 1 else '[red]bad[/red]'}")
    conn.close()


def cmd_schedule(_args):
    """Set training schedule: nudge schedule 03:00 / nudge schedule auto / nudge schedule manual"""
    from nudge import scheduler
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
        if scheduler.install(val, cfg.get("schedule_window_minutes", 180)):
            save_config(cfg)
            console.print(f"[green]Schedule set: {val}[/green]")
        else:
            console.print("[red]Could not install scheduler. Leaving existing schedule unchanged.[/red]")
    else:
        console.print(f"Current: {cfg.get('train_schedule', '03:00')}")
        console.print("[dim]nudge schedule 03:00 / auto / manual[/dim]")


def cmd_promptgen(_args):
    topics = collect.normalize_topics(getattr(_args, "topics", None))
    count = max(1, int(getattr(_args, "count", 40)))
    server = getattr(_args, "server", "openai")
    base_url = getattr(_args, "base_url", "") or _server_base_url(server, {}, prefix="prompt")
    model = getattr(_args, "model", None)
    if not model:
        console.print("[red]Provide --model for prompt generation.[/red]")
        return
    if not base_url:
        console.print("[red]Provide --base-url for the prompt generator.[/red]")
        return
    console.print(f"[bold]Generating {count} prompts with {model}...[/bold]")
    prompts = collect.helper_generate_prompts(
        model=model,
        count=count,
        topics=topics,
        server=server,
        base_url=base_url,
        api_key=_api_key(getattr(_args, "api_key", None), "prompt"),
    )
    output = getattr(_args, "output", "") or str(Path.home() / ".nudge" / "prompts.jsonl")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    collect.save_prompts(output, prompts)
    console.print(f"[green]Saved {len(prompts)} prompts:[/green] {output}")


def cmd_collect(_args):
    cfg = _load_model_cfg()
    if not cfg:
        return
    conn = db.connect()
    latest = db.latest_adapter(conn)
    topics = collect.normalize_topics(getattr(_args, "topics", None))
    count = max(1, int(getattr(_args, "count", 40)))
    if getattr(_args, "file", None):
        prompts = collect.load_prompts(_args.file)
    elif getattr(_args, "prompt_model", None):
        prompt_server = getattr(_args, "prompt_server", "openai")
        prompt_base_url = getattr(_args, "prompt_base_url", "") or _server_base_url(prompt_server, cfg, prefix="prompt")
        if not prompt_base_url:
            console.print("[red]Provide --prompt-base-url for helper prompt generation.[/red]")
            conn.close()
            return
        console.print(f"[bold]Generating {count} prompts with {getattr(_args, 'prompt_model')}...[/bold]")
        prompts = collect.helper_generate_prompts(
            model=_args.prompt_model,
            count=count,
            topics=topics,
            server=prompt_server,
            base_url=prompt_base_url,
            api_key=_api_key(getattr(_args, "prompt_api_key", None), "prompt"),
        )
    else:
        prompts = collect.sample_prompts(count, topics)

    server = cfg.get("server", "ollama")
    base_url = _server_base_url(server, cfg)
    use_local_mlx = server == "ollama" and (not base_url or not collect.ollama_available(base_url))
    if server != "ollama" and server != "mlx" and not base_url:
        console.print("[red]Set server_base_url in config or use --file with an existing chat workflow.[/red]")
        conn.close()
        return
    if use_local_mlx:
        console.print("[dim]Ollama is not reachable. Using local MLX inference for collection.[/dim]")
        server = "mlx"
    turns = max(1, int(getattr(_args, "turns", 1)))
    if turns > 1 and not getattr(_args, "prompt_model", None):
        console.print("[yellow]Multi-turn needs --prompt-model for follow-up user turns. Falling back to 1 turn.[/yellow]")
        turns = 1
    judge_model = getattr(_args, "judge_model", None)
    judge_server = getattr(_args, "judge_server", "openai")
    judge_base_url = getattr(_args, "judge_base_url", "") or _server_base_url(judge_server, cfg, prefix="judge")
    judge_api_key = _api_key(getattr(_args, "judge_api_key", None), "judge")
    if judge_model and judge_server != "mlx" and not judge_base_url:
        console.print("[red]Provide --judge-base-url for the judge model.[/red]")
        conn.close()
        return
    target_model = _ollama_target_model(cfg["model"], latest) if cfg.get("server", "ollama") == "ollama" else cfg["model"]
    target_adapter = latest["path"] if latest and server == "mlx" else None

    good = bad = ignored = 0
    for i, item in enumerate(prompts, 1):
        seed_prompt = item["prompt"].strip()
        console.print(Panel(seed_prompt, title=f"Prompt {i}/{len(prompts)} [{item['topic']}]", border_style="cyan"))
        messages = [{"role": "user", "content": seed_prompt}]
        try:
            for turn in range(turns):
                response = collect.chat(
                    server=server,
                    model=target_model,
                    messages=messages,
                    base_url=base_url,
                    api_key=_api_key(cfg.get("server_api_key"), "server"),
                    timeout=180,
                    adapter_path=target_adapter,
                    lora_rank=cfg.get("lora_rank", 8),
                )
                messages.append({"role": "assistant", "content": response})
                if turn + 1 >= turns:
                    break
                followup = collect.continue_conversation(
                    model=_args.prompt_model,
                    transcript=messages,
                    server=prompt_server,
                    base_url=prompt_base_url,
                    api_key=_api_key(getattr(_args, "prompt_api_key", None), "prompt"),
                )
                messages.append({"role": "user", "content": followup})
        except Exception as exc:
            console.print(f"[red]Inference failed:[/red] {exc}")
            break
        train_prompt, train_response = collect.flatten_transcript(messages)
        console.print(Panel(train_response, title="Model Response", border_style="green"))

        if judge_model:
            try:
                decision = collect.judge_response(
                    model=judge_model,
                    prompt=train_prompt,
                    response=train_response,
                    server=judge_server,
                    base_url=judge_base_url,
                    api_key=judge_api_key,
                    transcript=messages,
                )
            except Exception as exc:
                console.print(f"[yellow]Judge failed, ignored sample:[/yellow] {exc}")
                ignored += 1
                continue
            console.print(f"[bold]Judge:[/bold] {decision}")
            action = {"good": "g", "bad": "b", "ignore": "i"}[decision]
        else:
            action = Prompt.ask("Rate", choices=["g", "b", "i", "q"], default="i")

        if action == "q":
            break
        if action == "i":
            ignored += 1
            continue
        rating = 1 if action == "g" else -1
        db.add_feedback(
            conn,
            cfg["model"],
            train_prompt,
            train_response,
            rating,
            context=json.dumps({
                "topic": item["topic"],
                "messages": messages[:-1],
                "transcript": messages,
                "served_model": target_model,
                "adapter_version": latest["version"] if latest else None,
            }),
            source="collect",
        )
        if rating == 1:
            good += 1
        else:
            bad += 1

    console.print(f"[green]Collected[/green] {good}+ {bad}- | [dim]{ignored} ignored[/dim]")
    if server == "mlx":
        collect._clear_local_mlx()
    _maybe_train(cfg, conn)
    conn.close()


def cmd_loop(_args):
    cmd_collect(_args)


COMMANDS = {
    "init": cmd_init, "good": lambda a: cmd_rate(a, 1), "bad": lambda a: cmd_rate(a, -1),
    "undo": cmd_undo, "train": cmd_train, "status": cmd_status,
    "rollback": cmd_rollback, "reset": cmd_reset, "on": cmd_on, "off": cmd_off,
    "history": cmd_history, "schedule": cmd_schedule, "smoke": cmd_smoke,
    "promptgen": cmd_promptgen, "collect": cmd_collect, "loop": cmd_loop,
}


def main():
    parser = argparse.ArgumentParser(prog="nudge", description="Personal RL for AI agents")
    sub = parser.add_subparsers(dest="command")
    for name in COMMANDS:
        if name in ("rate", "schedule", "train", "promptgen", "collect", "loop"):
            continue  # these have custom args, added below
        sub.add_parser(name)

    train_p = sub.add_parser("train", help="Run training now")
    train_p.add_argument("--background", action="store_true")

    # nudge rate <id> <good|bad>
    rate_p = sub.add_parser("rate", help="Change a rating: nudge rate 42 good")
    rate_p.add_argument("id")
    rate_p.add_argument("value", choices=["good", "bad"])

    # nudge schedule <time>
    sched_p = sub.add_parser("schedule", help="Set training schedule")
    sched_p.add_argument("time", nargs="?")

    promptgen_p = sub.add_parser("promptgen", help="Generate prompts with a fast helper model")
    promptgen_p.add_argument("--model")
    promptgen_p.add_argument("--server", default="openai")
    promptgen_p.add_argument("--base-url")
    promptgen_p.add_argument("--api-key")
    promptgen_p.add_argument("--topics", default="code,math,instructions,personality")
    promptgen_p.add_argument("--count", type=int, default=40)
    promptgen_p.add_argument("--output")

    collect_p = sub.add_parser("collect", help="Run real inference and rate good/bad/ignore")
    collect_p.add_argument("--file")
    collect_p.add_argument("--prompt-model")
    collect_p.add_argument("--prompt-server", default="openai")
    collect_p.add_argument("--prompt-base-url")
    collect_p.add_argument("--prompt-api-key")
    collect_p.add_argument("--judge-model")
    collect_p.add_argument("--judge-server", default="openai")
    collect_p.add_argument("--judge-base-url")
    collect_p.add_argument("--judge-api-key")
    collect_p.add_argument("--topics", default="code,math,instructions,personality")
    collect_p.add_argument("--count", type=int, default=40)
    collect_p.add_argument("--turns", type=int, default=1)

    loop_p = sub.add_parser("loop", help="Alias for collect")
    loop_p.add_argument("--file")
    loop_p.add_argument("--prompt-model")
    loop_p.add_argument("--prompt-server", default="openai")
    loop_p.add_argument("--prompt-base-url")
    loop_p.add_argument("--prompt-api-key")
    loop_p.add_argument("--judge-model")
    loop_p.add_argument("--judge-server", default="openai")
    loop_p.add_argument("--judge-base-url")
    loop_p.add_argument("--judge-api-key")
    loop_p.add_argument("--topics", default="code,math,instructions,personality")
    loop_p.add_argument("--count", type=int, default=40)
    loop_p.add_argument("--turns", type=int, default=1)

    args = parser.parse_args()
    if args.command == "rate":
        cmd_rerate(args)
    elif args.command in COMMANDS:
        COMMANDS[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
