"""CLI. argparse. No Click, no Typer. Setup wizard + all commands."""
# two entry points: `nudge <cmd>` from terminal, `/rl <cmd>` from inside agents.
# both hit the same functions. wizard is `nudge init`.

import argparse
import json
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table

from . import db, trainer

console = Console()
CONFIG_PATH = Path.home() / ".nudge" / "config.json"
ADAPTER_ROOT = Path.home() / ".nudge" / "adapters"

PRESETS = {
    "careful":    {"lr": 2e-6, "traj_clip": [0.996, 1.001], "steps": 16,
                   "desc": "Tiny changes. ~50 ratings to notice."},
    "balanced":   {"lr": 4e-6, "traj_clip": [0.992, 1.002], "steps": 8,
                   "desc": "Good default. ~25 ratings to see results."},
    "aggressive": {"lr": 1e-5, "traj_clip": [0.98, 1.02], "steps": 16,
                   "desc": "Fast. ~16 ratings. Higher risk."},
}

DEFAULTS = {
    "token_clip": [0.5, 2.0], "kl_coeff": 0.001, "lora_rank": 16,
    "grad_accum": 4, "grad_clip": 1.0, "batch_min": 16,
    "replay_ratio": 0.5, "ema_decay": 0.99, "pos_weight": 1.2,
    "adapter_keep": 0,  # 0 = keep all adapters forever
    "train_schedule": "03:00",  # default: train at 3am when batch is ready
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


def _swap_latest(cfg, conn):
    latest = db.latest_adapter(conn)
    return latest and trainer.hot_swap(cfg.get("server", "ollama"), latest["path"], cfg["model"])


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
        lr = float(Prompt.ask("Learning rate", default="4e-6"))
        steps = IntPrompt.ask("Steps per round", default=8)
        preset = {"lr": lr, "traj_clip": [0.992, 1.002], "steps": steps}

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
        scheduler.install(schedule)
        console.print(f"[green]Training scheduled daily at {schedule}[/green]")

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
        f"2. Rate responses (panel or /rl good / /rl bad)\n"
        f"3. Once you hit {cfg['batch_min']} ratings, training runs at {cfg.get('train_schedule', '03:00')} (or [bold]nudge train[/bold] anytime)\n"
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

    # Stop hook — fires after each assistant turn
    stop_hooks = hooks.get("Stop", [])
    entry = {"type": "command", "command": f"python3 {script} stop"}
    if not any(script in str(h.get("command", "")) for h in stop_hooks):
        stop_hooks.append(entry)

    # UserPromptSubmit — intercepts /rl commands
    prompt_hooks = hooks.get("UserPromptSubmit", [])
    entry2 = {"type": "command", "command": f"python3 {script} prompt"}
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

    stop_hooks = hooks.get("Stop", [])
    entry = {"hooks": [{"type": "command", "command": f"python3 {script} stop", "timeout": 30}]}
    if not any(script in json.dumps(h) for h in stop_hooks):
        stop_hooks.append(entry)

    hooks["Stop"] = stop_hooks
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
    conn = db.connect()
    n = _trainable_untrained(conn)
    batch_min = cfg.get("batch_min", 16)
    if n < batch_min:
        console.print(f"[yellow]Only {n} rated responses. Need at least {batch_min}.[/yellow]")
        conn.close()
        return
    # only ask for confirmation when human is at the keyboard
    if sys.stdin.isatty():
        msg = (f"[yellow]{n} ratings — might be weak with so few. Train anyway?[/yellow]"
               if n < batch_min * 2 else f"{n} ratings ready. Train now?")
        if not Confirm.ask(msg):
            conn.close()
            return
    console.print("[bold]Training...[/bold]")
    metrics = trainer.train(cfg, conn)
    if metrics:
        console.print(f"[green]Done![/green] Loss: {metrics['avg_loss']:.4f}, "
                       f"Batch: {metrics['batch_size']}, EMA: {metrics['ema_mean']:.3f}")
        ok = _swap_latest(cfg, conn)
        if ok is not None:
            console.print("[green]Adapter loaded.[/green]" if ok
                          else "[yellow]Hot-swap failed. Restart server manually.[/yellow]")
    else:
        console.print("[yellow]Training failed.[/yellow]")
    conn.close()


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
    # only auto-train if schedule is "auto" — otherwise the scheduler handles it
    if cfg.get("train_schedule", "03:00") != "auto":
        return
    if _trainable_untrained(conn) >= cfg.get("batch_min", 16):
        console.print("[dim]Batch ready, training...[/dim]")
        metrics = trainer.train(cfg, conn)
        if metrics:
            console.print(f"[green]Trained![/green] Loss: {metrics['avg_loss']:.4f}")
            _swap_latest(cfg, conn)


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
        if val not in ("auto", "manual") and ":" not in val:
            console.print("[red]Use HH:MM format, 'auto', or 'manual'[/red]")
            return
        cfg["train_schedule"] = val
        save_config(cfg)
        scheduler.install(val)
        console.print(f"[green]Schedule set: {val}[/green]")
    else:
        console.print(f"Current: {cfg.get('train_schedule', '03:00')}")
        console.print("[dim]nudge schedule 03:00 / auto / manual[/dim]")


COMMANDS = {
    "init": cmd_init, "good": lambda a: cmd_rate(a, 1), "bad": lambda a: cmd_rate(a, -1),
    "undo": cmd_undo, "train": cmd_train, "status": cmd_status,
    "rollback": cmd_rollback, "reset": cmd_reset, "on": cmd_on, "off": cmd_off,
    "history": cmd_history, "schedule": cmd_schedule,
}


def main():
    parser = argparse.ArgumentParser(prog="nudge", description="Personal RL for AI agents")
    sub = parser.add_subparsers(dest="command")
    for name in COMMANDS:
        if name in ("rate", "schedule"):
            continue  # these have custom args, added below
        sub.add_parser(name)

    # nudge rate <id> <good|bad>
    rate_p = sub.add_parser("rate", help="Change a rating: nudge rate 42 good")
    rate_p.add_argument("id")
    rate_p.add_argument("value", choices=["good", "bad"])

    # nudge schedule <time>
    sched_p = sub.add_parser("schedule", help="Set training schedule")
    sched_p.add_argument("time", nargs="?")

    args = parser.parse_args()
    if args.command == "rate":
        cmd_rerate(args)
    elif args.command in COMMANDS:
        COMMANDS[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
