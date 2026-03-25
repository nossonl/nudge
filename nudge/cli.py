"""CLI. argparse. No Click, no Typer. Setup wizard + all commands."""
# two entry points: `nudge <cmd>` from terminal, `/rl <cmd>` from inside agents.
# both hit the same functions. wizard is `nudge init`.

import argparse
import json
import shutil
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
    "adapter_keep": 0,  # 0 = keep all adapters forever, set a number to auto-cleanup
}

MODELS = {
    "Qwen": [
        "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B",
        "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B",
        "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-122B-A10B", "Qwen/Qwen3.5-397B-A17B",
        "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen/QwQ-32B",
    ],
    "Meta": [
        "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    ],
    "Mistral": [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "mistralai/Mistral-Small-4-119B-2603",
        "mistralai/Mistral-Large-Instruct-2411",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Codestral-22B-v0.1",
        "mistralai/Devstral-Small-2505",
        "mistralai/Magistral-Small-2506",
    ],
    "Google": [
        "google/gemma-2-9b-it", "google/gemma-2-27b-it",
        "google/gemma-3-1b-it", "google/gemma-3-4b-it",
        "google/gemma-3-12b-it", "google/gemma-3-27b-it",
        "google/gemma-3n-E2B-it", "google/gemma-3n-E4B-it",
    ],
    "Microsoft": [
        "microsoft/Phi-4-mini-instruct", "microsoft/Phi-4-mini-reasoning",
        "microsoft/phi-4", "microsoft/Phi-4-reasoning",
        "microsoft/Phi-3.5-mini-instruct", "microsoft/Phi-3.5-MoE-instruct",
    ],
    "DeepSeek": [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    ],
    "GLM": [
        "zai-org/GLM-5", "zai-org/GLM-4.7-Flash", "zai-org/GLM-4.7",
        "zai-org/GLM-4.6", "zai-org/GLM-4.5", "zai-org/GLM-4.5-Air",
    ],
    "Moonshot": [
        "moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2-Instruct",
        "moonshotai/Kimi-Dev-72B", "moonshotai/Moonlight-16B-A3B-Instruct",
    ],
    "MiniMax": [
        "MiniMaxAI/MiniMax-M2.5", "MiniMaxAI/MiniMax-M2.1",
        "MiniMaxAI/MiniMax-M2", "MiniMaxAI/MiniMax-Text-01-hf",
    ],
    "Nvidia": [
        "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        "nvidia/OpenReasoning-Nemotron-32B",
    ],
    "Cohere": [
        "CohereLabs/c4ai-command-a-03-2025",
        "CohereLabs/c4ai-command-r7b-12-2024",
        "CohereLabs/aya-expanse-8b", "CohereLabs/aya-expanse-32b",
    ],
    "Yi": ["01-ai/Yi-1.5-9B-Chat", "01-ai/Yi-1.5-34B-Chat", "01-ai/Yi-Coder-9B-Chat"],
    "InternLM": ["internlm/internlm3-8b-instruct", "internlm/internlm2_5-7b-chat"],
    "Falcon": ["tiiuae/Falcon3-7B-Instruct", "tiiuae/Falcon3-10B-Instruct"],
    "AllenAI": ["allenai/Olmo-3-7B-Instruct", "allenai/Olmo-3.1-32B-Instruct"],
}

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
    console.print(Panel("Personal RL for your AI — rate responses, train a local LoRA.",
                        title="Welcome to Nudge", border_style="cyan"))

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
        console.print(f"  [cyan]{i}[/cyan]. {c}")

    choice = _clamp(IntPrompt.ask("Company", default=1), 1, len(companies))
    if choice <= len(MODELS):
        company = list(MODELS.keys())[choice - 1]
        models = MODELS[company]
        console.print(f"\n[bold]{company} models:[/bold]")
        for i, m in enumerate(models, 1):
            console.print(f"  [cyan]{i}[/cyan]. {m}")
        mc = _clamp(IntPrompt.ask("Model", default=1), 1, len(models))
        model_name = models[mc - 1]
    else:
        model_name = Prompt.ask("HuggingFace model ID")

    if any(kw in model_name.lower() for kw in ["moe", "mixture"]):
        console.print("\n[yellow]MoE detected. Works but uses more memory (LoRA on all experts).[/yellow]")

    # preset
    console.print("\n[bold]Training preset:[/bold]")
    for i, (name, info) in enumerate(PRESETS.items(), 1):
        console.print(f"  [cyan]{i}[/cyan]. [bold]{name}[/bold] — {info['desc']}")
    pc = _clamp(IntPrompt.ask("Preset", default=2), 1, 3)
    preset_name = list(PRESETS.keys())[pc - 1]
    preset = PRESETS[preset_name]

    # server
    console.print("\n[bold]Inference server:[/bold]")
    for i, s in enumerate(["Ollama", "vLLM", "Other"], 1):
        console.print(f"  [cyan]{i}[/cyan]. {s}")
    sc = _clamp(IntPrompt.ask("Server", default=1), 1, 3)
    server = ["ollama", "vllm", "other"][sc - 1]

    cfg = {"model": model_name, "server": server, "preset": preset_name,
           "agents": agents, "panel_enabled": True,
           "lr": preset["lr"], "traj_clip": preset["traj_clip"],
           "steps": preset["steps"], **DEFAULTS}
    save_config(cfg)
    db.connect().close()  # init tables
    _install_hooks(cfg)

    console.print("\n")
    console.print(Panel(
        "[bold]Rate BOTH good AND bad responses.[/bold]\n\n"
        "If you only rate bad, your model learns what to avoid but not what to do.\n"
        "Both signals are essential.\n\n"
        "[dim]Your adapter only works on this exact model. Switching models = fresh start.[/dim]",
        title="Important", border_style="yellow"))
    console.print(Panel(
        f"Model: [bold]{model_name}[/bold] | Preset: [bold]{preset_name}[/bold] | Server: [bold]{server}[/bold]\n"
        f"Config: [dim]{CONFIG_PATH}[/dim]\n\n"
        f"1. Use your AI agent normally\n"
        f"2. Rate responses (panel or /rl good / /rl bad)\n"
        f"3. After {cfg['batch_min']} ratings, training starts automatically\n"
        f"4. [bold]nudge status[/bold] to check progress",
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
    """Rate last response. Used by both cmd_good and cmd_bad."""
    cfg = _load_model_cfg()
    if not cfg:
        return
    conn = db.connect()
    pending = db.latest_pending(conn)
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
        console.print(f"[yellow]Not enough data. {_trainable_untrained(conn)} trainable untrained, "
                       f"need {cfg.get('batch_min', 16)}.[/yellow]")
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
    prev = db.rollback(conn)
    if prev:
        console.print(f"[green]Rolled back to v{prev['version']}[/green]")
        trainer.hot_swap(cfg.get("server", "ollama"), prev["path"], cfg.get("model", ""))
    else:
        console.print("[yellow]No previous adapter.[/yellow]")
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
    if _trainable_untrained(conn) >= cfg.get("batch_min", 16):
        console.print("[dim]Batch ready, training...[/dim]")
        metrics = trainer.train(cfg, conn)
        if metrics:
            console.print(f"[green]Trained![/green] Loss: {metrics['avg_loss']:.4f}")
            _swap_latest(cfg, conn)


COMMANDS = {
    "init": cmd_init, "good": lambda a: cmd_rate(a, 1), "bad": lambda a: cmd_rate(a, -1),
    "undo": cmd_undo, "train": cmd_train, "status": cmd_status,
    "rollback": cmd_rollback, "reset": cmd_reset, "on": cmd_on, "off": cmd_off,
}


def main():
    parser = argparse.ArgumentParser(prog="nudge", description="Personal RL for AI agents")
    sub = parser.add_subparsers(dest="command")
    for name in COMMANDS:
        sub.add_parser(name)
    args = parser.parse_args()
    handler = COMMANDS.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
