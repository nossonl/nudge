#!/usr/bin/env python3
"""Codex desktop app hook. Buttons after each response + prompt-command backups."""
# click left → green (good). click right → red (bad).
# X closes the circles.
# circles reappear after every response. prompt hook also supports /rl good or /rl bad.

import os
import atexit
import subprocess
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from reinforceclaw import db
from reinforceclaw.hooks._common import (
    load_config, maybe_train, pending_context, pop_pending, read_stdin, save_pending,
    last_msg_from, training_context, handle_agent_command, restore_pending,
)

SOURCE = "codex"
PANEL_LOG_PATH = Path.home() / ".reinforceclaw" / "panel.log"
PANEL_LOG_MAX_BYTES = 2 * 1024 * 1024
_PANEL_LOG = None


def _panel_log_file():
    global _PANEL_LOG
    db.secure_private_dir(PANEL_LOG_PATH.parent)
    if _PANEL_LOG is None or _PANEL_LOG.closed:
        try:
            if PANEL_LOG_PATH.exists() and not PANEL_LOG_PATH.is_symlink() and PANEL_LOG_PATH.stat().st_size > PANEL_LOG_MAX_BYTES:
                PANEL_LOG_PATH.replace(PANEL_LOG_PATH.with_suffix(".log.1"))
        except OSError:
            pass
        fd = os.open(PANEL_LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o600)
        _PANEL_LOG = os.fdopen(fd, "a", encoding="utf-8")
        atexit.register(_PANEL_LOG.close)
    return _PANEL_LOG


def _panel_log(message):
    log = _panel_log_file()
    log.write(message.rstrip() + "\n")
    log.flush()


def _show_buttons(key, config):
    import tkinter as tk

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    try:
        root.attributes("-transparent", True)
        root.configure(bg="systemTransparent")
    except tk.TclError:
        root.attributes("-alpha", 0.95)
        root.configure(bg="#2d2d2d")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"100x32+{sw // 2 - 50}+{sh - 70}")

    colors = {1: "#22c55e", -1: "#ef4444"}
    consumed = False

    def _click(canvas, oval, other_canvas, other_oval, rating):
        nonlocal consumed
        if consumed:
            return
        consumed = True
        pending = pop_pending(SOURCE, key)
        if not pending:
            _panel_log(f"pending {key} already consumed")
            root.destroy()
            return
        canvas.itemconfig(oval, fill=colors[rating])
        other_canvas.itemconfig(other_oval, fill="")
        conn = db.connect()
        try:
            added = False
            try:
                db.add_feedback(
                    conn, pending["model"], pending["prompt"], pending["response"], rating,
                    context=pending.get("context"), source=SOURCE, event_id=pending.get("key"),
                    rollout_context=pending.get("rollout_context"),
                )
                added = True
                maybe_train(conn, config)
            except Exception:
                if not added:
                    restore_pending(pending)
                raise
        finally:
            conn.close()
        root.destroy()

    bg = root.cget("bg")
    frame = tk.Frame(root, bg=bg)
    frame.pack(expand=True)

    c1 = tk.Canvas(frame, width=24, height=24, bg=bg, highlightthickness=0, cursor="hand2")
    o1 = c1.create_oval(2, 2, 22, 22, fill="", outline="black", width=2)
    c1.pack(side="left", padx=3)

    c2 = tk.Canvas(frame, width=24, height=24, bg=bg, highlightthickness=0, cursor="hand2")
    o2 = c2.create_oval(2, 2, 22, 22, fill="", outline="black", width=2)
    c2.pack(side="left", padx=3)

    # X button to dismiss
    def _dismiss(*, consume=True):
        nonlocal consumed
        if consume and not consumed:
            consumed = True
            pop_pending(SOURCE, key)
        root.destroy()

    close = tk.Label(frame, text="✕", font=("Arial", 10), fg="gray", bg=bg, cursor="hand2")
    close.pack(side="left", padx=3)
    close.bind("<Button-1>", lambda e: _dismiss())

    c1.bind("<Button-1>", lambda e: _click(c1, o1, c2, o2, 1))
    c2.bind("<Button-1>", lambda e: _click(c2, o2, c1, o1, -1))

    root.after(300000, lambda: _dismiss(consume=False))
    root.mainloop()


def handle_stop():
    data = read_stdin()
    config = load_config()
    if not config.get("model"):
        return

    last_msg = data.get("last_assistant_message", "") or last_msg_from(data, "assistant")
    if not last_msg:
        return

    prompt = last_msg_from(data, "user") or "(codex)"
    try:
        key = save_pending(
            SOURCE, config["model"], prompt, last_msg,
            context=pending_context(data), rollout_context=training_context(data, prompt, last_msg),
        )
    except ValueError as exc:
        _panel_log(f"pending skipped: {exc}")
        return
    if not config.get("panel_enabled", True):
        return

    log = _panel_log_file()
    try:
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "panel", key],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log,
            start_new_session=True,
        )
    except Exception as exc:
        log.write(f"panel spawn error: {type(exc).__name__}: {exc}\n")
        log.flush()


def handle_prompt():
    handle_agent_command(SOURCE, read_stdin())


def handle_panel():
    if len(sys.argv) < 3:
        return
    key = sys.argv[2]
    try:
        _show_buttons(key, load_config())
    except Exception as exc:
        _panel_log(f"panel error: {type(exc).__name__}: {exc}")
        return


if __name__ == "__main__":
    {"stop": handle_stop, "prompt": handle_prompt, "panel": handle_panel}.get(
        sys.argv[1] if len(sys.argv) > 1 else "", lambda: None
    )()
