#!/usr/bin/env python3
"""Codex desktop app hook. Buttons after each response + prompt-command backups."""
# click left → green (good). click right → red (bad).
# click filled one again → unfills (undo). X closes the circles.
# circles reappear after every response. prompt hook also supports /good and /bad.

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from nudge import db
from nudge.hooks._common import load_config, maybe_train, read_stdin

SOURCE = "codex"
COMMANDS = {
    "/rl good": "good", "/rl bad": "bad", "/rl undo": "undo",
    "/rl train": "train", "/rl status": "status",
    "/rl rollback": "rollback", "/rl reset": "reset",
    "/rl on": "on", "/rl off": "off",
    "/good": "good", "/bad": "bad",
}


def _show_buttons(feedback_id, config):
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

    selected = [None]
    colors = {1: "#22c55e", -1: "#ef4444"}

    def _click(canvas, oval, other_canvas, other_oval, rating):
        if selected[0] == rating:
            canvas.itemconfig(oval, fill="")
            conn = db.connect()
            db.update_feedback_rating(conn, feedback_id, 0)
            conn.close()
            selected[0] = None
            return
        canvas.itemconfig(oval, fill=colors[rating])
        other_canvas.itemconfig(other_oval, fill="")
        conn = db.connect()
        db.update_feedback_rating(conn, feedback_id, rating)
        maybe_train(conn, config)
        conn.close()
        selected[0] = rating

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
    close = tk.Label(frame, text="✕", font=("Arial", 10), fg="gray", bg=bg, cursor="hand2")
    close.pack(side="left", padx=3)
    close.bind("<Button-1>", lambda e: root.destroy())

    c1.bind("<Button-1>", lambda e: _click(c1, o1, c2, o2, 1))
    c2.bind("<Button-1>", lambda e: _click(c2, o2, c1, o1, -1))

    root.mainloop()


def handle_stop():
    data = read_stdin()
    config = load_config()
    if not config.get("model"):
        return

    last_msg = data.get("last_assistant_message", "")
    if not last_msg:
        return

    conn = db.connect()
    fid = db.add_feedback(conn, config["model"], "(codex)", last_msg, 0, source=SOURCE)
    conn.close()
    if not config.get("panel_enabled", True):
        return

    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "panel", str(fid)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def handle_prompt():
    data = read_stdin()
    prompt_obj = data.get("prompt", {})
    prompt = prompt_obj.get("content", "").strip() if isinstance(prompt_obj, dict) else str(prompt_obj).strip()
    cmd = COMMANDS.get(" ".join(prompt.lower().split()))
    if cmd is None:
        sys.exit(0)

    config = load_config()
    conn = db.connect()
    if cmd in ("good", "bad"):
        if not config.get("model"):
            conn.close()
            print(json.dumps({"result": "block", "reason": "nudge not initialized"}))
            return
        rating = 1 if cmd == "good" else -1
        pending = db.latest_pending(conn, source=SOURCE)
        if pending:
            db.update_feedback_rating(conn, pending["id"], rating)
        else:
            last_msg = data.get("last_assistant_message", "(from codex)")
            db.add_feedback(conn, config["model"], "(from codex)", last_msg, rating, source=SOURCE)
        maybe_train(conn, config)
    elif cmd == "undo":
        db.remove_last(conn)
    elif cmd == "train":
        from nudge.hooks._common import queue_training
        queue_training()
    elif cmd == "status":
        pass
    elif cmd == "rollback":
        prev = db.rollback(conn)
        if prev:
            from nudge import trainer
            trainer.hot_swap(config.get("server", "ollama"), prev["path"], config.get("model", ""))
    elif cmd == "reset":
        from nudge.cli import reset_state
        conn.close()
        reset_state()
        print(json.dumps({"result": "block", "reason": f"nudge: /rl {cmd}"}))
        return
    elif cmd == "on":
        config["panel_enabled"] = True
        from nudge.cli import save_config
        save_config(config)
    elif cmd == "off":
        config["panel_enabled"] = False
        from nudge.cli import save_config
        save_config(config)

    conn.close()
    print(json.dumps({"result": "block", "reason": f"nudge: /rl {cmd}"}))


def handle_panel():
    if len(sys.argv) < 3:
        return
    _show_buttons(int(sys.argv[2]), load_config())


if __name__ == "__main__":
    {"stop": handle_stop, "prompt": handle_prompt, "panel": handle_panel}.get(
        sys.argv[1] if len(sys.argv) > 1 else "", lambda: None
    )()
