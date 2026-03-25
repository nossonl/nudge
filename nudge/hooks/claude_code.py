#!/usr/bin/env python3
"""Claude Code hooks. Stop hook + /rl command interceptor."""
# stop hook: fires after each assistant turn. records turn, shows panel, queues training.
# prompt hook: intercepts /rl commands before claude sees them.
# stdout goes back to claude code — only print the final JSON response.

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from nudge import db
from nudge.hooks._common import load_config, read_stdin, maybe_train

SOURCE = "claude_code"


def _last_msg_from(data, role):
    """Pull the last message of a given role from hook data."""
    for m in reversed(data.get("messages", [])):
        if m.get("role") == role:
            return m.get("content", "")
    return ""


def handle_stop():
    data = read_stdin()
    config = load_config()
    if not config.get("model"):
        return

    last_msg = data.get("last_assistant_message", "") or _last_msg_from(data, "assistant")
    if not last_msg:
        return

    prompt = _last_msg_from(data, "user")
    conn = db.connect()
    fid = db.add_feedback(conn, config["model"], prompt, last_msg, 0, source=SOURCE)

    if config.get("panel_enabled", True):
        from nudge.feedback import collect_rating
        rating = collect_rating()
        if isinstance(rating, int):
            db.update_feedback_rating(conn, fid, rating)
            maybe_train(conn, config)

    conn.close()


def handle_prompt():
    data = read_stdin()

    # claude code sends prompt as {"content": "...", "type": "user"}
    prompt_obj = data.get("prompt", {})
    prompt = prompt_obj.get("content", "").strip() if isinstance(prompt_obj, dict) else str(prompt_obj).strip()

    cmds = {
        "/rl good": "good", "/rl bad": "bad", "/rl undo": "undo",
        "/rl train": "train", "/rl status": "status",
        "/rl rollback": "rollback", "/rl reset": "reset",
        "/rl on": "on", "/rl off": "off",
        "/good": "good", "/bad": "bad",
    }

    cmd = next((a for p, a in cmds.items() if prompt.lower().startswith(p)), None)
    if cmd is None:
        sys.exit(0)

    config = load_config()
    conn = db.connect()

    if cmd in ("good", "bad"):
        if not config.get("model"):
            sys.stderr.write("Run nudge init first.\n")
            conn.close()
            print(json.dumps({"result": "block", "reason": "nudge not initialized"}))
            return
        rating = 1 if cmd == "good" else -1
        pending = db.latest_pending(conn, source=SOURCE)
        if pending:
            db.update_feedback_rating(conn, pending["id"], rating)
        else:
            last_msg = data.get("last_assistant_message", "(from chat)")
            db.add_feedback(conn, config["model"], "(from chat)", last_msg, rating, source=SOURCE)
        label = "\033[32mgood\033[0m" if rating == 1 else "\033[31mbad\033[0m"
        sys.stderr.write(f"Rated: {label}\n")
        maybe_train(conn, config)
    elif cmd == "undo":
        r = db.remove_last(conn)
        sys.stderr.write("\033[33mRemoved last rating\033[0m\n" if r else "Nothing to undo.\n")
    elif cmd == "train":
        from nudge.hooks._common import queue_training
        queue_training()
        sys.stderr.write("Training started in background.\n")
    elif cmd == "status":
        counts = db.count(conn)
        ema, _ = db.get_ema(conn)
        a = db.latest_adapter(conn)
        sys.stderr.write(
            f"Adapter: {'v'+str(a['version']) if a else 'none'} | "
            f"Ratings: {counts['total']} ({counts['good']}+ {counts['bad']}-) | "
            f"Untrained: {db.count_trainable_untrained(conn)} | EMA: {ema:.3f}\n")
    elif cmd == "rollback":
        prev = db.rollback(conn)
        if prev:
            from nudge import trainer
            trainer.hot_swap(config.get("server", "ollama"), prev["path"], config.get("model", ""))
        sys.stderr.write(f"\033[32mRolled back to v{prev['version']}\033[0m\n" if prev
                         else "No previous adapter.\n")
    elif cmd == "reset":
        from nudge.cli import reset_state
        conn.close()
        reset_state()
        sys.stderr.write("\033[32mReset complete.\033[0m\n")
        print(json.dumps({"result": "block", "reason": f"nudge: /rl {cmd}"}))
        return
    elif cmd == "on":
        config["panel_enabled"] = True
        from nudge.cli import save_config
        save_config(config)
        sys.stderr.write("\033[32mPanel on.\033[0m\n")
    elif cmd == "off":
        config["panel_enabled"] = False
        from nudge.cli import save_config
        save_config(config)
        sys.stderr.write("\033[33mPanel off.\033[0m\n")

    conn.close()
    print(json.dumps({"result": "block", "reason": f"nudge: /rl {cmd}"}))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    {"stop": handle_stop, "prompt": handle_prompt}.get(sys.argv[1], lambda: sys.exit(1))()
