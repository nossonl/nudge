#!/usr/bin/env python3
"""Claude Code hooks. Stop hook + /rl command interceptor."""
# stop hook: fires after each assistant turn. shows panel, saves only rated turns.
# prompt hook: intercepts /rl commands before claude sees them.
# stdout goes back to claude code — only print the final JSON response.

import sys
import traceback
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from reinforceclaw import db
from reinforceclaw.hooks._common import (
    load_config, read_stdin, maybe_train, pending_context, pop_pending, save_pending,
    last_msg_from, training_context, handle_agent_command,
)

SOURCE = "claude_code"


def handle_stop():
    conn = None
    try:
        data = read_stdin()
        config = load_config()
        if not config.get("model"):
            return

        last_msg = data.get("last_assistant_message", "") or last_msg_from(data, "assistant")
        if not last_msg:
            return

        prompt = last_msg_from(data, "user")
        context = pending_context(data)
        rollout = training_context(data, prompt, last_msg)
        try:
            key = save_pending(SOURCE, config["model"], prompt, last_msg, context=context, rollout_context=rollout)
        except ValueError as exc:
            sys.stderr.write(f"reinforceclaw: pending skipped: {exc}\n")
            return
        if not config.get("panel_enabled", True):
            return
        from reinforceclaw.feedback import collect_rating
        rating = collect_rating()
        if isinstance(rating, int):
            conn = db.connect()
            db.add_feedback(conn, config["model"], prompt, last_msg, rating, context=context, source=SOURCE, event_id=key, rollout_context=rollout)
            pop_pending(SOURCE, key, context=context)
            maybe_train(conn, config)
        elif rating is None:
            pop_pending(SOURCE, key, context=context)
        elif rating == "unavailable":
            # Keep the pending turn so /rl good or /rl bad can rate it from non-TTY clients.
            pass
    except Exception as exc:
        sys.stderr.write(f"reinforceclaw hook error: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")
    finally:
        if conn is not None:
            conn.close()


def handle_prompt():
    handle_agent_command(SOURCE, read_stdin(), sys.stderr.write)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    {"stop": handle_stop, "prompt": handle_prompt}.get(sys.argv[1], lambda: sys.exit(1))()
