"""Shared helpers for all hooks. Config, stdin, training, rating."""

import json
import os
import fcntl
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from nudge import db

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
TRAIN_LOG_PATH = Path.home() / ".nudge" / "train.log"
TRAIN_LOCK_PATH = Path.home() / ".nudge" / "train.lock"
TRAIN_RETRY_PATH = Path.home() / ".nudge" / "train.retry"


def load_config():
    p = Path.home() / ".nudge" / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def read_stdin():
    try:
        return json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        return {}


def _train_running() -> bool:
    TRAIN_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(TRAIN_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def _base_env():
    TRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT_STR}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else PROJECT_ROOT_STR
    )
    return env


def _spawn_train(argv):
    with TRAIN_LOG_PATH.open("a", encoding="utf-8") as log:
        subprocess.Popen(
            argv,
            cwd=PROJECT_ROOT,
            env=_base_env(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def queue_training(delay_seconds=0):
    """Spawn training in background — never block the hook."""
    if _train_running():
        return
    if delay_seconds > 0:
        try:
            due = float(TRAIN_RETRY_PATH.read_text())
            if due > time.time():
                return
        except Exception:
            pass
        TRAIN_RETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        TRAIN_RETRY_PATH.write_text(str(time.time() + delay_seconds))
        _spawn_train([sys.executable, "-m", "nudge.hooks._common", "retry", str(delay_seconds)])
        return
    _spawn_train([sys.executable, "-m", "nudge.cli", "train", "--background"])


def maybe_train(conn, config):
    """Train now if schedule=auto and batch ready. Otherwise the scheduler handles it."""
    schedule = config.get("train_schedule", "03:00")
    if schedule != "auto":
        return  # scheduled training handles it, don't train in the hook
    if db.count_trainable_untrained(conn) >= config.get("batch_min", 24):
        queue_training()


def _retry_after(delay_seconds: float):
    time.sleep(max(0.0, delay_seconds))
    TRAIN_RETRY_PATH.unlink(missing_ok=True)
    queue_training()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "retry":
        _retry_after(float(sys.argv[2]))
