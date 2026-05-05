"""Shared helpers for all hooks. Config, stdin, training, rating."""

import json
import os
import fcntl
import hashlib
import re
import secrets
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)
from reinforceclaw import db

TRAIN_LOG_PATH = Path.home() / ".reinforceclaw" / "train.log"
TRAIN_LOCK_PATH = Path.home() / ".reinforceclaw" / "train.lock"
TRAIN_QUEUE_LOCK_PATH = Path.home() / ".reinforceclaw" / "train.queue.lock"
TRAIN_RETRY_PATH = Path.home() / ".reinforceclaw" / "train.retry"
PENDING_DIR = Path.home() / ".reinforceclaw" / "pending"
PENDING_LOCK_PATH = Path.home() / ".reinforceclaw" / "pending.lock"
RESET_MARK_PATH = Path.home() / ".reinforceclaw" / "reset.marker"
MAX_HOOK_STDIN_BYTES = db.MAX_TEXT_BYTES * 4
DEFAULT_RETRY_DELAY = 30
TRAIN_LOG_MAX_BYTES = 10 * 1024 * 1024
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CONFIG_CACHE = None
_CONFIG_MTIME = None
_SENSITIVE_ENV = re.compile(r"(?i)(^|_)(api|auth|credential|key|password|pat|private|secret|session|token)($|_)")
_HF_ENV_KEYS = {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"}
_DROP_ENV_KEYS = {"AWS_ACCESS_KEY_ID", "AWS_SESSION_TOKEN", "DATABASE_URL", "SSH_AUTH_SOCK"}
_URL_CREDENTIALS = re.compile(r"^[a-z][a-z0-9+.-]*://[^/\s:@]+:[^@\s]+@", re.I)
SAFE_COMMANDS = {"good", "bad", "undo", "status"}
ADMIN_COMMANDS = {"train", "rollback", "reset", "on", "off"}
_COMMAND_NAMES = ("good", "bad", "undo", "status", "train", "rollback", "reset", "on", "off")
_GOOD_ALIASES = {"good", "yes", "y", "+1", "up", "thumbs up", "👍", "👍🏻", "👍🏼", "👍🏽", "👍🏾", "👍🏿"}
_BAD_ALIASES = {"bad", "no", "n", "-1", "down", "thumbs down", "👎", "👎🏻", "👎🏼", "👎🏽", "👎🏾", "👎🏿"}
_COMMAND_ALIASES = {**{cmd: cmd for cmd in _COMMAND_NAMES}, **{cmd: "good" for cmd in _GOOD_ALIASES}, **{cmd: "bad" for cmd in _BAD_ALIASES}}
PENDING_TTL_SECONDS = 1800
_CMD_PAIRS = [(prefix, alias, cmd) for prefix in ("rl", "rc", "reinforceclaw") for alias, cmd in _COMMAND_ALIASES.items()]
COMMANDS = {
    **{f"/{prefix} {alias}": cmd for prefix, alias, cmd in _CMD_PAIRS},
    **{f"/{cmd}": cmd for cmd in _COMMAND_NAMES},
}


def load_config():
    global _CONFIG_CACHE, _CONFIG_MTIME
    from reinforceclaw.cli import load_config as cli_load_config
    path = Path.home() / ".reinforceclaw" / "config.json"
    try:
        st = path.stat()
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = None
    if _CONFIG_CACHE is None or stamp != _CONFIG_MTIME:
        _CONFIG_CACHE, _CONFIG_MTIME = cli_load_config(persist=False), stamp
    return dict(_CONFIG_CACHE)


def read_stdin():
    try:
        raw = getattr(sys.stdin, "buffer", sys.stdin).read(MAX_HOOK_STDIN_BYTES + 1)
        raw = raw.encode("utf-8") if isinstance(raw, str) else raw
        if len(raw) > MAX_HOOK_STDIN_BYTES:
            return {}
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, EOFError, OSError, UnicodeDecodeError):
        return {}


def block(reason):
    print(json.dumps({"decision": "block", "reason": reason}))


def _content_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for part in (_content_text(item) for item in value) if part)
    if isinstance(value, dict):
        return _content_text(value.get("text") or value.get("content") or "")
    return ""


def _recent_transcript_rows(transcript_path, limit=64, max_bytes=262144):
    try:
        path = Path(transcript_path).expanduser()
        if path.is_symlink():
            return []
        resolved = path.resolve(strict=True)
        allowed = ((Path.home() / ".claude").resolve(strict=False), (Path.home() / ".codex").resolve(strict=False))
        if not any(resolved == root or root in resolved.parents for root in allowed):
            return []
        with path.open("rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            start = max(0, size - max_bytes)
            fh.seek(start)
            text = fh.read().decode("utf-8", "replace")
        lines = text.splitlines()
        if start and lines:
            lines = lines[1:]
        rows = []
        for line in lines[-limit:]:
            if line.strip():
                row = json.loads(line)
                rows.append(row.get("message") if isinstance(row, dict) and isinstance(row.get("message"), dict) else row)
        return rows
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []


def last_msg_from(data, role):
    role = str(role).lower()
    rows = data.get("messages")
    if isinstance(rows, list):
        for msg in reversed(rows):
            if isinstance(msg, dict) and str(msg.get("role", "")).lower() == role:
                return _content_text(msg.get("content", ""))

    transcript_path = data.get("transcript_path")
    if not transcript_path:
        return ""
    for msg in reversed(_recent_transcript_rows(transcript_path)):
        if isinstance(msg, dict) and str(msg.get("role", "")).lower() == role:
            return _content_text(msg.get("content", ""))
    return ""


def _message_list(data, limit=12):
    rows = data.get("messages")
    if isinstance(rows, list):
        source = rows[-limit:]
    else:
        source = []
        transcript_path = data.get("transcript_path")
        if transcript_path:
            source = _recent_transcript_rows(transcript_path)[-limit:]
    out = []
    for msg in source:
        if not isinstance(msg, dict) or msg.get("role") not in {"system", "user", "assistant"}:
            continue
        text = _content_text(msg.get("content", "")).strip()
        if text:
            out.append({"role": msg["role"], "content": text})
    return out[-limit:]


def training_context(data, prompt="", response="", limit=12):
    messages = _message_list(data, limit)
    p_strip, r_strip = prompt.strip(), response.strip()
    if r_strip and messages and messages[-1]["role"] == "assistant" and messages[-1]["content"].strip() == r_strip:
        messages.pop()
    if p_strip and not any(m["role"] == "user" and m["content"].strip() == p_strip for m in messages[-2:]):
        messages.append({"role": "user", "content": prompt})
    payload = {"messages": messages[-limit:]}
    for key in ("cwd", "session_id", "sessionId", "conversation_id", "conversationId", "transcript_path"):
        if data.get(key):
            payload[key] = str(data[key])
    return json.dumps(payload)


def normalize_command(value, *, allow_bare=False):
    text = _content_text(value).replace("\ufe0f", "")
    text = " ".join(text.strip().strip("`*_.! ").casefold().split())
    return COMMANDS.get(text) or (_COMMAND_ALIASES.get(text) if allow_bare else None)


def command_from_prompt(data):
    prompt_obj = data.get("prompt", {})
    prompt = prompt_obj.get("content", "") if isinstance(prompt_obj, dict) else prompt_obj
    return normalize_command(prompt)


def handle_agent_command(source, data, write=lambda _s: None):
    cmd = command_from_prompt(data)
    if cmd is None:
        sys.exit(0)
    config, context, conn = load_config(), pending_context(data), db.connect()
    reason = f"reinforceclaw: /rl {cmd}"
    try:
        if cmd in ADMIN_COMMANDS and not config.get("agent_admin_commands", False):
            write("Admin /rl commands are disabled for agent hooks. Use the reinforceclaw CLI.\n")
            return block("reinforceclaw: admin command disabled")
        if cmd in ("good", "bad"):
            if not config.get("model"):
                reason = "reinforceclaw not initialized"
                write("Run reinforceclaw init first.\n")
                return block(reason)
            rating = 1 if cmd == "good" else -1
            pending = pop_pending(source, context=context) if context else pop_pending(source)
            if not pending:
                reason = "reinforceclaw: no captured response"
                write("No captured response to rate.\n")
                return block(reason)
            added = False
            try:
                db.add_feedback(
                    conn, pending["model"], pending["prompt"], pending["response"], rating,
                    context=pending.get("context") or context, source=source,
                    event_id=pending.get("key"), rollout_context=pending.get("rollout_context"),
                )
                added = True
                maybe_train(conn, config)
            except Exception:
                if not added:
                    restore_pending(pending)
                raise
            write(f"Rated: {'good' if rating == 1 else 'bad'}\n")
        elif cmd == "undo":
            row = db.remove_last(conn, source=source, context=context, model=config.get("model"))
            write("Removed last rating\n" if row else "Nothing to undo.\n")
        elif cmd == "train":
            queue_training(force=True)
            write("Training started in background.\n")
        elif cmd == "status":
            model = config.get("model")
            counts, adapter = db.count(conn, model=model), db.latest_adapter(conn, model=model)
            ema, _ = db.get_ema(conn, model=model)
            reason = (f"Adapter: {'v'+str(adapter['version']) if adapter else 'none'} | "
                      f"Ratings: {counts['total']} ({counts['good']}+ {counts['bad']}-) | "
                      f"Untrained: {db.count_trainable_untrained(conn, model=model)} | EMA: {ema:.3f}")
            write(reason + "\n")
        elif cmd == "rollback":
            from reinforceclaw.cli import rollback_adapter
            prev, error = rollback_adapter(conn, config)
            reason = f"Rolled back to v{prev['version']}" if prev else "No previous adapter."
            if error:
                reason = f"{reason}; {error}" if prev else error
            write(reason + "\n")
        elif cmd == "reset":
            from reinforceclaw.cli import reset_state
            conn.close(); conn = None
            try:
                reset_state()
                reason = "reinforceclaw: /rl reset"
                write("Reset complete.\n")
            except RuntimeError as exc:
                reason = str(exc)
                write(reason + "\n")
        elif cmd in ("on", "off"):
            config["panel_enabled"] = cmd == "on"
            from reinforceclaw.cli import save_config
            save_config(config)
            write(f"Panel {cmd}.\n")
        block(reason)
    finally:
        if conn is not None:
            conn.close()


def pending_context(data):
    for key in ("session_id", "sessionId", "conversation_id", "conversationId", "transcript_path", "cwd"):
        value = data.get(key)
        if value:
            return str(value)
    return ""


def _safe_name(value):
    return (re.sub(r"[^a-zA-Z0-9_-]", "_", str(value)).strip("_") or "pending")[:120]


def _scrub_text(value):
    return db.scrub_text(value)


def _pending_prefix(source, context=None):
    safe = _safe_name(source)
    return f"{safe}-{hashlib.sha256(str(context).encode()).hexdigest()[:16]}" if context else safe


def _pending_path(source, context=None, key=None):
    suffix = f"-{_safe_name(key)}" if key else ""
    return PENDING_DIR / f"{_pending_prefix(source, context)}{suffix}.json"


def _sorted_pending(pattern):
    if PENDING_DIR.is_symlink() or not PENDING_DIR.is_dir():
        return []
    items = []
    for path in PENDING_DIR.glob(pattern):
        try:
            items.append((path.lstat().st_mtime, path))
        except OSError:
            pass
    return [path for _, path in sorted(items, reverse=True)]


@contextmanager
def _pending_lock():
    db.secure_private_dir(PENDING_LOCK_PATH.parent)
    fd = os.open(PENDING_LOCK_PATH, os.O_CREAT | os.O_RDWR | _O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _pending_paths(source, context=None, key=None):
    if key:
        safe_key = _safe_name(key)
        if context:
            return [_pending_path(source, context, safe_key)]
        return _sorted_pending(f"{_safe_name(source)}*-{safe_key}.json")
    return _sorted_pending(f"{_pending_prefix(source, context)}*.json")


def _valid_pending_payload(payload, source, key=None, context=None):
    if not isinstance(payload, dict) or _safe_name(payload.get("source")) != _safe_name(source):
        return False
    if key and payload.get("key") != key:
        return False
    if context is not None and str(payload.get("context") or "") != str(context):
        return False
    return True


def _read_pending_payload(path):
    fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW)
    with os.fdopen(fd, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_pending(source, model, prompt, response, context=None, channel=None, rollout_context=None):
    key = secrets.token_hex(16)
    db.secure_private_dir(PENDING_DIR.parent)
    with _pending_lock():
        db.secure_private_dir(PENDING_DIR)
        _prune_pending()
        payload = {"key": key, "source": source, "model": model, "prompt": _scrub_text(prompt), "response": _scrub_text(response), "ts": time.time()}
        if context:
            payload["context"] = _scrub_text(context)
        if channel:
            payload["channel"] = str(channel)
        if rollout_context:
            payload["rollout_context"] = _scrub_text(rollout_context)
        _write_pending_payload(payload)
    return key


def _write_pending_payload(payload):
    path = _pending_path(payload["source"], payload.get("context"), payload["key"])
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW
    committed = False
    try:
        fd = os.open(tmp, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
        db.secure_private_file(path)
        committed = True
    finally:
        if not committed:
            tmp.unlink(missing_ok=True)


def restore_pending(payload):
    if not isinstance(payload, dict) or not payload.get("key") or not payload.get("source"):
        return
    payload = {
        **payload,
        "prompt": _scrub_text(payload.get("prompt", "")),
        "response": _scrub_text(payload.get("response", "")),
    }
    if payload.get("context"):
        payload["context"] = _scrub_text(payload["context"])
    if payload.get("rollout_context"):
        payload["rollout_context"] = _scrub_text(payload["rollout_context"])
    db.secure_private_dir(PENDING_DIR.parent)
    with _pending_lock():
        db.secure_private_dir(PENDING_DIR)
        _write_pending_payload(payload)


def _prune_pending(max_age=PENDING_TTL_SECONDS):
    if PENDING_DIR.is_symlink() or not PENDING_DIR.is_dir():
        return
    cutoff = time.time() - max_age
    for path in PENDING_DIR.glob("*.json"):
        try:
            if path.lstat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            pass


def prune_pending(prefix=None, *, max_age=PENDING_TTL_SECONDS, reset_at=0.0):
    with _pending_lock():
        if PENDING_DIR.is_symlink() or not PENDING_DIR.is_dir():
            return
        cutoff = time.time() - max_age
        pattern = f"{_safe_name(prefix)}*.json" if prefix else "*.json"
        for path in PENDING_DIR.glob(pattern):
            try:
                ts = path.lstat().st_mtime
                if ts < reset_at or ts < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                pass


def pop_pending(source, key=None, context=None):
    with _pending_lock():
        _prune_pending()
        for path in _pending_paths(source, context, key):
            claimed = path.with_name(f".{path.name}.{os.getpid()}-{secrets.token_hex(8)}.claim")
            try:
                path.replace(claimed)
                payload = _read_pending_payload(claimed)
            except (json.JSONDecodeError, UnicodeDecodeError):
                claimed.unlink(missing_ok=True)
                path.unlink(missing_ok=True)
                continue
            except OSError:
                claimed.unlink(missing_ok=True)
                continue
            claimed.unlink(missing_ok=True)
            if _valid_pending_payload(payload, source, key, context):
                return payload
    return None


def reset_pending():
    with _pending_lock():
        if PENDING_DIR.is_symlink() or PENDING_DIR.is_file():
            PENDING_DIR.unlink(missing_ok=True)
        else:
            shutil.rmtree(PENDING_DIR, ignore_errors=True)


def _train_lock_held() -> bool:
    db.secure_private_dir(TRAIN_LOCK_PATH.parent)
    if TRAIN_LOCK_PATH.is_symlink():
        return True
    fd = os.open(TRAIN_LOCK_PATH, os.O_CREAT | os.O_RDWR | _O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def _retry_due():
    try:
        if TRAIN_RETRY_PATH.is_symlink():
            return 0.0
        return float(TRAIN_RETRY_PATH.read_text())
    except (OSError, ValueError):
        return 0.0


def _write_retry_due(due):
    db.secure_private_dir(TRAIN_RETRY_PATH.parent)
    tmp = TRAIN_RETRY_PATH.with_name(f".{TRAIN_RETRY_PATH.name}.{secrets.token_hex(8)}.tmp")
    committed = False
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(str(due))
        os.replace(tmp, TRAIN_RETRY_PATH)
        committed = True
    finally:
        if not committed:
            tmp.unlink(missing_ok=True)


def _base_env():
    db.secure_private_dir(TRAIN_LOG_PATH.parent)
    env = dict(os.environ)
    for key in tuple(env):
        if key in _HF_ENV_KEYS:
            continue
        if _SENSITIVE_ENV.search(key) or key in _DROP_ENV_KEYS or _URL_CREDENTIALS.search(str(env.get(key, ""))):
            env.pop(key, None)
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT_STR}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else PROJECT_ROOT_STR
    )
    return env


def _spawn_train(argv):
    with _open_train_log() as log:
        subprocess.Popen(
            argv,
            cwd=PROJECT_ROOT,
            env=_base_env(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _open_train_log():
    db.secure_private_dir(TRAIN_LOG_PATH.parent)
    try:
        if TRAIN_LOG_PATH.exists() and not TRAIN_LOG_PATH.is_symlink() and TRAIN_LOG_PATH.stat().st_size > TRAIN_LOG_MAX_BYTES:
            TRAIN_LOG_PATH.replace(TRAIN_LOG_PATH.with_suffix(".log.1"))
    except OSError:
        pass
    return os.fdopen(os.open(TRAIN_LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND | _O_NOFOLLOW, 0o600), "a", encoding="utf-8")


def queue_training(delay_seconds=0, *, force=False):
    """Spawn training in background — never block the hook."""
    db.secure_private_dir(TRAIN_QUEUE_LOCK_PATH.parent)
    if TRAIN_QUEUE_LOCK_PATH.is_symlink():
        return
    fd = os.open(TRAIN_QUEUE_LOCK_PATH, os.O_CREAT | os.O_RDWR | _O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return
        if _train_lock_held():
            return
        now = time.time()
        if _retry_due() > now and not force:
            return
        _write_retry_due(now + (delay_seconds if delay_seconds > 0 else DEFAULT_RETRY_DELAY))
        try:
            argv = ([sys.executable, "-m", "reinforceclaw.hooks._common", "retry", str(delay_seconds)]
                    if delay_seconds > 0 else [sys.executable, "-m", "reinforceclaw.cli", "train", "--background"])
            _spawn_train(argv)
        except Exception:
            TRAIN_RETRY_PATH.unlink(missing_ok=True)
            return
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def maybe_train(conn, config):
    """Train now if schedule=auto and batch ready. Otherwise the scheduler handles it."""
    schedule = config.get("train_schedule", "auto")
    if schedule != "auto":
        return  # scheduled training handles it, don't train in the hook
    if db.count_trainable_untrained(conn, model=config.get("model")) >= config.get("batch_min", 32):
        queue_training()


def _retry_after(delay_seconds: float):
    try:
        due = _retry_due() or (time.time() + max(0.0, delay_seconds))
        while True:
            current_due = _retry_due()
            if not current_due or abs(current_due - due) > 1.0:
                return
            remaining = due - time.time()
            if remaining <= 0:
                break
            time.sleep(min(60.0, remaining))
        TRAIN_RETRY_PATH.unlink(missing_ok=True)
        queue_training()
    except Exception as exc:
        db.secure_private_dir(TRAIN_LOG_PATH.parent)
        with _open_train_log() as log:
            log.write(f"retry error: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "retry":
        _retry_after(float(sys.argv[2]))
    elif len(sys.argv) > 1:
        sys.stderr.write("usage: python -m reinforceclaw.hooks._common retry <seconds>\n")
        sys.exit(2)
