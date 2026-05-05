"""OpenClaw backend. Receives POSTs from the TS plugin, writes to reinforceclaw DB."""
# the TS plugin (openclaw-plugin/) runs inside the gateway and catches messages
# from all 23+ platforms. it sends them here; only rated turns go to SQLite.
# run: python -m reinforceclaw.hooks.openclaw

import json
import hmac
import os
import socket
import time
from collections import deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import BoundedSemaphore, RLock
from urllib.parse import urlparse
from reinforceclaw import db
from reinforceclaw.hooks._common import (
    ADMIN_COMMANDS, RESET_MARK_PATH, maybe_train, normalize_command, pop_pending,
    prune_pending, queue_training, restore_pending, save_pending,
)

_config_stamp = None
_config = None
_config_lock = RLock()
_rate_lock = RLock()
_last_prune = 0.0
_request_times = deque()
MAX_PENDING_AGE = 1800
MAX_BODY = db.MAX_TEXT_BYTES * 4
MAX_PATH = 512
MAX_SESSION_KEY = 512
MAX_REQUESTS_PER_MINUTE = 240
MAX_THREADS = 32
SECRET_HEADER = "X-ReinforceClaw-Secret"


def _cfg():
    global _config, _config_stamp
    from reinforceclaw.cli import load_config
    path = Path.home() / ".reinforceclaw" / "config.json"
    try:
        st = path.stat()
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = None
    with _config_lock:
        if _config is None or stamp != _config_stamp:
            _config, _config_stamp = load_config(persist=False), stamp
        return dict(_config)


def _save_cfg(cfg):
    global _config, _config_stamp
    from reinforceclaw.cli import CONFIG_PATH, save_config
    with _config_lock:
        save_config(cfg)
        try:
            st = CONFIG_PATH.stat()
            _config_stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            _config_stamp = None
        _config = dict(cfg)


def _shared_secret():
    secret = str(os.environ.get("REINFORCECLAW_OPENCLAW_SECRET") or _cfg().get("openclaw_secret") or "")
    return secret if len(secret) >= 32 else None


def _authorized(headers):
    secret = _shared_secret()
    if not secret:
        return False
    got = headers.get(SECRET_HEADER)
    return bool(got) and hmac.compare_digest(got, secret)


def _status_payload(conn):
    model = _cfg().get("model")
    counts = db.count(conn, model=model)
    ema, _ = db.get_ema(conn, model=model)
    adapter = db.latest_adapter(conn, model=model)
    return {
        "adapter": f"v{adapter['version']}" if adapter else "none",
        "ratings": counts["total"],
        "good": counts["good"],
        "bad": counts["bad"],
        "untrained": db.count_trainable_untrained(conn, model=model),
        "ema": round(ema, 3),
    }


def _send_json(handler, payload, code=200):
    body = json.dumps(payload).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(body)


def _send_error(handler, code):
    handler.send_response(code)
    handler.send_header("Content-Length", "0")
    handler.send_header("Connection", "close")
    handler.end_headers()


def _fail(handler, message, code=400):
    _send_json(handler, {"ok": False, "message": message}, code)


def _session_key(body):
    value = str(body.get("sessionKey") or "").strip()
    return value if 0 < len(value.encode("utf-8")) <= MAX_SESSION_KEY else ""


def _local_origin(origin):
    try:
        parsed = urlparse(origin)
        return parsed.scheme in {"http", "https"} and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    except ValueError:
        return False


def _local_host(host):
    host = str(host or "").lower()
    return not host or host.startswith("[::1]") or host.split(":", 1)[0] in {"127.0.0.1", "localhost", "::1"}


def _status_response(self):
    conn = db.connect()
    try:
        payload = _status_payload(conn)
    finally:
        conn.close()
    _send_json(self, payload)


def _rate_limited():
    now = time.time()
    with _rate_lock:
        while _request_times and now - _request_times[0] > 60:
            _request_times.popleft()
        if len(_request_times) >= MAX_REQUESTS_PER_MINUTE:
            return True
        _request_times.append(now)
        return False


def _prune_sessions():
    global _last_prune
    now = time.time()
    with _config_lock:
        if now - _last_prune < 30:
            return
        _last_prune = now
    try:
        reset_at = Path(RESET_MARK_PATH).stat().st_mtime
    except OSError:
        reset_at = 0
    prune_pending("openclaw", max_age=MAX_PENDING_AGE, reset_at=reset_at)


def _pop_session(session_key):
    return pop_pending("openclaw", context=session_key)


def _rollout_context(body, prompt, session_key):
    ctx = body.get("context")
    if isinstance(ctx, str):
        return ctx
    if isinstance(ctx, dict):
        return json.dumps(ctx)
    if isinstance(ctx, list):
        return json.dumps({"messages": ctx})
    return json.dumps({"messages": [{"role": "user", "content": prompt}], "session": session_key, "channel": body.get("channel", "openclaw")})


def _add_rating_feedback(conn, cfg, item, session_key, rating):
    db.add_feedback(
        conn, item.get("model") or cfg["model"], item["prompt"], item["response"], rating,
        context=session_key, source=item.get("channel") or item.get("source", "openclaw"),
        event_id=item.get("key"), rollout_context=item.get("rollout_context"),
    )


def _run_command(cmd, session_key, pending_key=None):
    cmd = normalize_command(cmd, allow_bare=True) or str(cmd).strip().casefold()
    pending_key = str(pending_key or "").strip()
    _prune_sessions()
    cfg = _cfg()
    conn = db.connect()
    try:
        if cmd in ADMIN_COMMANDS and not cfg.get("agent_admin_commands", False):
            return {"ok": False, "message": "admin command disabled"}
        if cmd in ("good", "bad"):
            item = None
            if pending_key:
                item = pop_pending("openclaw", context=session_key, key=pending_key) or pop_pending("openclaw", key=pending_key)
            else:
                item = _pop_session(session_key)
            if item:
                rating = 1 if cmd == "good" else -1
                rating_context = item.get("context") or session_key
                try:
                    _add_rating_feedback(conn, cfg, item, rating_context, rating)
                except Exception:
                    restore_pending(item)
                    raise
            else:
                return {"ok": False, "message": "No captured response to rate yet."}
            maybe_train(conn, cfg)
            return {"ok": True, "message": f"rated: {cmd}"}
        if cmd == "undo":
            row = db.remove_last(conn, context=session_key, model=cfg.get("model"))
            return {"ok": bool(row), "message": "removed last rating" if row else "nothing to undo"}
        if cmd == "train":
            queue_training(force=True)
            return {"ok": True, "message": "training queued"}
        if cmd == "status":
            return {"ok": True, "message": _status_payload(conn)}
        if cmd == "rollback":
            from reinforceclaw.cli import rollback_adapter
            prev, error = rollback_adapter(conn, cfg)
            if prev:
                msg = f"rolled back to v{prev['version']}"
                return {"ok": True, "message": f"{msg}; {error}" if error else msg}
            return {"ok": False, "message": error or "no previous adapter"}
        if cmd == "reset":
            from reinforceclaw.cli import reset_state
            try:
                conn.close()
                conn = None
                reset_state()
            except RuntimeError as exc:
                return {"ok": False, "message": str(exc)}
            return {"ok": True, "message": "reset complete; restart or repoint your model server"}
        if cmd in ("on", "off"):
            cfg = {**cfg, "panel_enabled": cmd == "on"}
            _save_cfg(cfg)
            return {"ok": True, "message": f"panel {cmd}"}
        return {"ok": False, "message": "unknown command"}
    finally:
        if conn is not None:
            conn.close()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.request.settimeout(10)

    def do_POST(self):
        if not _authorized(self.headers):
            _send_error(self, 403); return
        if _rate_limited():
            _send_error(self, 429); return
        if len(self.path) > MAX_PATH:
            _send_error(self, 414); return
        origin = self.headers.get("Origin")
        if origin and not _local_origin(origin):
            _send_error(self, 403); return
        if not _local_host(self.headers.get("Host", "")):
            _send_error(self, 403); return
        if self.path == "/feedback/status":
            _status_response(self); return
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type and content_type != "application/json":
            _send_error(self, 415); return
        length = self.headers.get("Content-Length")
        if not length or not length.isdigit():
            _send_error(self, 411); return
        size = int(length)
        if size <= 0 or size > MAX_BODY:
            _send_error(self, 400); return
        try:
            raw = self.rfile.read(size)
            if len(raw) != size:
                _send_error(self, 400); return
            body = json.loads(raw)
            if not isinstance(body, dict):
                _send_error(self, 400); return
        except (ValueError, UnicodeDecodeError, socket.timeout):
            _send_error(self, 400); return

        if self.path == "/feedback/capture":
            _prune_sessions()
            cfg = _cfg()
            sk = _session_key(body)
            if not sk:
                _fail(self, "missing sessionKey"); return
            response = str(body.get("response") or "").strip()
            if not response:
                _fail(self, "empty response"); return
            if not cfg.get("model"):
                _fail(self, "reinforceclaw not initialized"); return
            prompt = body.get("prompt")
            prompt = prompt.strip() if isinstance(prompt, str) else ""
            if not prompt:
                _fail(self, "empty prompt"); return
            rollout = _rollout_context(body, prompt, sk)
            try:
                key = save_pending(
                    "openclaw", cfg["model"], prompt, response,
                    context=sk, channel=body.get("channel", "openclaw"), rollout_context=rollout,
                )
            except ValueError as exc:
                _fail(self, str(exc), 413); return
            _send_json(self, {"ok": True, "key": key})
            return

        elif self.path == "/feedback/rate":
            _prune_sessions()
            sk = _session_key(body)
            if not sk:
                _fail(self, "missing sessionKey"); return
            rating = body.get("rating", 0)
            if isinstance(rating, bool) or not isinstance(rating, int) or rating not in (-1, 0, 1):
                _send_error(self, 400); return
            if rating == 0:
                pop_pending("openclaw", context=sk)
                _send_json(self, {"ok": True})
                return
            item = _pop_session(sk)
            if not item:
                _fail(self, "No captured response to rate yet."); return
            try:
                conn = db.connect()
                cfg = _cfg()
                try:
                    try:
                        _add_rating_feedback(conn, cfg, item, sk, rating)
                    except Exception:
                        restore_pending(item)
                        raise
                    maybe_train(conn, cfg)
                finally:
                    conn.close()
            except Exception:
                _fail(self, "rating failed", 500); return
            _send_json(self, {"ok": True})
            return
        elif self.path == "/feedback/command":
            sk = _session_key(body)
            if not sk:
                _fail(self, "missing sessionKey"); return
            try:
                result = _run_command(body.get("command", ""), sk, body.get("pendingKey"))
            except Exception:
                _fail(self, "command failed", 500); return
            _send_json(self, result, 200 if result.get("ok") else 400)
            return

        _fail(self, "unknown path", 404)

    def do_GET(self):
        if not _authorized(self.headers):
            _send_error(self, 403); return
        if _rate_limited():
            _send_error(self, 429); return
        if len(self.path) > MAX_PATH:
            _send_error(self, 414); return
        origin = self.headers.get("Origin")
        if origin and not _local_origin(origin):
            _send_error(self, 403); return
        if not _local_host(self.headers.get("Host", "")):
            _send_error(self, 403); return
        if self.path == "/feedback/status":
            _status_response(self)
        else:
            _send_error(self, 404)

    def log_message(self, *a): pass  # silence per-request logging


class LimitedThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args, max_threads=MAX_THREADS, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_slots = BoundedSemaphore(max_threads)

    def process_request(self, request, client_address):
        if not self._thread_slots.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._thread_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._thread_slots.release()


def run(port=8420):
    if not _shared_secret():
        raise RuntimeError("reinforceclaw openclaw backend requires openclaw_secret")
    print(f"reinforceclaw openclaw backend on :{port}")
    LimitedThreadingHTTPServer.allow_reuse_address = True
    LimitedThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    run()
