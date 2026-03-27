"""OpenClaw backend. Receives POSTs from the TS plugin, writes to nudge DB."""
# the TS plugin (openclaw-plugin/) runs inside the gateway and catches messages
# from all 23+ platforms. it sends them here. we store, rate, and train.
# run: python -m nudge.hooks.openclaw

import json
from collections import OrderedDict
from http.server import HTTPServer, BaseHTTPRequestHandler
from nudge import db
from nudge.hooks._common import queue_training

_config = None
_latest_by_session = OrderedDict()  # capped at 1000 entries
MAX_SESSIONS = 1000
MAX_BODY = 1_048_576  # 1MB


def _cfg():
    # reload every time — user might change settings while server runs
    global _config
    from nudge.cli import load_config
    _config = load_config()
    return _config


def _maybe_train(conn):
    cfg = _cfg()
    if cfg.get("train_schedule", "03:00") != "auto":
        return
    if db.count_trainable_untrained(conn) >= cfg.get("batch_min", 24):
        queue_training()


def _status_response(self):
    conn = db.connect()
    counts = db.count(conn)
    ema, _ = db.get_ema(conn)
    a = db.latest_adapter(conn)
    trainable = db.count_trainable_untrained(conn)
    conn.close()
    self.send_response(200)
    self.end_headers()
    self.wfile.write(json.dumps({
        "adapter": f"v{a['version']}" if a else "none",
        "ratings": counts["total"],
        "good": counts["good"], "bad": counts["bad"],
        "untrained": trainable,
        "ema": round(ema, 3),
    }).encode())


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = self.headers.get("Content-Length")
        if not length or not length.isdigit() or int(length) > MAX_BODY:
            self.send_response(400); self.end_headers(); return

        if self.path == "/feedback/status":
            _status_response(self); return

        try:
            body = json.loads(self.rfile.read(int(length)))
        except (json.JSONDecodeError, ValueError):
            self.send_response(400); self.end_headers(); return

        if self.path == "/feedback/capture":
            cfg = _cfg()
            if cfg.get("model"):
                conn = db.connect()
                fid = db.add_feedback(
                    conn, cfg["model"],
                    body.get("prompt", "(openclaw)"),
                    body.get("response", ""),
                    0, source=body.get("channel", "openclaw"),
                )
                # cap session map at 1000
                if len(_latest_by_session) >= MAX_SESSIONS:
                    _latest_by_session.popitem(last=False)
                _latest_by_session[body.get("sessionKey", "")] = fid
                conn.close()

        elif self.path == "/feedback/rate":
            sk = body.get("sessionKey", "")
            rating = body.get("rating", 0)
            if rating not in (-1, 0, 1):
                self.send_response(400); self.end_headers(); return
            fid = body.get("feedback_id") or _latest_by_session.pop(sk, None)
            if fid and rating:
                conn = db.connect()
                db.update_feedback_rating(conn, fid, rating)
                _maybe_train(conn)
                conn.close()

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def do_GET(self):
        if self.path == "/feedback/status":
            _status_response(self)
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *a): pass  # silence per-request logging


def run(port=8420):
    print(f"nudge openclaw backend on :{port}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    run()
