"""OpenClaw backend. Receives POSTs from the TS plugin, writes to nudge DB."""
# the TS plugin (openclaw-plugin/) runs inside the gateway and catches messages
# from all 23+ platforms. it sends them here. we store, rate, and train.
# run: python -m nudge.hooks.openclaw

import json
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from nudge import db

_config = None
_latest_by_session = {}  # session → feedback_id of the last unrated response


def _cfg():
    global _config
    if _config is None:
        from nudge.cli import load_config
        _config = load_config()
    return _config


def _maybe_train():
    """kick off training in background if we have enough ratings"""
    conn = db.connect()
    if db.count_trainable_untrained(conn) >= _cfg().get("batch_min", 16):
        subprocess.Popen(
            [sys.executable, "-m", "nudge.cli", "train"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    conn.close()


MAX_BODY = 1_048_576  # 1MB cap — don't let anyone blow up our memory


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = self.headers.get("Content-Length")
        if not length or not length.isdigit() or int(length) > MAX_BODY:
            self.send_response(400)
            self.end_headers()
            return
        body = json.loads(self.rfile.read(int(length)))
        path = self.path

        if path == "/feedback/capture":
            # TS plugin sends prompt+response pair after each bot reply
            cfg = _cfg()
            if cfg.get("model"):
                conn = db.connect()
                fid = db.add_feedback(
                    conn, cfg["model"],
                    body.get("prompt", "(openclaw)"),
                    body.get("response", ""),
                    0,  # unrated until user scores it
                    source=body.get("channel", "openclaw"),
                )
                _latest_by_session[body.get("sessionKey", "")] = fid
                conn.close()

        elif path == "/feedback/rate":
            # user said /rl good or /rl bad from any platform
            sk = body.get("sessionKey", "")
            rating = body.get("rating", 0)
            fid = body.get("feedback_id") or _latest_by_session.pop(sk, None)
            if fid and rating:
                conn = db.connect()
                db.update_feedback_rating(conn, fid, rating)
                conn.close()
                _maybe_train()

        elif path == "/feedback/status":
            conn = db.connect()
            counts = db.count(conn)
            ema, _ = db.get_ema(conn)
            a = db.latest_adapter(conn)
            conn.close()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({
                "adapter": f"v{a['version']}" if a else "none",
                "ratings": counts["total"],
                "good": counts["good"], "bad": counts["bad"],
                "untrained": counts["untrained"],
                "ema": round(ema, 3),
            }).encode())
            return

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def do_GET(self):
        # GET /feedback/status also works
        if self.path == "/feedback/status":
            self.do_POST()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


def run(port=8420):
    print(f"nudge openclaw backend on :{port}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    run()
