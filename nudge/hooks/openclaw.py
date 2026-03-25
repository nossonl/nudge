"""OpenClaw hook. HTTP receiver for TS-side fetch() calls. Run: python -m nudge.hooks.openclaw"""
# openclaw hooks are TS by default. this python server catches the POSTs.
# simpler than writing TS — one language for everything.

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from nudge import db

_pending = {}  # session_key → last user message
_config = None


def _get_config():
    global _config
    if _config is None:
        from nudge.cli import load_config
        _config = load_config()
    return _config


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        path = self.path

        if path == "/feedback/message-in":
            # user message → stash for pairing
            _pending[body.get("sessionKey", "")] = body.get("content", "")

        elif path == "/feedback/message-out":
            # bot response → pair with last user message, ask for rating
            sk = body.get("sessionKey", "")
            prompt = _pending.pop(sk, "(openclaw)")
            response = body.get("content", "")
            cfg = _get_config()
            if cfg.get("model") and response:
                conn = db.connect()
                # auto-store as unrated, or let user rate via nudge CLI
                db.add_feedback(conn, cfg["model"], prompt, response, 0, source="openclaw")
                conn.close()

        elif path == "/feedback/rate":
            # explicit rating from user (via bot command or UI)
            conn = db.connect()
            fid = body.get("feedback_id")
            rating = body.get("rating", 0)
            if fid and rating:
                db.update_feedback_rating(conn, fid, rating)
            conn.close()

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, *args):
        pass  # quiet


def run(port=8420):
    print(f"Nudge OpenClaw receiver on :{port}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    run()
