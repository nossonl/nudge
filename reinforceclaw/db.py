"""SQLite store. WAL mode. 3 tables. That's it."""

import json
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path.home() / ".reinforceclaw" / "reinforceclaw.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    context TEXT,
    rating INTEGER NOT NULL,       -- +1 good, -1 bad.
    source TEXT DEFAULT 'cli',     -- cli | hook | discord | telegram | whatsapp
    trained INTEGER DEFAULT 0,
    adapter_version INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS adapters (
    version INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    parent_version INTEGER,
    status TEXT DEFAULT 'active',
    metrics TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS ema_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    reward_mean REAL DEFAULT 0.0,
    count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS training_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state TEXT
);
CREATE TABLE IF NOT EXISTS background_history (
    hour INTEGER PRIMARY KEY,
    pressure_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0
);
INSERT OR IGNORE INTO ema_state (id, reward_mean, count) VALUES (1, 0.0, 0);
INSERT OR IGNORE INTO training_state (id, state) VALUES (1, NULL);
"""


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


# -- feedback ops --

def _validate_rating(rating):
    if rating not in (-1, 0, 1):
        raise ValueError("rating must be -1, 0, or 1")


def add_feedback(conn, model, prompt, response, rating, context=None, source="cli"):
    _validate_rating(rating)
    cur = conn.execute(
        "INSERT INTO feedback (model, prompt, response, context, rating, source) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (model, prompt, response, context, rating, source),
    )
    conn.commit()
    return cur.lastrowid


def get_feedback_by_ids(conn, ids):
    if not ids:
        return []
    ph = ",".join("?" for _ in ids)
    rows = conn.execute(f"SELECT * FROM feedback WHERE id IN ({ph})", ids).fetchall()
    by_id = {row["id"]: dict(row) for row in rows}
    return [by_id[i] for i in ids if i in by_id]


def get_untrained(conn, limit=0):
    sql = "SELECT * FROM feedback WHERE trained=0 AND rating!=0 ORDER BY created_at ASC"
    params = ()
    if limit > 0:
        sql += " LIMIT ?"
        params = (limit,)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def get_replay(conn, limit=0):
    sql = "SELECT * FROM feedback WHERE trained=1 ORDER BY RANDOM()"
    params = ()
    if limit > 0:
        sql += " LIMIT ?"
        params = (limit,)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def mark_trained(conn, ids, adapter_version):
    if not ids:
        return
    ph = ",".join("?" for _ in ids)
    conn.execute(
        f"UPDATE feedback SET trained=1, adapter_version=? WHERE id IN ({ph})",
        [adapter_version] + ids,
    )
    conn.commit()


def update_feedback_rating(conn, feedback_id, rating):
    _validate_rating(rating)
    conn.execute("UPDATE feedback SET rating=? WHERE id=?", (rating, feedback_id))
    conn.commit()


def latest_pending(conn, source=None):
    sql = "SELECT * FROM feedback WHERE rating=0"
    params = []
    if source is not None:
        sql += " AND source=?"
        params.append(source)
    row = conn.execute(f"{sql} ORDER BY id DESC LIMIT 1", params).fetchone()
    return dict(row) if row else None


def remove_last(conn):
    row = conn.execute("SELECT * FROM feedback WHERE rating!=0 ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return None
    conn.execute("DELETE FROM feedback WHERE id = ?", (row["id"],))
    conn.commit()
    return dict(row)


def recent(conn, limit=20):
    """Last N ratings for history view."""
    rows = conn.execute(
        "SELECT id, prompt, rating, source, created_at FROM feedback "
        "ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def count(conn):
    row = conn.execute(
        "SELECT COUNT(*) as total, "
        "COALESCE(SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END),0) as good, "
        "COALESCE(SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END),0) as bad, "
        "COALESCE(SUM(CASE WHEN trained=0 THEN 1 ELSE 0 END),0) as untrained "
        "FROM feedback"
    ).fetchone()
    return dict(row)


def count_trainable_untrained(conn):
    row = conn.execute(
        "SELECT COALESCE(SUM(CASE WHEN trained=0 AND rating!=0 THEN 1 ELSE 0 END),0) AS total "
        "FROM feedback"
    ).fetchone()
    return row["total"]


# -- ema ops --

def get_ema(conn):
    r = conn.execute("SELECT reward_mean, count FROM ema_state WHERE id=1").fetchone()
    return r["reward_mean"], r["count"]


def get_training_state(conn):
    row = conn.execute("SELECT state FROM training_state WHERE id=1").fetchone()
    if not row or not row["state"]:
        return None
    return json.loads(row["state"])


def save_training_state(conn, state):
    conn.execute("UPDATE training_state SET state=? WHERE id=1", (json.dumps(state),))
    conn.commit()


def clear_training_state(conn):
    conn.execute("UPDATE training_state SET state=NULL WHERE id=1")
    conn.commit()


def record_background_event(conn, kind, hour):
    conn.execute("INSERT OR IGNORE INTO background_history (hour, pressure_count, success_count) VALUES (?,0,0)", (hour,))
    field = "pressure_count" if kind == "pressure" else "success_count"
    conn.execute(f"UPDATE background_history SET {field}={field}+1 WHERE hour=?", (hour,))
    conn.commit()


def background_history(conn):
    rows = conn.execute("SELECT hour, pressure_count, success_count FROM background_history").fetchall()
    return {row["hour"]: dict(row) for row in rows}


def update_ema(conn, mean, n):
    conn.execute("UPDATE ema_state SET reward_mean=?, count=? WHERE id=1", (mean, n))
    conn.commit()


def record_training_round(conn, mean, count, version, path, parent=None, metrics=None, feedback_ids=None):
    payload = dict(metrics or {})
    payload.setdefault("ema_mean", mean)
    payload.setdefault("ema_count", count)
    if feedback_ids is not None:
        payload.setdefault("feedback_ids", list(feedback_ids))
    with conn:
        conn.execute(
            "INSERT INTO adapters (version, path, parent_version, status, metrics) VALUES (?,?,?,?,?)",
            (version, path, parent, "candidate", json.dumps(payload) if payload else None),
        )


# -- adapter ops --

def list_adapters(conn):
    rows = conn.execute(
        "SELECT version, status, created_at FROM adapters ORDER BY version DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def add_adapter(conn, version, path, parent=None, metrics=None):
    conn.execute("UPDATE adapters SET status='inactive' WHERE status='active'")
    conn.execute(
        "INSERT INTO adapters (version, path, parent_version, metrics) VALUES (?,?,?,?)",
        (version, path, parent, json.dumps(metrics) if metrics else None),
    )
    conn.commit()


def latest_adapter(conn):
    r = conn.execute(
        "SELECT * FROM adapters WHERE status='active' ORDER BY version DESC LIMIT 1"
    ).fetchone()
    return dict(r) if r else None


def latest_candidate(conn):
    r = conn.execute(
        "SELECT * FROM adapters WHERE status='candidate' ORDER BY version DESC LIMIT 1"
    ).fetchone()
    return dict(r) if r else None


def activate_adapter(conn, version):
    with conn:
        conn.execute("UPDATE adapters SET status='inactive' WHERE status='active'")
        conn.execute("UPDATE adapters SET status='active' WHERE version=?", (version,))
    return latest_adapter(conn)


def activate_training_round(conn, version, mean=None, count=None, feedback_ids=None):
    row = conn.execute("SELECT metrics FROM adapters WHERE version=?", (version,)).fetchone()
    metrics = json.loads(row["metrics"]) if row and row["metrics"] else {}
    if mean is None:
        mean = metrics.get("ema_mean")
    if count is None:
        count = metrics.get("ema_count")
    feedback_ids = list(feedback_ids or metrics.get("feedback_ids") or [])
    with conn:
        if mean is not None and count is not None:
            conn.execute("UPDATE ema_state SET reward_mean=?, count=? WHERE id=1", (mean, count))
        conn.execute("UPDATE adapters SET status='inactive' WHERE status='active'")
        conn.execute("UPDATE adapters SET status='active' WHERE version=?", (version,))
        if feedback_ids:
            ph = ",".join("?" for _ in feedback_ids)
            conn.execute(
                f"UPDATE feedback SET trained=1, adapter_version=? WHERE id IN ({ph})",
                [version] + feedback_ids,
            )


def reject_adapter(conn, version):
    conn.execute("UPDATE adapters SET status='rejected' WHERE version=?", (version,))
    conn.commit()


def rollback_to(conn, version):
    row = conn.execute("SELECT version FROM adapters WHERE version=?", (version,)).fetchone()
    if not row:
        return None
    conn.execute("UPDATE adapters SET status='inactive' WHERE status='active'")
    conn.execute("UPDATE adapters SET status='rolled_back' WHERE version > ?", (version,))
    conn.execute("UPDATE adapters SET status='active' WHERE version = ?", (version,))
    conn.commit()
    return latest_adapter(conn)


def rollback(conn):
    latest = latest_adapter(conn)
    if not latest:
        return None
    previous = conn.execute(
        "SELECT version FROM adapters WHERE status='inactive' ORDER BY version DESC LIMIT 1"
    ).fetchone()
    if not previous:
        return None
    conn.execute("UPDATE adapters SET status='rolled_back' WHERE version=?", (latest["version"],))
    conn.execute("UPDATE adapters SET status='active' WHERE version = ?", (previous["version"],))
    conn.commit()
    return latest_adapter(conn)


def cleanup_adapters(conn, keep=20):
    rows = conn.execute("SELECT version, path FROM adapters ORDER BY version DESC").fetchall()
    paths = []
    for r in rows[keep:]:
        paths.append(r["path"])
        conn.execute("DELETE FROM adapters WHERE version=?", (r["version"],))
    if paths:
        conn.commit()
    return paths


# -- nuclear option --

def reset_all(conn):
    conn.executescript(
        "DELETE FROM feedback; DELETE FROM adapters; "
        "DELETE FROM background_history; "
        "UPDATE ema_state SET reward_mean=0.0, count=0 WHERE id=1;"
        "UPDATE training_state SET state=NULL WHERE id=1;"
    )
    conn.commit()
