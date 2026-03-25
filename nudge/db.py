"""SQLite store. WAL mode. 3 tables. That's it."""

import json
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path.home() / ".nudge" / "nudge.db"

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
    status TEXT DEFAULT 'active',  -- active | rolled_back
    metrics TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS ema_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    reward_mean REAL DEFAULT 0.0,
    count INTEGER DEFAULT 0
);
INSERT OR IGNORE INTO ema_state (id, reward_mean, count) VALUES (1, 0.0, 0);
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


def _query(conn, where, order, limit=0):
    sql = f"SELECT * FROM feedback WHERE {where} ORDER BY {order}"
    if limit > 0:
        sql += f" LIMIT {limit}"
    return [dict(r) for r in conn.execute(sql).fetchall()]


def get_untrained(conn, limit=0):
    # skip unrated (0) entries from openclaw etc.
    return _query(conn, "trained=0 AND rating!=0", "created_at ASC", limit)


def get_replay(conn, limit=0):
    return _query(conn, "trained=1", "RANDOM()", limit)


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


def update_ema(conn, mean, n):
    conn.execute("UPDATE ema_state SET reward_mean=?, count=? WHERE id=1", (mean, n))
    conn.commit()


# -- adapter ops --

def add_adapter(conn, version, path, parent=None, metrics=None):
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


def rollback(conn):
    """Kill newest active adapter, return the one before it."""
    latest = latest_adapter(conn)
    if not latest:
        return None
    conn.execute("UPDATE adapters SET status='rolled_back' WHERE version=?", (latest["version"],))
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
        "UPDATE ema_state SET reward_mean=0.0, count=0 WHERE id=1;"
    )
    conn.commit()
