"""SQLite store. WAL mode. Small, local, boring."""

import json
import os
import random
import re
import sqlite3
from pathlib import Path

PRIVATE_ROOT = Path.home() / ".reinforceclaw"
DB_PATH = PRIVATE_ROOT / "reinforceclaw.db"
MAX_TEXT_BYTES = 2_000_000
TRUNCATION_MARKER = "\n...[reinforceclaw truncated oversized text]"
RECENT_LIMIT = 20
MAX_FEEDBACK_ROWS = 100_000
MAX_SOURCE_BYTES = 128
MAX_EVENT_ID_BYTES = 512
_SECRET_RE = re.compile(
    r"(?i)(?P<prefix>authorization:\s*bearer\s+|bearer\s+|[?&]token=|"
    r"['\"]?\b(?:HF_TOKEN|HUGGINGFACE_HUB_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACEHUB_API_TOKEN|OPENAI_API_KEY|"
    r"ANTHROPIC_API_KEY|GEMINI_API_KEY|GOOGLE_API_KEY|XAI_API_KEY|AWS_ACCESS_KEY_ID|"
    r"AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|api[-_]?key|token|secret|password)\b['\"]?\s*[:=]\s*['\"]?)"
    r"(?P<secret>[^\s'\"&,;]+)|(?P<standalone>hf_[A-Za-z0-9_-]{12,}|"
    r"sk-(?:ant-|proj-)?[A-Za-z0-9_-]{16,}|A[KS]IA[A-Z0-9]{16}|github_pat_[A-Za-z0-9_]{36,}|"
    r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|[sp]k_live_[A-Za-z0-9]{16,}|"
    r"AIza[0-9A-Za-z_-]{35}|https://hooks\.slack\.com/services/[A-Za-z0-9_/+-]{20,}|"
    r"\b\d{6,12}:[A-Za-z0-9_-]{30,}\b|mfa\.[A-Za-z0-9_-]{20,}|"
    r"[A-Za-z0-9_-]{24}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}|"
    r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----|"
    r"https?://[^\s/@:]+:[^\s/@]+@)"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    context TEXT,
    rollout_context TEXT,
    event_id TEXT,
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
CREATE TABLE IF NOT EXISTS model_ema_state (
    model TEXT PRIMARY KEY,
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
CREATE TABLE IF NOT EXISTS migrations (
    name TEXT PRIMARY KEY,
    applied_at TEXT DEFAULT (datetime('now'))
);
"""

_TABLES = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", SCHEMA))
_ALTER_TABLES = {"feedback", "adapters", "ema_state"}
_ALTER_COLUMNS = {
    "feedback": {
        "context": "TEXT",
        "rollout_context": "TEXT",
        "event_id": "TEXT",
        "source": "TEXT DEFAULT 'cli'",
        "trained": "INTEGER DEFAULT 0",
        "adapter_version": "INTEGER",
    },
    "adapters": {"parent_version": "INTEGER", "status": "TEXT DEFAULT 'active'", "metrics": "TEXT", "created_at": "TEXT"},
    "ema_state": {"reward_mean": "REAL DEFAULT 0.0", "count": "INTEGER DEFAULT 0"},
}
_FEEDBACK_WHERE = {"trained=0 AND rating!=0", "trained=1 AND rating!=0"}
_FEEDBACK_ORDER = {"created_at ASC", "id ASC"}


def secure_private_dir(path: Path = PRIVATE_ROOT) -> Path:
    path = Path(path).expanduser()
    if path.is_symlink():
        raise PermissionError(f"refusing symlink private dir: {path}")
    old = os.umask(0o077)
    try:
        path.mkdir(parents=True, exist_ok=True)
    finally:
        os.umask(old)
    if path.is_symlink():
        raise PermissionError(f"refusing symlink private dir: {path}")
    try:
        path.chmod(0o700)
    except OSError as exc:
        raise PermissionError(f"could not secure private dir: {path}") from exc
    return path


def secure_private_file(path: Path) -> Path:
    path = Path(path).expanduser()
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return path
    except OSError as exc:
        raise PermissionError(f"could not secure private file: {path}") from exc
    try:
        os.fchmod(fd, 0o600)
    except OSError as exc:
        raise PermissionError(f"could not secure private file: {path}") from exc
    finally:
        os.close(fd)
    return path


def _secure_sqlite_files(path: Path) -> None:
    for item in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        secure_private_file(item)


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path).expanduser() if db_path else DB_PATH
    secure_private_dir(path.parent)
    if path.is_symlink():
        raise PermissionError(f"refusing symlink database path: {path}")
    db_existed = path.exists()
    if db_existed:
        secure_private_file(path)
    old = os.umask(0o077)
    try:
        conn = sqlite3.connect(str(path))
    finally:
        os.umask(old)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA secure_delete=ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.execute("INSERT OR IGNORE INTO ema_state (id, reward_mean, count) VALUES (1, 0.0, 0)")
    conn.execute("INSERT OR IGNORE INTO training_state (id, state) VALUES (1, NULL)")
    conn.commit()
    _secure_sqlite_files(path)
    return conn


def init(db_path: Path | None = None) -> None:
    connect(db_path).close()


def _columns(conn, table):
    if table not in _TABLES:
        raise ValueError(f"invalid table: {table}")
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _add_missing_columns(conn, table, columns):
    if table not in _ALTER_TABLES:
        raise ValueError(f"invalid migration table: {table}")
    if columns != _ALTER_COLUMNS[table]:
        raise ValueError(f"invalid migration columns: {table}")
    existing = _columns(conn, table)
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def _migrate(conn):
    """Bring old local databases forward without dropping user ratings."""
    for table, columns in _ALTER_COLUMNS.items():
        _add_missing_columns(conn, table, columns)
    if not conn.execute("SELECT 1 FROM migrations WHERE name='feedback_v2'").fetchone():
        # Legacy unrated rows are intentionally discarded; ignored responses should not train.
        conn.execute("DELETE FROM feedback WHERE rating=0")
        conn.execute("DROP INDEX IF EXISTS idx_feedback_unique_rating")
        conn.execute("INSERT OR IGNORE INTO migrations(name) VALUES('feedback_v2')")
    conn.execute("DROP INDEX IF EXISTS idx_feedback_content_no_event")
    conn.execute("INSERT OR IGNORE INTO migrations(name) VALUES('feedback_no_event_unique_removed')")
    if not conn.execute("SELECT 1 FROM migrations WHERE name='feedback_event_unique'").fetchone():
        conn.execute(
            "DELETE FROM feedback WHERE event_id IS NOT NULL AND id NOT IN ("
            "SELECT MIN(id) FROM feedback WHERE event_id IS NOT NULL GROUP BY source, event_id)"
        )
        conn.execute("INSERT OR IGNORE INTO migrations(name) VALUES('feedback_event_unique')")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_event ON feedback(source, event_id) WHERE event_id IS NOT NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_untrained ON feedback(trained, rating, source, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_recent_rated ON feedback(id DESC) WHERE rating!=0")
    conn.execute("DROP INDEX IF EXISTS idx_feedback_recent")
    conn.commit()


# -- feedback ops --

def _validate_rating(rating, allow_delete=False):
    allowed = (-1, 0, 1) if allow_delete else (-1, 1)
    if rating not in allowed:
        raise ValueError("rating must be -1, 0, or 1" if allow_delete else "rating must be -1 or 1")


def redact_secrets(value):
    return _SECRET_RE.sub(lambda m: f"{m.group('prefix') or ''}[REDACTED]", str(value))


def scrub_text(value):
    text = redact_secrets(value)
    raw = text.encode("utf-8")
    if len(raw) <= MAX_TEXT_BYTES:
        return text
    marker = TRUNCATION_MARKER.encode("utf-8")
    keep = max(0, MAX_TEXT_BYTES - len(marker))
    head = keep // 2
    tail = keep - head
    return raw[:head].decode("utf-8", "ignore") + TRUNCATION_MARKER + raw[-tail:].decode("utf-8", "ignore")


def _bounded_text(value, max_bytes, default=""):
    text = str(value if value is not None else default)
    raw = text.encode("utf-8")
    return text if len(raw) <= max_bytes else raw[:max_bytes].decode("utf-8", "ignore")


def _row_get(row, key, default=None):
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


def _mark_feedback_mutation(conn, row):
    if not row:
        return
    stale_versions = set()
    model = _row_get(row, "model")
    try:
        state = get_training_state(conn) or {}
    except (json.JSONDecodeError, TypeError):
        state = {}
    if row["adapter_version"] is not None:
        stale_versions.update(
            r["version"] for r in conn.execute(
                "SELECT version, path, metrics FROM adapters WHERE status!='rejected' AND version>=?",
                (row["adapter_version"],),
            )
            if _adapter_matches_model(r, model)
        )
    for adapter in conn.execute("SELECT version, path, metrics FROM adapters WHERE status!='rejected' AND metrics IS NOT NULL"):
        try:
            if int(row["id"]) in set(json.loads(adapter["metrics"]).get("feedback_ids") or []):
                stale_versions.add(adapter["version"])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    if stale_versions:
        ph = ",".join("?" for _ in stale_versions)
        conn.execute(f"UPDATE adapters SET status='stale' WHERE version IN ({ph})", tuple(stale_versions))
    if int(row["id"]) in (set(state.get("batch_ids") or []) | set(state.get("fresh_ids") or [])) or state.get("parent_version") in stale_versions:
        conn.execute("UPDATE training_state SET state=NULL WHERE id=1")


def add_feedback(conn, model, prompt, response, rating, context=None, source="cli", event_id=None, rollout_context=None):
    _validate_rating(rating)
    prompt, response = scrub_text(prompt), scrub_text(response)
    context = scrub_text(context) if context else None
    rollout_context = scrub_text(rollout_context) if rollout_context else None
    source = _bounded_text(source or "cli", MAX_SOURCE_BYTES, "cli")
    event_id = _bounded_text(event_id, MAX_EVENT_ID_BYTES) if event_id else None
    with conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO feedback (model, prompt, response, context, rollout_context, event_id, rating, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (model, prompt, response, context, rollout_context, event_id, rating, source),
        )
    if cur.rowcount == 0:
        row = (
            conn.execute("SELECT id FROM feedback WHERE source=? AND event_id=? LIMIT 1", (source, event_id)).fetchone()
            if event_id else
            conn.execute(
                "SELECT id FROM feedback WHERE model=? AND prompt=? AND response=? AND rating=? AND source=? ORDER BY id DESC LIMIT 1",
                (model, prompt, response, rating, source),
            ).fetchone()
        )
        return row["id"] if row else None
    return cur.lastrowid


def get_feedback_by_ids(conn, ids):
    if not ids:
        return []
    ids = tuple(int(i) for i in ids)
    ph = ",".join("?" for _ in ids)
    rows = conn.execute(f"SELECT * FROM feedback WHERE id IN ({ph})", ids).fetchall()
    by_id = {row["id"]: dict(row) for row in rows}
    return [by_id[i] for i in ids if i in by_id]


def ignore_feedback_ids(conn, ids):
    ids = tuple(int(i) for i in ids)
    if not ids:
        return 0
    ph = ",".join("?" for _ in ids)
    rows = conn.execute(f"SELECT id, model, adapter_version FROM feedback WHERE id IN ({ph})", ids).fetchall()
    with conn:
        for row in rows:
            _mark_feedback_mutation(conn, row)
        cur = conn.execute(
            f"UPDATE feedback SET rating=0, trained=1, adapter_version=NULL WHERE id IN ({ph})",
            ids,
        )
        conn.execute("UPDATE training_state SET state=NULL WHERE id=1")
    return cur.rowcount


def _feedback_query(base_where: str, order: str, limit: int, source=None, model=None):
    if base_where not in _FEEDBACK_WHERE or order not in _FEEDBACK_ORDER:
        raise ValueError("invalid feedback query")
    sql = f"SELECT * FROM feedback WHERE {base_where}"
    filters, params = _feedback_filters(source, model)
    sql += filters
    sql += f" ORDER BY {order}"
    if limit > 0:
        sql += " LIMIT ?"
        params = (*params, limit)
    return sql, params


def _source_clause(source):
    return (" AND source=?", (source,)) if source is not None else ("", ())


def _feedback_filters(source=None, model=None):
    parts, params = [], []
    if source is not None:
        parts.append("source=?")
        params.append(source)
    if model is not None:
        parts.append("model=?")
        params.append(model)
    return (" AND " + " AND ".join(parts), tuple(params)) if parts else ("", ())


def get_untrained(conn, limit=0, source=None, model=None):
    sql, params = _feedback_query("trained=0 AND rating!=0", "created_at ASC", limit, source, model)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def get_replay(conn, limit=0, source=None, model=None):
    if limit > 0:
        filters, params = _feedback_filters(source, model)
        bounds = conn.execute(f"SELECT MIN(id) lo, MAX(id) hi FROM feedback WHERE trained=1 AND rating!=0{filters}", params).fetchone()
        if bounds["lo"] is None:
            return []
        start = random.randint(bounds["lo"], bounds["hi"])
        rows = conn.execute(
            f"SELECT * FROM feedback WHERE trained=1 AND rating!=0{filters} AND id>=? ORDER BY id ASC LIMIT ?",
            (*params, start, limit),
        ).fetchall()
        if len(rows) < limit:
            rows += conn.execute(
                f"SELECT * FROM feedback WHERE trained=1 AND rating!=0{filters} AND id<? ORDER BY id ASC LIMIT ?",
                (*params, start, limit - len(rows)),
            ).fetchall()
        return [dict(r) for r in rows]
    sql, params = _feedback_query("trained=1 AND rating!=0", "id ASC", limit, source, model)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def revise_feedback_rating(conn, feedback_id, rating):
    _validate_rating(rating, allow_delete=True)
    row = conn.execute("SELECT id, model, adapter_version FROM feedback WHERE id=?", (feedback_id,)).fetchone()
    with conn:
        _mark_feedback_mutation(conn, row)
        cur = (
            conn.execute("DELETE FROM feedback WHERE id=?", (feedback_id,))
            if rating == 0 else
            conn.execute("UPDATE feedback SET rating=?, trained=0, adapter_version=NULL WHERE id=?", (rating, feedback_id))
        )
    return cur.rowcount


def remove_last(conn, source=None, context=None, model=None):
    where = "rating!=0"
    params = []
    if source is not None:
        where += " AND source=?"
        params.append(source)
    if context is not None:
        where += " AND context=?"
        params.append(context)
    if model is not None:
        where += " AND model=?"
        params.append(model)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(f"SELECT * FROM feedback WHERE {where} ORDER BY id DESC LIMIT 1", params).fetchone()
        if row:
            conn.execute("DELETE FROM feedback WHERE id=?", (row["id"],))
            _mark_feedback_mutation(conn, row)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return dict(row) if row else None


def _delete_oldest(conn, where, limit):
    if not limit or limit <= 0:
        return 0
    return conn.execute(
        f"DELETE FROM feedback WHERE id IN (SELECT id FROM feedback WHERE {where} ORDER BY id ASC LIMIT ?)",
        (int(limit),),
    ).rowcount


def prune_feedback(conn, max_rows=MAX_FEEDBACK_ROWS, max_age_days=0, max_untrained_rows=0):
    """Prune old feedback without silently discarding trainable rows unless capped."""
    deleted = 0
    with conn:
        if max_age_days and max_age_days > 0:
            cur = conn.execute(
                "DELETE FROM feedback WHERE trained=1 AND created_at < datetime('now', ?)",
                (f"-{int(max_age_days)} days",),
            )
            deleted += cur.rowcount
        if max_rows and max_rows > 0:
            trained = conn.execute("SELECT COUNT(*) AS n FROM feedback WHERE trained=1").fetchone()["n"]
            deleted += _delete_oldest(conn, "trained=1", max(0, trained - int(max_rows)))
        if max_untrained_rows and max_untrained_rows > 0:
            untrained = conn.execute("SELECT COUNT(*) AS n FROM feedback WHERE trained=0 AND rating!=0").fetchone()["n"]
            deleted += _delete_oldest(conn, "trained=0 AND rating!=0", max(0, untrained - int(max_untrained_rows)))
    if deleted >= 100 and not conn.in_transaction:
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")
        except sqlite3.Error:
            pass
    return deleted


def recent(conn, limit=RECENT_LIMIT):
    """Last N ratings for history view."""
    rows = conn.execute(
        "SELECT id, prompt, rating, source, created_at FROM feedback "
        "WHERE rating!=0 ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def count(conn, model=None):
    filters, params = _feedback_filters(model=model)
    row = conn.execute(
        "SELECT COUNT(*) as total, "
        "COALESCE(SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END),0) as good, "
        "COALESCE(SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END),0) as bad, "
        "COALESCE(SUM(CASE WHEN trained=0 THEN 1 ELSE 0 END),0) as untrained "
        f"FROM feedback WHERE rating!=0{filters}",
        params,
    ).fetchone()
    return dict(row)


def count_trainable_untrained(conn, source=None, model=None):
    filters, params = _feedback_filters(source, model)
    row = conn.execute(f"SELECT COUNT(*) AS total FROM feedback WHERE trained=0 AND rating!=0{filters}", params).fetchone()
    return row["total"]


# -- ema ops --

def get_ema(conn, model=None):
    if model:
        r = conn.execute("SELECT reward_mean, count FROM model_ema_state WHERE model=?", (model,)).fetchone()
        return (r["reward_mean"], r["count"]) if r else (0.0, 0)
    r = conn.execute("SELECT reward_mean, count FROM ema_state WHERE id=1").fetchone()
    return r["reward_mean"], r["count"]


def set_ema(conn, mean, count, model=None):
    if model:
        conn.execute(
            "INSERT INTO model_ema_state (model, reward_mean, count) VALUES (?, ?, ?) "
            "ON CONFLICT(model) DO UPDATE SET reward_mean=excluded.reward_mean, count=excluded.count",
            (model, mean, count),
        )
    else:
        conn.execute("UPDATE ema_state SET reward_mean=?, count=? WHERE id=1", (mean, count))


def get_training_state(conn):
    row = conn.execute("SELECT state FROM training_state WHERE id=1").fetchone()
    if not row or not row["state"]:
        return None
    try:
        return json.loads(row["state"])
    except (TypeError, json.JSONDecodeError):
        clear_training_state(conn)
        return None


def save_training_state(conn, state):
    with conn:
        conn.execute("UPDATE training_state SET state=? WHERE id=1", (json.dumps(state),))


def clear_training_state(conn):
    with conn:
        conn.execute("UPDATE training_state SET state=NULL WHERE id=1")


def record_background_event(conn, kind, hour):
    if kind not in {"pressure", "success"}:
        raise ValueError("background event kind must be pressure or success")
    pressure, success = (1, 0) if kind == "pressure" else (0, 1)
    with conn:
        conn.execute(
            "INSERT INTO background_history (hour, pressure_count, success_count) VALUES (?, ?, ?) "
            "ON CONFLICT(hour) DO UPDATE SET "
            "pressure_count=pressure_count+excluded.pressure_count, "
            "success_count=success_count+excluded.success_count",
            (hour, pressure, success),
        )


def background_history(conn):
    rows = conn.execute("SELECT hour, pressure_count, success_count FROM background_history").fetchall()
    return {row["hour"]: dict(row) for row in rows}


def _adapter_metrics(row):
    try:
        return json.loads(row["metrics"]) if row["metrics"] else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _adapter_model(row):
    metrics = _adapter_metrics(row)
    if metrics.get("model"):
        return metrics["model"]
    path = _row_get(row, "path")
    if not path:
        return None
    try:
        adapter_dir = Path(path).expanduser()
        if adapter_dir.is_file():
            adapter_dir = adapter_dir.parent
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        return cfg.get("base_model") or cfg.get("base_model_name_or_path")
    except (OSError, TypeError, json.JSONDecodeError):
        return None


def _adapter_matches_model(row, model):
    return model is None or _adapter_model(row) == model


def _adapter_versions(conn, model=None, *, statuses=None, after=None):
    sql, params = "SELECT version, path, metrics FROM adapters WHERE 1=1", []
    if statuses:
        sql += " AND status IN (" + ",".join("?" for _ in statuses) + ")"
        params.extend(statuses)
    if after is not None:
        sql += " AND version>?"
        params.append(after)
    return [r["version"] for r in conn.execute(sql, params).fetchall() if _adapter_matches_model(r, model)]


def _update_versions(conn, sql_prefix, versions):
    versions = tuple(int(v) for v in versions)
    if not versions:
        return 0
    ph = ",".join("?" for _ in versions)
    return conn.execute(f"{sql_prefix} WHERE version IN ({ph})", versions).rowcount


def record_training_round(conn, mean, count, version, path, parent=None, metrics=None, feedback_ids=None, clear_state=False):
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
        if clear_state:
            conn.execute("UPDATE training_state SET state=NULL WHERE id=1")


# -- adapter ops --

def list_adapters(conn, model=None):
    rows = conn.execute(
        "SELECT version, path, status, created_at, metrics FROM adapters ORDER BY version DESC"
    ).fetchall()
    return [dict(r) for r in rows if _adapter_matches_model(r, model)]


def latest_adapter(conn, model=None):
    rows = conn.execute("SELECT * FROM adapters WHERE status='active' ORDER BY version DESC").fetchall()
    return next((dict(r) for r in rows if _adapter_matches_model(r, model)), None)


def can_activate_adapter(conn, version):
    row = conn.execute("SELECT status, parent_version FROM adapters WHERE version=?", (version,)).fetchone()
    if not row or row["status"] in {"rejected", "stale"}:
        return False
    parent = row["parent_version"]
    if parent is None:
        return True
    parent_row = conn.execute("SELECT status FROM adapters WHERE version=?", (parent,)).fetchone()
    return bool(parent_row and parent_row["status"] not in {"rejected", "stale"})


def activate_training_round(conn, version, mean=None, count=None, feedback_ids=None, *, max_rows=MAX_FEEDBACK_ROWS, max_age_days=0, max_untrained_rows=0, model=None):
    if not can_activate_adapter(conn, version):
        return None
    row = conn.execute("SELECT path, metrics, status FROM adapters WHERE version=?", (version,)).fetchone()
    metrics = _adapter_metrics(row)
    actual_model = metrics.get("model") or _adapter_model(row)
    if model and actual_model and actual_model != model:
        return None
    model = model or actual_model
    if model is None:
        active_rows = conn.execute("SELECT path, metrics FROM adapters WHERE status='active'").fetchall()
        if any(_adapter_model(active) is not None for active in active_rows):
            return None
    if mean is None:
        mean = metrics.get("ema_mean")
    if count is None:
        count = metrics.get("ema_count")
    feedback_ids = list(feedback_ids or metrics.get("feedback_ids") or [])
    feedback_ids = tuple(int(i) for i in feedback_ids)
    if feedback_ids and model:
        ph = ",".join("?" for _ in feedback_ids)
        rows = conn.execute(f"SELECT id, model FROM feedback WHERE id IN ({ph})", feedback_ids).fetchall()
        allowed = {row["id"] for row in rows if row["model"] == model}
        feedback_ids = tuple(i for i in feedback_ids if i in allowed)
    with conn:
        if mean is not None and count is not None:
            set_ema(conn, mean, count, model)
        _update_versions(conn, "UPDATE adapters SET status='inactive'", _adapter_versions(conn, model, statuses=("active",)))
        conn.execute("UPDATE adapters SET status='active' WHERE version=?", (version,))
        if feedback_ids:
            ph = ",".join("?" for _ in feedback_ids)
            conn.execute(
                f"UPDATE feedback SET trained=1, adapter_version=? WHERE id IN ({ph})",
                (version, *feedback_ids),
            )
    prune_feedback(conn, max_rows=max_rows, max_age_days=max_age_days, max_untrained_rows=max_untrained_rows)
    return latest_adapter(conn, model)


def reject_adapter(conn, version):
    with conn:
        conn.execute("UPDATE adapters SET status='rejected' WHERE version=?", (version,))


def rollback_to(conn, version, model=None):
    row = conn.execute(
        "SELECT version, path, metrics FROM adapters WHERE version=? AND status IN ('active','inactive','rolled_back')",
        (version,),
    ).fetchone()
    if not row or not _adapter_matches_model(row, model):
        return None
    model = model or _adapter_model(row)
    metrics = _adapter_metrics(row)
    affected = _adapter_versions(conn, model, after=version)
    with conn:
        _update_versions(conn, "UPDATE adapters SET status='inactive'", _adapter_versions(conn, model, statuses=("active",)))
        _update_versions(conn, "UPDATE adapters SET status='rolled_back'", affected)
        conn.execute("UPDATE adapters SET status='active' WHERE version = ?", (version,))
        if affected:
            ph = ",".join("?" for _ in affected)
            conn.execute(f"UPDATE feedback SET trained=0, adapter_version=NULL WHERE adapter_version IN ({ph})", tuple(affected))
        conn.execute("UPDATE training_state SET state=NULL WHERE id=1")
        if metrics.get("ema_mean") is not None and metrics.get("ema_count") is not None:
            set_ema(conn, metrics["ema_mean"], metrics["ema_count"], model)
    return latest_adapter(conn, model)


def rollback(conn, model=None):
    latest = latest_adapter(conn, model)
    if not latest:
        return None
    previous = next(
        (row for row in conn.execute("SELECT version, path, metrics FROM adapters WHERE status IN ('inactive','rolled_back') ORDER BY version DESC").fetchall()
         if row["version"] < latest["version"] and _adapter_matches_model(row, model)),
        None,
    )
    if not previous:
        return None
    with conn:
        conn.execute("UPDATE adapters SET status='rolled_back' WHERE version=?", (latest["version"],))
        conn.execute("UPDATE adapters SET status='active' WHERE version = ?", (previous["version"],))
        conn.execute("UPDATE feedback SET trained=0, adapter_version=NULL WHERE adapter_version=?", (latest["version"],))
        conn.execute("UPDATE training_state SET state=NULL WHERE id=1")
        metrics = _adapter_metrics(previous)
        if metrics.get("ema_mean") is not None and metrics.get("ema_count") is not None:
            set_ema(conn, metrics["ema_mean"], metrics["ema_count"], model or _adapter_model(previous))
    return latest_adapter(conn, model)


def cleanup_adapters(conn, keep=20, model=None):
    rows = conn.execute(
        "SELECT version, path, metrics FROM adapters "
        "WHERE status IN ('inactive', 'rejected', 'rolled_back', 'stale') "
        "ORDER BY version DESC",
    ).fetchall()
    rows = [r for r in rows if _adapter_matches_model(r, model)]
    return [(r["version"], r["path"]) for r in rows[max(0, int(keep)):]]


def delete_adapter(conn, version):
    with conn:
        conn.execute("DELETE FROM adapters WHERE version=?", (version,))


# -- nuclear option --

def reset_all(conn):
    with conn:
        conn.execute("DELETE FROM feedback")
        conn.execute("DELETE FROM adapters")
        conn.execute("DELETE FROM model_ema_state")
        conn.execute("DELETE FROM background_history")
        conn.execute("UPDATE ema_state SET reward_mean=0.0, count=0 WHERE id=1")
        conn.execute("UPDATE training_state SET state=NULL WHERE id=1")
    # Checkpoint after the transaction so SQLite can truncate WAL/SHM files.
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("VACUUM")
