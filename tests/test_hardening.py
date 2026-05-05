import os
import io
import json
import struct
import sqlite3
import argparse
import sys

import pytest

from reinforceclaw import backend_cuda, cli, db, scheduler, trainer
from reinforceclaw import profile as profile_mod
from reinforceclaw.hooks import _common, claude_code, openclaw


def test_read_stdin_enforces_raw_byte_limit(monkeypatch):
    monkeypatch.setattr(_common, "MAX_HOOK_STDIN_BYTES", 8)
    monkeypatch.setattr(sys, "stdin", type("FakeStdin", (), {"buffer": io.BytesIO(b'{"x":1}  ')})())
    assert _common.read_stdin() == {}


def test_db_rejects_broken_symlink(tmp_path):
    path = tmp_path / "broken.db"
    path.symlink_to(tmp_path / "missing.db")
    with pytest.raises(PermissionError):
        db.connect(path)


def test_connect_reinitializes_replaced_db_same_process(tmp_path):
    path = tmp_path / "rc.db"
    db.connect(path).close()
    path.unlink()
    (tmp_path / "rc.db-wal").unlink(missing_ok=True)
    (tmp_path / "rc.db-shm").unlink(missing_ok=True)
    path.write_bytes(b"")
    conn = db.connect(path)
    try:
        assert db.count(conn)["total"] == 0
    finally:
        conn.close()


def test_prune_preserves_untrained_feedback(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    ids = [db.add_feedback(conn, "m", f"p{i}", f"r{i}", 1, source="s") for i in range(10)]
    db.record_training_round(conn, 0.0, 0, 1, str(tmp_path / "adapter.safetensors"), feedback_ids=ids[:3], clear_state=True)
    db.activate_training_round(conn, 1, 0.0, 0, ids[:3], max_rows=5)
    rows = {row["trained"]: row["n"] for row in conn.execute("SELECT trained, COUNT(*) n FROM feedback GROUP BY trained")}
    assert rows.get(0, 0) == 7
    assert rows.get(1, 0) <= 3


def test_prune_can_cap_untrained_feedback(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    for i in range(8):
        db.add_feedback(conn, "m", f"p{i}", f"r{i}", 1, source="s")
    db.prune_feedback(conn, max_rows=100, max_untrained_rows=3)
    assert db.count_trainable_untrained(conn, source="s") == 3


def test_migration_dedupes_duplicate_event_ids(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            context TEXT,
            rollout_context TEXT,
            event_id TEXT,
            rating INTEGER NOT NULL,
            source TEXT NOT NULL,
            trained INTEGER DEFAULT 0,
            adapter_version INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE adapters (
            version INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            parent_version INTEGER,
            status TEXT DEFAULT 'inactive',
            metrics TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE ema_state (id INTEGER PRIMARY KEY CHECK(id=1), reward_mean REAL DEFAULT 0.0, count INTEGER DEFAULT 0);
        CREATE TABLE training_state (id INTEGER PRIMARY KEY CHECK(id=1), state TEXT);
        CREATE TABLE background_history (hour INTEGER PRIMARY KEY, pressure_count INTEGER DEFAULT 0, success_count INTEGER DEFAULT 0);
        CREATE TABLE migrations (name TEXT PRIMARY KEY, applied_at TEXT DEFAULT CURRENT_TIMESTAMP);
    """)
    rows = [
        ("m", "p1", "r1", "dup", 1, "s"),
        ("m", "p2", "r2", "dup", -1, "s"),
    ]
    conn.executemany("INSERT INTO feedback (model, prompt, response, event_id, rating, source) VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    conn = db.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM feedback WHERE event_id='dup'").fetchone()[0] == 1
    finally:
        conn.close()


def test_migration_preserves_legacy_eventless_duplicate_content(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            context TEXT,
            rollout_context TEXT,
            event_id TEXT,
            rating INTEGER NOT NULL,
            source TEXT NOT NULL,
            trained INTEGER DEFAULT 0,
            adapter_version INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE adapters (
            version INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            parent_version INTEGER,
            status TEXT DEFAULT 'inactive',
            metrics TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE ema_state (id INTEGER PRIMARY KEY CHECK(id=1), reward_mean REAL DEFAULT 0.0, count INTEGER DEFAULT 0);
        CREATE TABLE training_state (id INTEGER PRIMARY KEY CHECK(id=1), state TEXT);
        CREATE TABLE background_history (hour INTEGER PRIMARY KEY, pressure_count INTEGER DEFAULT 0, success_count INTEGER DEFAULT 0);
        CREATE TABLE migrations (name TEXT PRIMARY KEY, applied_at TEXT DEFAULT CURRENT_TIMESTAMP);
    """)
    rows = [
        ("m", "same prompt", "same response", None, 1, "s"),
        ("m", "same prompt", "same response", None, 1, "s"),
    ]
    conn.executemany("INSERT INTO feedback (model, prompt, response, event_id, rating, source) VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    conn = db.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM feedback WHERE event_id IS NULL").fetchone()[0] == 2
    finally:
        conn.close()


def test_pending_key_is_secret_and_context_scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(_common, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(_common, "PENDING_LOCK_PATH", tmp_path / "pending.lock")
    key = _common.save_pending("claude_code", "m", "p", "r", context="ctx")
    assert len(key) == 32
    int(key, 16)
    assert _common.pop_pending("claude_code", key, context="other") is None
    assert _common.pop_pending("claude_code", key, context="ctx")["response"] == "r"


def test_restore_pending_recreates_consumed_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(_common, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(_common, "PENDING_LOCK_PATH", tmp_path / "pending.lock")
    key = _common.save_pending("codex", "m", "p", "r", context="ctx")
    payload = _common.pop_pending("codex", key, context="ctx")
    assert _common.pop_pending("codex", key, context="ctx") is None
    _common.restore_pending(payload)
    assert _common.pop_pending("codex", key, context="ctx")["response"] == "r"


def test_openclaw_command_pending_key_survives_prompt_session_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "rc.db")
    monkeypatch.setattr(_common, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(_common, "PENDING_LOCK_PATH", tmp_path / "pending.lock")
    monkeypatch.setattr(openclaw, "_cfg", lambda: {"model": "m"})
    monkeypatch.setattr(openclaw, "maybe_train", lambda *_args, **_kwargs: None)
    key = _common.save_pending("openclaw", "m", "p", "r", context="actual-session")
    assert openclaw._run_command("good", "button-session", key)["ok"] is True
    conn = db.connect(tmp_path / "rc.db")
    try:
        row = conn.execute("SELECT context, event_id, rating FROM feedback").fetchone()
        assert (row["context"], row["event_id"], row["rating"]) == ("actual-session", key, 1)
    finally:
        conn.close()


def test_restore_pending_truncates_oversized_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(_common, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(_common, "PENDING_LOCK_PATH", tmp_path / "pending.lock")
    payload = {"key": "abc", "source": "codex", "model": "m", "prompt": "p", "response": "x" * (db.MAX_TEXT_BYTES + 1)}
    _common.restore_pending(payload)
    restored = _common.pop_pending("codex", "abc")
    assert db.TRUNCATION_MARKER in restored["response"]
    assert len(restored["response"].encode("utf-8")) <= db.MAX_TEXT_BYTES


def test_pop_pending_rejects_payload_context_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(_common, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(_common, "PENDING_LOCK_PATH", tmp_path / "pending.lock")
    key = _common.save_pending("codex", "m", "p", "r", context="ctx")
    path = _common._pending_path("codex", "ctx", key)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["context"] = "other"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _common.pop_pending("codex", key, context="ctx") is None


def test_pop_pending_refuses_symlink_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(_common, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(_common, "PENDING_LOCK_PATH", tmp_path / "pending.lock")
    _common.PENDING_DIR.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"key": "abc", "source": "codex", "response": "secret"}), encoding="utf-8")
    _common._pending_path("codex", key="abc").symlink_to(outside)
    assert _common.pop_pending("codex", "abc") is None


def test_cmd_rate_restores_pending_if_add_feedback_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(_common, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(_common, "PENDING_LOCK_PATH", tmp_path / "pending.lock")
    monkeypatch.setattr(cli, "_load_model_cfg", lambda: {"model": "m"})
    monkeypatch.setattr(cli, "pop_pending", _common.pop_pending)
    monkeypatch.setattr(cli, "restore_pending", _common.restore_pending)
    key = _common.save_pending("codex", "m", "p", "r")

    def boom(*_args, **_kwargs):
        raise sqlite3.OperationalError("db locked")

    monkeypatch.setattr(db, "add_feedback", boom)
    with pytest.raises(sqlite3.OperationalError):
        cli.cmd_rate(argparse.Namespace(), 1)
    assert _common.pop_pending("codex", key)["response"] == "r"


def test_cmd_rate_uses_newest_pending_across_agents(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(_common, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(_common, "PENDING_LOCK_PATH", tmp_path / "pending.lock")
    monkeypatch.setattr(cli, "pop_pending", _common.pop_pending)
    monkeypatch.setattr(cli, "restore_pending", _common.restore_pending)
    _common.save_pending("claude_code", "m", "old", "old")
    key = _common.save_pending("codex", "m", "new", "new")
    picked = cli._pop_newest_pending("claude_code", "codex")
    assert picked["key"] == key
    assert _common.pop_pending("claude_code")["response"] == "old"


def test_claude_panel_consumes_pending_before_training_queue_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "rc.db")
    monkeypatch.setattr(_common, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(_common, "PENDING_LOCK_PATH", tmp_path / "pending.lock")
    monkeypatch.setattr(claude_code, "load_config", lambda: {"model": "m", "panel_enabled": True})
    monkeypatch.setattr(claude_code, "read_stdin", lambda: {"last_assistant_message": "r"})
    monkeypatch.setattr(claude_code, "last_msg_from", lambda _data, role: "p" if role == "user" else "r")
    monkeypatch.setattr(claude_code, "pending_context", lambda _data: "ctx")
    monkeypatch.setattr(claude_code, "training_context", lambda *_args, **_kwargs: "{}")
    monkeypatch.setattr("reinforceclaw.feedback.collect_rating", lambda: 1)
    monkeypatch.setattr(claude_code, "maybe_train", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("queue failed")))
    claude_code.handle_stop()
    conn = db.connect(tmp_path / "rc.db")
    try:
        assert db.count(conn)["total"] == 1
    finally:
        conn.close()
    assert _common.pop_pending("claude_code", context="ctx") is None


def test_ignore_feedback_marks_adapter_stale(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    fid = db.add_feedback(conn, "m", "p", "r", 1, source="s")
    db.record_training_round(conn, 0.0, 0, 1, str(tmp_path / "adapter.safetensors"), feedback_ids=[fid], clear_state=True)
    db.activate_training_round(conn, 1, 0.0, 0, [fid])
    assert db.ignore_feedback_ids(conn, [fid]) == 1
    assert db.latest_adapter(conn) is None
    row = conn.execute("SELECT rating, trained, adapter_version FROM feedback WHERE id=?", (fid,)).fetchone()
    assert (row["rating"], row["trained"], row["adapter_version"]) == (0, 1, None)


def test_feedback_mutation_stales_inactive_dependent_adapters(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    fid = db.add_feedback(conn, "m", "p", "r", 1, source="s")
    for version in (1, 2):
        db.record_training_round(conn, 0.0, 0, version, str(tmp_path / f"a{version}.safetensors"), feedback_ids=[fid], clear_state=True)
        db.activate_training_round(conn, version, 0.0, 0, [fid])
    db.revise_feedback_rating(conn, fid, -1)
    statuses = {row["version"]: row["status"] for row in db.list_adapters(conn)}
    assert statuses == {1: "stale", 2: "stale"}


def test_cleanup_adapters_includes_rolled_back(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    for version in (1, 2, 3):
        db.record_training_round(conn, 0.0, 0, version, str(tmp_path / f"a{version}.safetensors"), clear_state=True)
        db.activate_training_round(conn, version, 0.0, 0, [])
    db.rollback_to(conn, 1)
    assert {version for version, _path in db.cleanup_adapters(conn, keep=0)} == {2, 3}


def test_latest_adapter_is_model_scoped(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    db.record_training_round(conn, 0.0, 0, 1, str(tmp_path / "old.safetensors"), metrics={"model": "qwen-3.5"}, clear_state=True)
    db.activate_training_round(conn, 1, 0.0, 0, [])
    assert db.latest_adapter(conn, model="qwen-3.5")["version"] == 1
    assert db.latest_adapter(conn, model="qwen-3.6") is None


def test_rollback_refuses_other_model_adapter(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    db.record_training_round(conn, 0.0, 0, 1, str(tmp_path / "old.safetensors"), metrics={"model": "qwen-3.5"}, clear_state=True)
    db.activate_training_round(conn, 1, 0.0, 0, [])
    db.record_training_round(conn, 0.0, 0, 2, str(tmp_path / "new.safetensors"), metrics={"model": "qwen-3.6"}, clear_state=True)
    db.activate_training_round(conn, 2, 0.0, 0, [])
    assert db.rollback_to(conn, 1, model="qwen-3.6") is None
    assert db.latest_adapter(conn, model="qwen-3.6")["version"] == 2


def test_training_queries_are_model_scoped(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    db.add_feedback(conn, "old", "p1", "r1", 1, source="s")
    db.add_feedback(conn, "new", "p2", "r2", -1, source="s")
    assert db.count_trainable_untrained(conn, source="s", model="new") == 1
    assert [r["model"] for r in db.get_untrained(conn, source="s", model="new")] == ["new"]


def test_ema_is_model_scoped(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    with conn:
        db.set_ema(conn, 0.7, 12, "old")
        db.set_ema(conn, -0.2, 3, "new")
    assert db.get_ema(conn, "old") == (0.7, 12)
    assert db.get_ema(conn, "new") == (-0.2, 3)
    assert db.get_ema(conn, "missing") == (0.0, 0)


def test_activation_keeps_other_model_active(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    db.record_training_round(conn, 0.1, 1, 1, str(tmp_path / "old.safetensors"), metrics={"model": "old"}, clear_state=True)
    db.activate_training_round(conn, 1, 0.1, 1, [])
    db.record_training_round(conn, 0.2, 2, 2, str(tmp_path / "new.safetensors"), metrics={"model": "new"}, clear_state=True)
    db.activate_training_round(conn, 2, 0.2, 2, [])
    assert db.latest_adapter(conn, "old")["version"] == 1
    assert db.latest_adapter(conn, "new")["version"] == 2


def test_activation_marks_only_same_model_feedback(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    old_id = db.add_feedback(conn, "old", "p1", "r1", 1)
    new_id = db.add_feedback(conn, "new", "p2", "r2", 1)
    db.record_training_round(
        conn, 0.1, 1, 1, str(tmp_path / "old.safetensors"),
        metrics={"model": "old"}, feedback_ids=[old_id, new_id], clear_state=True,
    )
    db.activate_training_round(conn, 1)
    rows = {
        row["id"]: row["trained"]
        for row in conn.execute("SELECT id, trained FROM feedback ORDER BY id")
    }
    assert rows == {old_id: 1, new_id: 0}


def test_rollback_to_is_model_scoped(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    old_id = db.add_feedback(conn, "old", "p1", "r1", 1)
    new_id = db.add_feedback(conn, "new", "p2", "r2", 1)
    db.record_training_round(conn, 0.1, 1, 1, str(tmp_path / "old1.safetensors"), metrics={"model": "old"}, feedback_ids=[old_id], clear_state=True)
    db.activate_training_round(conn, 1, 0.1, 1, [old_id])
    db.record_training_round(conn, 0.2, 2, 2, str(tmp_path / "new1.safetensors"), metrics={"model": "new"}, feedback_ids=[new_id], clear_state=True)
    db.activate_training_round(conn, 2, 0.2, 2, [new_id])
    db.record_training_round(conn, 0.3, 3, 3, str(tmp_path / "old2.safetensors"), metrics={"model": "old"}, feedback_ids=[old_id], clear_state=True)
    db.activate_training_round(conn, 3, 0.3, 3, [old_id])
    assert db.rollback_to(conn, 1, model="old")["version"] == 1
    assert db.latest_adapter(conn, "new")["version"] == 2
    assert conn.execute("SELECT trained, adapter_version FROM feedback WHERE id=?", (new_id,)).fetchone()["adapter_version"] == 2


def test_plain_rollback_does_not_roll_forward_to_rolled_back_adapter(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    for version in (1, 2, 3):
        db.record_training_round(conn, 0.0, 0, version, str(tmp_path / f"a{version}.safetensors"), metrics={"model": "m"}, clear_state=True)
        db.activate_training_round(conn, version, 0.0, 0, [])
    db.rollback_to(conn, 1, model="m")
    assert db.rollback(conn, model="m") is None
    assert db.latest_adapter(conn, "m")["version"] == 1


def test_feedback_mutation_stales_only_same_model_version_chain(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    old_id = db.add_feedback(conn, "old", "p1", "r1", 1)
    new_id = db.add_feedback(conn, "new", "p2", "r2", 1)
    db.record_training_round(conn, 0.0, 0, 1, str(tmp_path / "old.safetensors"), metrics={"model": "old"}, feedback_ids=[old_id], clear_state=True)
    db.activate_training_round(conn, 1, 0.0, 0, [old_id])
    db.record_training_round(conn, 0.0, 0, 2, str(tmp_path / "new.safetensors"), metrics={"model": "new"}, feedback_ids=[new_id], clear_state=True)
    db.activate_training_round(conn, 2, 0.0, 0, [new_id])
    db.revise_feedback_rating(conn, old_id, -1)
    assert db.latest_adapter(conn, "new")["version"] == 2
    assert db.latest_adapter(conn, "old") is None


def test_cleanup_adapters_is_model_scoped(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    for version, model in ((1, "old"), (2, "new"), (3, "old"), (4, "new")):
        db.record_training_round(conn, 0.0, 0, version, str(tmp_path / f"a{version}.safetensors"), metrics={"model": model}, clear_state=True)
        db.activate_training_round(conn, version, 0.0, 0, [])
    assert [version for version, _ in db.cleanup_adapters(conn, keep=0, model="old")] == [1]
    assert [version for version, _ in db.cleanup_adapters(conn, keep=0, model="new")] == [2]


def test_cleanup_adapters_clamps_negative_keep(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    for version in (1, 2, 3):
        db.record_training_round(conn, 0.0, 0, version, str(tmp_path / f"a{version}.safetensors"), clear_state=True)
        db.activate_training_round(conn, version, 0.0, 0, [])
    assert {version for version, _path in db.cleanup_adapters(conn, keep=-1)} == {1, 2}


def test_legacy_adapter_model_detected_from_adapter_config(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    adapter_dir = tmp_path / "v1"
    adapter_dir.mkdir()
    adapter = adapter_dir / "adapter_model.safetensors"
    adapter.write_bytes(b"x")
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "legacy-model"}), encoding="utf-8")
    db.record_training_round(conn, 0.0, 0, 1, str(adapter), metrics={}, clear_state=True)
    db.activate_training_round(conn, 1, 0.0, 0, [])
    assert db.latest_adapter(conn, model="legacy-model")["version"] == 1


def test_remove_last_uses_compatible_transaction(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    first = db.add_feedback(conn, "m", "p1", "r1", 1, source="s")
    second = db.add_feedback(conn, "m", "p2", "r2", -1, source="s")
    removed = db.remove_last(conn, source="s")
    assert removed["id"] == second
    assert [row["id"] for row in conn.execute("SELECT id FROM feedback ORDER BY id")] == [first]


def test_remove_last_can_be_model_scoped(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    old_id = db.add_feedback(conn, "old", "p1", "r1", 1)
    new_id = db.add_feedback(conn, "new", "p2", "r2", -1)
    removed = db.remove_last(conn, model="old")
    assert removed["id"] == old_id
    assert [row["id"] for row in conn.execute("SELECT id FROM feedback ORDER BY id")] == [new_id]


def test_background_event_upsert_updates_both_counters(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    db.record_background_event(conn, "pressure", 4)
    db.record_background_event(conn, "success", 4)
    row = db.background_history(conn)[4]
    assert (row["pressure_count"], row["success_count"]) == (1, 1)


def test_corrupt_training_state_is_cleared(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    conn.execute("UPDATE training_state SET state='{' WHERE id=1")
    conn.commit()
    assert db.get_training_state(conn) is None
    assert conn.execute("SELECT state FROM training_state WHERE id=1").fetchone()["state"] is None


def test_corrupt_adapter_metrics_do_not_crash_activation(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    conn.execute(
        "INSERT INTO adapters(version, path, status, metrics) VALUES(1, ?, 'candidate', '{')",
        (str(tmp_path / "adapter.safetensors"),),
    )
    conn.commit()
    assert db.activate_training_round(conn, 1)["version"] == 1


def test_unknown_model_candidate_does_not_inactivate_other_models(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    db.record_training_round(conn, 0.0, 1, 1, str(tmp_path / "v1" / "adapter_model.safetensors"), metrics={"model": "m1"})
    assert db.activate_training_round(conn, 1, model="m1")["version"] == 1
    db.record_training_round(conn, 0.0, 1, 2, str(tmp_path / "v2" / "adapter_model.safetensors"), metrics={})
    assert db.activate_training_round(conn, 2) is None
    assert db.latest_adapter(conn, model="m1")["version"] == 1


def test_redacts_github_fine_grained_pat():
    assert "github_pat_" not in db.redact_secrets("token=github_pat_" + "A" * 40)


def test_redacts_json_style_secret_values():
    text = db.redact_secrets('{"api_key": "sk-proj-' + "A" * 32 + '", "token": "hf_' + "B" * 20 + '"}')
    assert "sk-proj-" not in text
    assert "hf_" not in text


def test_redacts_common_chat_pasted_secrets():
    samples = [
        "AIza" + "A" * 35,
        "https://hooks.slack.com/services/T000/B000/" + "A" * 24,
        "123456789:" + "A" * 35,
        "mfa." + "A" * 32,
        "eyJ" + "A" * 20 + "." + "B" * 20 + "." + "C" * 20,
        "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----",
    ]
    text = db.redact_secrets("\n".join(samples))
    assert "AIza" not in text
    assert "hooks.slack.com" not in text
    assert "123456789:" not in text
    assert "mfa." not in text
    assert "eyJ" not in text
    assert "PRIVATE KEY" not in text


def test_feedback_caps_untrusted_source_and_event_id(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    fid = db.add_feedback(conn, "m", "p", "r", 1, source="x" * 1000, event_id="e" * 2000)
    row = conn.execute("SELECT source, event_id FROM feedback WHERE id=?", (fid,)).fetchone()
    assert len(row["source"].encode("utf-8")) <= db.MAX_SOURCE_BYTES
    assert len(row["event_id"].encode("utf-8")) <= db.MAX_EVENT_ID_BYTES


def test_cli_atomic_write_preserves_symlink(tmp_path):
    target = tmp_path / "target.json"
    link = tmp_path / "config.json"
    target.write_text("{}\n", encoding="utf-8")
    link.symlink_to(target)
    cli._write_json_atomic(link, {"ok": True})
    assert link.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


def test_cli_private_config_refuses_symlink(tmp_path, monkeypatch):
    home = tmp_path / "home"
    private = home / ".reinforceclaw"
    private.mkdir(parents=True)
    target = tmp_path / "outside.json"
    target.write_text("{}\n", encoding="utf-8")
    link = private / "config.json"
    link.symlink_to(target)
    monkeypatch.setattr(cli, "CONFIG_PATH", link)
    with pytest.raises(PermissionError):
        cli.save_config({"openclaw_secret": "x" * 32})
    with pytest.raises(PermissionError):
        cli.load_config()


def test_cli_atomic_backup_is_private(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"secret":"value"}\n', encoding="utf-8")
    path.chmod(0o644)
    cli._write_json_atomic(path, {"ok": True}, backup=True)
    assert (path.parent / "config.json.bak").stat().st_mode & 0o777 == 0o600


def test_cli_atomic_backup_refuses_symlink_backup(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}\n", encoding="utf-8")
    backup_target = tmp_path / "backup-target"
    (tmp_path / "config.json.bak").symlink_to(backup_target)
    with pytest.raises(PermissionError):
        cli._write_json_atomic(path, {"ok": True}, backup=True)


def test_install_json_hooks_refuses_malformed_config(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        cli._install_json_hooks(path, "codex.py", "python codex.py")
    assert path.read_text(encoding="utf-8") == "{not json"


def test_scheduler_write_preserves_symlink(tmp_path):
    target = tmp_path / "target.service"
    link = tmp_path / "unit.service"
    target.write_text("old", encoding="utf-8")
    link.symlink_to(target)
    scheduler._write_text(link, "new")
    assert link.is_symlink()
    assert target.read_text(encoding="utf-8") == "new"


def test_scheduler_run_ok_times_out(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise scheduler.subprocess.TimeoutExpired(["systemctl"], 30)

    monkeypatch.setattr(scheduler.subprocess, "run", timeout)
    assert scheduler._run_ok(["systemctl"]) is False
    assert scheduler.LAST_ERROR == "systemctl timed out"


def test_save_pending_truncates_oversized_text(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(_common, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(_common, "PENDING_LOCK_PATH", tmp_path / "pending.lock")
    key = _common.save_pending("claude_code", "m", "p", "x" * (db.MAX_TEXT_BYTES + 1))
    restored = _common.pop_pending("claude_code", key)
    assert db.TRUNCATION_MARKER in restored["response"]
    assert len(restored["response"].encode("utf-8")) <= db.MAX_TEXT_BYTES


def test_save_pending_scrubs_context(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(_common, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(_common, "PENDING_LOCK_PATH", tmp_path / "pending.lock")
    key = _common.save_pending("codex", "m", "p", "r", context='{"api_key":"sk-proj-' + "A" * 32 + '"}')
    restored = _common.pop_pending("codex", key)
    assert "sk-proj-" not in restored["context"]


def test_pending_prune_refuses_symlink_pending_dir(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "codex.json").write_text("secret", encoding="utf-8")
    link = tmp_path / "pending"
    link.symlink_to(outside)
    monkeypatch.setattr(_common, "PENDING_DIR", link)
    monkeypatch.setattr(_common, "PENDING_LOCK_PATH", tmp_path / "pending.lock")
    _common.prune_pending()
    assert (outside / "codex.json").exists()


def test_publish_gate_rejects_all_zero_safetensors(tmp_path):
    path = tmp_path / "adapter_model.safetensors"
    header = json.dumps({"lora": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")
    assert trainer.publish_gate({}, path)["reason"] == "adapter_all_zero"


def test_publish_gate_rejects_oversized_safetensors_header(tmp_path):
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(struct.pack("<Q", 2_000_000) + b"{}")
    assert trainer.publish_gate({}, path)["reason"] == "adapter_safetensors_bad_header"


def test_publish_gate_rejects_nondict_safetensors_header(tmp_path):
    path = tmp_path / "adapter_model.safetensors"
    header = json.dumps([]).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"x")
    assert trainer.publish_gate({}, path)["reason"] == "adapter_safetensors_invalid"


def test_publish_gate_rejects_nan_safetensors(tmp_path):
    path = tmp_path / "adapter_model.safetensors"
    header = json.dumps({"lora": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", float("nan")))
    assert trainer.publish_gate({}, path)["reason"] == "adapter_nonfinite_tensor"


def test_response_start_uses_common_token_prefix():
    assert trainer._response_start_from_prefix([1, 2, 3], [1, 2, 9, 4], 0) == 2
    assert trainer._response_start_from_prefix([1, 2, 3], [1, 2, 3, 4], 2) == 1


def test_retry_config_preserves_token_clip_but_drops_secrets():
    cfg = {"token_clip": [0.7, 1.3], "openclaw_secret": "secret", "HF_TOKEN": "hf_secret"}
    sanitized = trainer._sanitize_retry_config(cfg)
    assert sanitized == {"token_clip": [0.7, 1.3]}


def test_micro_indices_shuffle_without_repeating_until_epoch_exhausted():
    indices = trainer._micro_indices(4, 2, 2)
    assert sorted(indices) == [0, 1, 2, 3]
    longer = trainer._micro_indices(2, 2, 2)
    assert len(longer) == 4
    assert sorted(longer[:2]) == [0, 1]
    assert sorted(longer[2:]) == [0, 1]


def test_fused_adamw_version_parser_accepts_prerelease():
    class Cuda:
        @staticmethod
        def is_available():
            return True

    torch = type("Torch", (), {"__version__": "2.1.0a0", "cuda": Cuda})()
    assert trainer._fused_adamw_supported(torch) is True


def test_profile_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(profile_mod, "_DETECT_CACHE_MAX", 2)
    profile_mod._DETECT_CACHE.clear()
    for name in ("gpt-4-a", "gpt-4-b", "gpt-4-c"):
        profile_mod.detect(name)
    assert len(profile_mod._DETECT_CACHE) == 2


def test_profile_detect_tolerates_bad_numeric_config(tmp_path):
    model_dir = tmp_path / "bad-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({
            "model_type": "bad",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "vocab_size": 10,
            "num_experts": "many",
        }),
        encoding="utf-8",
    )
    prof = profile_mod.detect(str(model_dir))
    assert prof.kind == "moe"
    assert prof.total_b > 0.0


def test_cli_profile_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(cli, "_PROFILE_CACHE_MAX", 2)
    monkeypatch.setattr(cli.profile, "detect", lambda name: profile_mod.ModelProfile("cloud", name, "unknown", 0.0, 0.0, "cloud-test", False, "test"))
    cli._PROFILE_CACHE.clear()
    for name in ("a", "b", "c"):
        cli._auto_tuned_values({"model": name, "preset": "balanced"})
    assert len(cli._PROFILE_CACHE) == 2


def test_sensitive_env_strips_github_pat(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("GITHUB_PAT", "secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://user:pass@example")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@example/db")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/sock")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = trainer._child_env()
    assert env["PATH"] == "/usr/bin"
    assert "GITHUB_PAT" not in env
    assert "HTTPS_PROXY" not in env
    assert "DATABASE_URL" not in env
    assert "SSH_AUTH_SOCK" not in env


def test_hook_base_env_keeps_path_and_drops_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("GITHUB_PAT", "secret")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@example/db")
    monkeypatch.setattr(_common, "TRAIN_LOG_PATH", tmp_path / "train.log")
    env = _common._base_env()
    assert env["PATH"] == "/usr/bin"
    assert "GITHUB_PAT" not in env
    assert "DATABASE_URL" not in env


def test_feedback_aliases_require_slash_outside_bridge_commands():
    assert _common.normalize_command("yes") is None
    assert _common.normalize_command("👍") is None
    assert _common.normalize_command("no") is None
    assert _common.normalize_command("/yes") is None
    assert _common.normalize_command("/rl yes") == "good"
    assert _common.normalize_command("/rc 👎") == "bad"
    assert _common.normalize_command("thumbs down", allow_bare=True) == "bad"


def test_openclaw_command_accepts_yes_no_aliases(monkeypatch, tmp_path):
    real_connect = db.connect
    monkeypatch.setattr(openclaw, "_cfg", lambda: {"model": "m", "batch_min": 999, "agent_admin_commands": False})
    monkeypatch.setattr(openclaw.db, "connect", lambda *args, **kwargs: real_connect(tmp_path / "rc.db"))
    monkeypatch.setattr(openclaw, "maybe_train", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(openclaw, "pop_pending", lambda *_args, **_kwargs: {
        "key": "k", "model": "m", "prompt": "p", "response": "r", "source": "openclaw"
    })
    assert openclaw._run_command("yes", "session")["ok"] is True
    assert openclaw._run_command("👎", "session")["ok"] is True


def test_openclaw_command_can_rate_specific_pending_key(monkeypatch, tmp_path):
    real_connect = db.connect
    seen = {}
    payload = {"key": "abc", "model": "m", "prompt": "p", "response": "r", "source": "openclaw"}
    monkeypatch.setattr(openclaw, "_cfg", lambda: {"model": "m", "batch_min": 999, "agent_admin_commands": False})
    monkeypatch.setattr(openclaw.db, "connect", lambda *args, **kwargs: real_connect(tmp_path / "rc.db"))
    monkeypatch.setattr(openclaw, "maybe_train", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(openclaw, "pop_pending", lambda *args, **kwargs: (seen.setdefault("kwargs", kwargs), payload)[1])
    assert openclaw._run_command("good", "session", "abc")["ok"] is True
    assert seen["kwargs"]["key"] == "abc"


def test_openclaw_admin_commands_require_bridge_config_opt_in(monkeypatch, tmp_path):
    real_connect = db.connect
    monkeypatch.setattr(openclaw, "_cfg", lambda: {"model": "m", "agent_admin_commands": False})
    monkeypatch.setattr(openclaw.db, "connect", lambda *args, **kwargs: real_connect(tmp_path / "rc.db"))
    called = []
    monkeypatch.setattr(openclaw, "queue_training", lambda **_kwargs: called.append(True))
    assert openclaw._run_command("train", "session")["ok"] is False
    monkeypatch.setattr(openclaw, "_cfg", lambda: {"model": "m", "agent_admin_commands": True})
    assert openclaw._run_command("train", "session")["ok"] is True
    assert called


def test_openclaw_reset_warns_about_serving_state(monkeypatch, tmp_path):
    real_connect = db.connect
    monkeypatch.setattr(openclaw, "_cfg", lambda: {"model": "m", "agent_admin_commands": True})
    monkeypatch.setattr(openclaw.db, "connect", lambda *args, **kwargs: real_connect(tmp_path / "rc.db"))
    monkeypatch.setattr("reinforceclaw.cli.reset_state", lambda: None)
    result = openclaw._run_command("reset", "session")
    assert result["ok"] is True
    assert "model server" in result["message"]


def test_openclaw_server_has_thread_limit():
    assert openclaw.LimitedThreadingHTTPServer.daemon_threads is True


def test_openclaw_json_response_sets_length_and_close():
    class Fake:
        def __init__(self):
            self.headers = {}
            self.wfile = type("W", (), {"write": lambda self, data: setattr(self, "data", data)})()

        def send_response(self, code):
            self.code = code

        def send_header(self, key, value):
            self.headers[key] = value

        def end_headers(self):
            pass

    fake = Fake()
    openclaw._send_json(fake, {"ok": True})
    assert fake.headers["Content-Length"] == str(len(fake.wfile.data))
    assert fake.headers["Connection"] == "close"


def test_openclaw_get_rejects_nonlocal_host(monkeypatch):
    monkeypatch.setattr(openclaw, "_authorized", lambda _headers: True)

    class Fake(openclaw.Handler):
        def __init__(self):
            self.headers = {openclaw.SECRET_HEADER: "x" * 32, "Host": "example.com"}
            self.path = "/feedback/status"
            self.sent = []

        def send_response(self, code):
            self.sent.append(code)

        def send_header(self, *_args):
            pass

        def end_headers(self):
            pass

    fake = Fake()
    openclaw.Handler.do_GET(fake)
    assert fake.sent[0] == 403


def test_systemd_escaping_strips_newlines_and_escapes_expansions():
    assert "\n" not in scheduler._systemd_env("a\nb")
    assert scheduler._systemd_arg("/tmp/$x/%n") == '"/tmp/$$x/%%n"'


def test_scheduler_env_skips_secret_proxy(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HTTPS_PROXY", "http://user:pass@example")
    monkeypatch.setattr(scheduler.Path, "home", lambda: tmp_path)
    env = scheduler._scheduler_env()
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "HTTPS_PROXY" not in env


def test_scheduler_env_does_not_persist_hf_token_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.delenv("REINFORCECLAW_SCHEDULER_PERSIST_HF_TOKEN", raising=False)
    monkeypatch.setattr(scheduler.Path, "home", lambda: tmp_path)
    assert "HF_TOKEN" not in scheduler._scheduler_env()


def test_schedule_auto_persists_even_if_old_timer_cleanup_fails(monkeypatch):
    saved = []
    monkeypatch.setattr(cli, "load_config", lambda: {"train_schedule": "03:00", "schedule_window_minutes": 180})
    monkeypatch.setattr(cli, "save_config", lambda cfg: saved.append(dict(cfg)))
    monkeypatch.setattr(scheduler, "install", lambda schedule, window: False)
    monkeypatch.setattr(scheduler, "LAST_ERROR", "cleanup failed")
    cli.cmd_schedule(argparse.Namespace(time="auto"))
    assert saved == [{"train_schedule": "auto", "schedule_window_minutes": 180}]


def test_queue_training_spawn_failure_is_best_effort(monkeypatch, tmp_path):
    monkeypatch.setattr(_common, "TRAIN_QUEUE_LOCK_PATH", tmp_path / "queue.lock")
    monkeypatch.setattr(_common, "TRAIN_LOCK_PATH", tmp_path / "train.lock")
    monkeypatch.setattr(_common, "TRAIN_RETRY_PATH", tmp_path / "train.retry")
    monkeypatch.setattr(_common, "TRAIN_LOG_PATH", tmp_path / "train.log")

    def fail_spawn(_argv):
        raise OSError("blocked")

    monkeypatch.setattr(_common, "_spawn_train", fail_spawn)
    _common.queue_training()
    assert not _common.TRAIN_RETRY_PATH.exists()


def test_queue_training_force_bypasses_retry_cooldown(monkeypatch, tmp_path):
    monkeypatch.setattr(_common, "TRAIN_QUEUE_LOCK_PATH", tmp_path / "queue.lock")
    monkeypatch.setattr(_common, "TRAIN_LOCK_PATH", tmp_path / "train.lock")
    monkeypatch.setattr(_common, "TRAIN_RETRY_PATH", tmp_path / "train.retry")
    monkeypatch.setattr(_common, "TRAIN_LOG_PATH", tmp_path / "train.log")
    monkeypatch.setattr(_common, "_train_lock_held", lambda: False)
    _common._write_retry_due(9999999999)
    spawned = []
    monkeypatch.setattr(_common, "_spawn_train", lambda argv: spawned.append(argv))
    _common.queue_training()
    assert not spawned
    _common.queue_training(force=True)
    assert spawned


def test_retry_after_uses_existing_retry_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(_common, "TRAIN_RETRY_PATH", tmp_path / "train.retry")
    monkeypatch.setattr(_common, "TRAIN_LOG_PATH", tmp_path / "train.log")
    _common._write_retry_due(100.0)
    monkeypatch.setattr(_common.time, "time", lambda: 101.0)
    monkeypatch.setattr(_common.time, "sleep", lambda _seconds: None)
    queued = []
    monkeypatch.setattr(_common, "queue_training", lambda: queued.append(True))
    _common._retry_after(900.0)
    assert queued == [True]
    assert not _common.TRAIN_RETRY_PATH.exists()


def test_openclaw_session_key_has_length_cap():
    assert openclaw._session_key({"sessionKey": "abc"}) == "abc"
    assert openclaw._session_key({"sessionKey": "x" * 600}) == ""


def test_cuda_headroom_requires_gpu_and_host_telemetry():
    hardware = backend_cuda.CUDAHardware("cuda", 10**10, None, 10**10, None)
    assert trainer._has_minimum_headroom(hardware) is False


def test_transcript_reader_refuses_symlink(tmp_path, monkeypatch):
    home = tmp_path / "home"
    claude = home / ".claude"
    outside = tmp_path / "outside.jsonl"
    claude.mkdir(parents=True)
    outside.write_text(json.dumps({"message": {"role": "user", "content": "secret"}}) + "\n", encoding="utf-8")
    link = claude / "transcript.jsonl"
    link.symlink_to(outside)
    monkeypatch.setattr(_common.Path, "home", lambda: home)
    assert _common._recent_transcript_rows(link) == []


def test_vllm_load_adapter_uses_configured_url(monkeypatch, tmp_path):
    seen = []
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    class Response:
        status_code = 200

    def fake_post(url, **kwargs):
        seen.append(url)
        return Response()

    monkeypatch.setattr("requests.post", fake_post)
    assert trainer.load_adapter("vllm", adapter, "m", "http://127.0.0.1:9999") is True
    assert seen == [
        "http://127.0.0.1:9999/v1/unload_lora_adapter",
        "http://127.0.0.1:9999/v1/load_lora_adapter",
    ]


def test_load_adapter_rejects_remote_server_without_opt_in(monkeypatch, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    monkeypatch.delenv("REINFORCECLAW_ALLOW_REMOTE_SERVER", raising=False)
    assert trainer.load_adapter("vllm", adapter, "m", "http://example.com:8000") is None


def test_ollama_http_fallback_uses_blob_adapter_api(monkeypatch, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter.gguf").write_bytes(b"adapter")
    calls = []

    class Completed:
        returncode = 1

    class Response:
        status_code = 200

    monkeypatch.setattr(trainer.subprocess, "run", lambda *args, **kwargs: Completed())
    monkeypatch.setattr("requests.head", lambda *args, **kwargs: type("Missing", (), {"status_code": 404})())

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("requests.post", fake_post)
    assert trainer.load_adapter("ollama", adapter, "base-model", "http://127.0.0.1:11434") is True
    assert calls[0][0].startswith("http://127.0.0.1:11434/api/blobs/sha256:")
    assert calls[1][0] == "http://127.0.0.1:11434/api/create"
    assert calls[1][1]["json"]["adapters"]["adapter.gguf"].startswith("sha256:")


def test_filter_tokenized_pairs_keeps_valid_rows_and_ignores_poison(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    good_id = db.add_feedback(conn, "m", "p1", "r1", 1)
    bad_id = db.add_feedback(conn, "m", "p2", "", -1)

    class V:
        def __init__(self, n):
            self.shape = (n,)

    batch = [{"id": good_id}, {"id": bad_id}]
    pairs = [
        (batch[0], {"id": good_id, "input_ids": V(3), "response_start": 2}),
        (batch[1], {"id": bad_id, "input_ids": V(3), "response_start": 3}),
    ]
    kept, skip = trainer._filter_tokenized_pairs(conn, batch, pairs, "test")
    assert skip is None
    assert [item["id"] for item, _tok in kept] == [good_id]
    row = conn.execute("SELECT rating, trained FROM feedback WHERE id=?", (bad_id,)).fetchone()
    assert (row["rating"], row["trained"]) == (0, 1)


def test_build_batch_reserves_fresh_feedback_even_with_full_replay_ratio(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    fresh_id = db.add_feedback(conn, "m", "fresh", "r", 1)
    replay_ids = [db.add_feedback(conn, "m", f"old{i}", "r", 1) for i in range(4)]
    conn.execute(
        "UPDATE feedback SET trained=1 WHERE id IN (" + ",".join("?" for _ in replay_ids) + ")",
        tuple(replay_ids),
    )
    batch, fresh_ids, fresh = trainer._build_batch(conn, 3, 1.0, model="m")
    assert fresh_ids == [fresh_id]
    assert fresh == [next(item for item in batch if item["id"] == fresh_id)]


def test_training_context_ignores_poisoned_message_rows():
    messages = trainer._build_chat_messages({"messages": ["bad", {"role": "tool", "content": "skip"}, {"role": "user", "content": "ok"}]}, "fallback")
    assert messages == [{"role": "user", "content": "ok"}]


def test_feedback_mutation_clears_resume_when_parent_adapter_stales(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    used = db.add_feedback(conn, "m", "p", "r", 1)
    paused = db.add_feedback(conn, "m", "p2", "r2", -1)
    db.record_training_round(conn, 0.0, 1, 1, str(tmp_path / "v1" / "adapter_model.safetensors"), metrics={"model": "m", "feedback_ids": [used]})
    db.activate_training_round(conn, 1, 0.0, 1, [used])
    db.save_training_state(conn, {"parent_version": 1, "batch_ids": [paused], "fresh_ids": [paused]})
    db.revise_feedback_rating(conn, used, -1)
    assert db.get_training_state(conn) is None


def test_cannot_activate_candidate_with_stale_parent(tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    db.record_training_round(conn, 0.0, 1, 1, str(tmp_path / "v1" / "adapter_model.safetensors"), metrics={"model": "m"})
    db.record_training_round(conn, 0.0, 1, 2, str(tmp_path / "v2" / "adapter_model.safetensors"), parent=1, metrics={"model": "m"})
    conn.execute("UPDATE adapters SET status='stale' WHERE version=1")
    assert db.can_activate_adapter(conn, 2) is False
    assert db.activate_training_round(conn, 2, 0.0, 1, []) is None


def test_candidate_activation_does_not_mark_db_active_if_server_load_fails(monkeypatch, tmp_path):
    conn = db.connect(tmp_path / "rc.db")
    db.record_training_round(conn, 0.0, 1, 1, str(tmp_path / "v1" / "adapter_model.safetensors"), metrics={"model": "m", "feedback_ids": []})
    monkeypatch.setattr(cli.trainer, "load_adapter", lambda *args, **kwargs: False)
    active, status = cli._activate_candidate(conn, {"model": "m", "server": "ollama"}, {"version": 1, "path": "x", "ema_mean": 0.0, "ema_count": 1, "feedback_ids": []})
    assert (active, status) == (None, False)
    assert db.latest_adapter(conn, model="m") is None
    assert conn.execute("SELECT status FROM adapters WHERE version=1").fetchone()["status"] == "candidate"


def test_adapter_root_must_stay_private(monkeypatch, tmp_path):
    monkeypatch.setattr(trainer.Path, "home", lambda: tmp_path)
    with pytest.raises(PermissionError):
        trainer._adapter_dir({"adapter_root": str(tmp_path / ".." / "elsewhere")})


def test_adapter_root_cannot_be_private_root(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setattr(trainer.Path, "home", lambda: home)
    with pytest.raises(PermissionError):
        trainer._adapter_dir({"adapter_root": str(home / ".reinforceclaw")})


def test_reset_state_deletes_configured_private_adapter_root(monkeypatch, tmp_path):
    home = tmp_path / "home"
    private = home / ".reinforceclaw"
    custom = private / "custom-adapters"
    custom.mkdir(parents=True)
    (custom / "leftover").write_text("x", encoding="utf-8")
    monkeypatch.setattr(trainer.Path, "home", lambda: home)
    monkeypatch.setattr(cli, "CONFIG_PATH", private / "config.json")
    monkeypatch.setattr(cli, "ADAPTER_ROOT", private / "adapters")
    monkeypatch.setattr(cli, "RESET_MARK_PATH", private / "reset.marker")
    monkeypatch.setattr(cli, "TRAIN_RETRY_PATH", private / "train.retry")
    monkeypatch.setattr(trainer, "TRAIN_LOCK_PATH", private / "train.lock")
    monkeypatch.setattr(db, "DB_PATH", private / "reinforceclaw.db")
    monkeypatch.setattr(_common, "PENDING_DIR", private / "pending")
    monkeypatch.setattr(_common, "PENDING_LOCK_PATH", private / "pending.lock")
    cli.save_config({"model": "m", "adapter_root": str(custom)})
    cli.reset_state()
    assert not custom.exists()


def test_openclaw_plugin_does_not_request_unsafe_runtime_access():
    root = cli.Path(__file__).resolve().parents[1] / "reinforceclaw" / "openclaw_plugin"
    source = (root / "src" / "index.ts").read_text(encoding="utf-8")
    manifest = json.loads((root / "openclaw.plugin.json").read_text(encoding="utf-8"))
    forbidden = ("child_process", "process.env", "spawn(", "readFileSync", "reinforceclawPython")
    assert not any(token in source for token in forbidden)
    assert "reinforceclawPython" not in manifest["configSchema"]["properties"]


def test_openclaw_bridge_launchd_service_uses_python_module(monkeypatch, tmp_path):
    loaded = []
    monkeypatch.setattr(scheduler.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(scheduler, "BRIDGE_PLIST_PATH", tmp_path / "bridge.plist")
    monkeypatch.setattr(scheduler, "BRIDGE_LOG", tmp_path / "bridge.log")
    monkeypatch.setattr(scheduler, "_run_ok", lambda cmd: loaded.append(cmd) or True)
    assert scheduler.install_openclaw_bridge() is True
    plist = scheduler.BRIDGE_PLIST_PATH.read_text(encoding="utf-8")
    assert "reinforceclaw.hooks.openclaw" in plist
    assert "REINFORCECLAW_OPENCLAW_SECRET" not in plist
    assert ["launchctl", "load", str(scheduler.BRIDGE_PLIST_PATH)] in loaded
