"""Integration tests — DB layer. All 17 scenarios."""

import json
import tempfile
from pathlib import Path
import pytest
from nudge import db


@pytest.fixture
def conn():
    with tempfile.TemporaryDirectory() as tmp:
        c = db.connect(Path(tmp) / "test.db")
        yield c
        c.close()


class TestDB:
    def test_tables(self, conn):
        names = [t["name"] for t in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
        assert "feedback" in names and "adapters" in names and "ema_state" in names

    def test_add_get(self, conn):
        fid = db.add_feedback(conn, "qwen-7b", "hello", "world", 1)
        assert fid == 1
        u = db.get_untrained(conn)
        assert len(u) == 1 and u[0]["rating"] == 1

    def test_mark_trained(self, conn):
        fid = db.add_feedback(conn, "m", "p", "r", -1)
        db.mark_trained(conn, [fid], 1)
        assert len(db.get_untrained(conn)) == 0

    def test_replay(self, conn):
        fid = db.add_feedback(conn, "m", "p", "r", 1)
        db.mark_trained(conn, [fid], 1)
        assert len(db.get_replay(conn, limit=10)) == 1

    def test_ema(self, conn):
        mean, n = db.get_ema(conn)
        assert mean == 0.0 and n == 0
        db.update_ema(conn, 0.5, 10)
        assert db.get_ema(conn) == (0.5, 10)

    def test_adapters(self, conn):
        assert db.latest_adapter(conn) is None
        db.add_adapter(conn, 1, "/tmp/v1/a.st", metrics={"loss": 0.5})
        db.add_adapter(conn, 2, "/tmp/v2/a.st", parent=1)
        assert db.latest_adapter(conn)["version"] == 2

    def test_rollback(self, conn):
        db.add_adapter(conn, 1, "/tmp/v1/a.st")
        db.add_adapter(conn, 2, "/tmp/v2/a.st", parent=1)
        prev = db.rollback(conn)
        assert prev["version"] == 1
        assert db.latest_adapter(conn)["version"] == 1

    def test_rollback_empty(self, conn):
        assert db.rollback(conn) is None

    def test_cleanup(self, conn):
        for i in range(1, 6):
            db.add_adapter(conn, i, f"/tmp/v{i}/a.st")
        removed = db.cleanup_adapters(conn, keep=3)
        assert len(removed) == 2

    def test_count(self, conn):
        db.add_feedback(conn, "m", "p", "r", 1)
        db.add_feedback(conn, "m", "p", "r", -1)
        db.add_feedback(conn, "m", "p", "r", 1)
        c = db.count(conn)
        assert c["total"] == 3 and c["good"] == 2 and c["bad"] == 1 and c["untrained"] == 3

    def test_count_empty(self, conn):
        c = db.count(conn)
        assert c == {"total": 0, "good": 0, "bad": 0, "untrained": 0}

    def test_remove_last(self, conn):
        db.add_feedback(conn, "m", "p1", "r1", 1)
        db.add_feedback(conn, "m", "p2", "r2", -1)
        r = db.remove_last(conn)
        assert r["rating"] == -1
        assert len(db.get_untrained(conn)) == 1

    def test_remove_last_empty(self, conn):
        assert db.remove_last(conn) is None

    def test_remove_last_ignores_pending(self, conn):
        db.add_feedback(conn, "m", "p1", "r1", 1)
        db.add_feedback(conn, "m", "p2", "r2", 0)
        removed = db.remove_last(conn)
        assert removed["rating"] == 1
        assert db.latest_pending(conn)["prompt"] == "p2"

    def test_latest_pending(self, conn):
        db.add_feedback(conn, "m", "p1", "r1", 0, source="hook")
        fid2 = db.add_feedback(conn, "m", "p2", "r2", 0, source="hook")
        pending = db.latest_pending(conn, source="hook")
        assert pending["id"] == fid2

    def test_update_feedback_rating(self, conn):
        fid = db.add_feedback(conn, "m", "p", "r", 0)
        db.update_feedback_rating(conn, fid, -1)
        assert db.latest_pending(conn) is None
        assert db.get_untrained(conn)[0]["rating"] == -1

    def test_reset(self, conn):
        db.add_feedback(conn, "m", "p", "r", 1)
        db.add_adapter(conn, 1, "/tmp/v1")
        db.update_ema(conn, 0.5, 10)
        db.reset_all(conn)
        assert len(db.get_untrained(conn)) == 0
        assert db.latest_adapter(conn) is None
        assert db.get_ema(conn) == (0.0, 0)

    def test_batch_16(self, conn):
        for i in range(16):
            db.add_feedback(conn, "m", f"p{i}", f"r{i}", 1 if i % 3 == 0 else -1)
        u = db.get_untrained(conn)
        assert len(u) == 16
        assert all(item["rating"] in (1, -1) for item in u)

    def test_wal(self, conn):
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    def test_sources(self, conn):
        db.add_feedback(conn, "m", "p", "r", 1, source="cli")
        db.add_feedback(conn, "m", "p", "r", -1, source="hook")
        rows = conn.execute("SELECT source FROM feedback ORDER BY id").fetchall()
        assert rows[0]["source"] == "cli" and rows[1]["source"] == "hook"

    def test_context(self, conn):
        ctx = json.dumps({"system": "You are helpful"})
        db.add_feedback(conn, "m", "p", "r", 1, context=ctx)
        assert json.loads(db.get_untrained(conn)[0]["context"])["system"] == "You are helpful"
