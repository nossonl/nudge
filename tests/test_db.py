"""DB tests — adapter lifecycle, EMA, ratings, rollback."""

import tempfile
from pathlib import Path
from reinforceclaw import db


def test_full_lifecycle():
    """Full lifecycle: rate → train → adapt → rollback → reset."""
    with tempfile.TemporaryDirectory() as tmp:
        conn = db.connect(Path(tmp) / "test.db")

        # add ratings
        for i in range(20):
            db.add_feedback(conn, "test-model", f"prompt {i}", f"response {i}",
                            1 if i % 3 == 0 else -1, source="test")

        counts = db.count(conn)
        assert counts["total"] == 20
        assert counts["good"] == 7  # 0,3,6,9,12,15,18
        assert counts["bad"] == 13
        assert counts["untrained"] == 20
        print(f"  ratings: {counts['total']} ({counts['good']}+ {counts['bad']}-)")

        # untrained count
        trainable = db.count_trainable_untrained(conn)
        assert trainable == 20
        print(f"  trainable untrained: {trainable}")

        # mark some as trained
        untrained = db.get_untrained(conn, limit=10)
        ids = [r["id"] for r in untrained]
        db.mark_trained(conn, ids, adapter_version=1)
        assert db.count_trainable_untrained(conn) == 10
        print(f"  after marking 10 trained: {db.count_trainable_untrained(conn)} left")

        # replay
        replay = db.get_replay(conn, limit=5)
        assert len(replay) == 5
        assert all(r["trained"] == 1 for r in replay)
        print(f"  replay batch: {len(replay)} items")

        # EMA
        db.update_ema(conn, -0.3, 20)
        mean, count = db.get_ema(conn)
        assert mean == -0.3 and count == 20
        print(f"  EMA: {mean} ({count} updates)")

        # adapter lifecycle
        db.add_adapter(conn, 1, "/tmp/v1/adapter.safetensors", metrics={"loss": 0.5})
        db.add_adapter(conn, 2, "/tmp/v2/adapter.safetensors", parent=1, metrics={"loss": 0.3})
        db.add_adapter(conn, 3, "/tmp/v3/adapter.safetensors", parent=2, metrics={"loss": 0.2})

        latest = db.latest_adapter(conn)
        assert latest["version"] == 3
        print(f"  latest adapter: v{latest['version']}")

        # only one should be active
        adapters = db.list_adapters(conn)
        active = [a for a in adapters if a["status"] == "active"]
        assert len(active) == 1, f"expected 1 active, got {len(active)}"
        print(f"  active adapters: {len(active)} (correct)")

        # rollback
        prev = db.rollback_to(conn, 2)
        assert prev["version"] == 2
        active_after = [a for a in db.list_adapters(conn) if a["status"] == "active"]
        assert len(active_after) == 1
        print(f"  rolled back to v{prev['version']}")

        # history
        recent = db.recent(conn, limit=5)
        assert len(recent) == 5
        print(f"  recent history: {len(recent)} items")

        # update rating
        fid = db.add_feedback(conn, "test-model", "test", "test", 0, source="test")
        db.update_feedback_rating(conn, fid, 1)
        pending = db.latest_pending(conn)
        assert pending is None  # no more pending after rating
        print(f"  rating update: works")

        # undo
        removed = db.remove_last(conn)
        assert removed is not None
        print(f"  undo: removed rating #{removed['id']}")

        # reset
        db.reset_all(conn)
        assert db.count(conn)["total"] == 0
        assert db.latest_adapter(conn) is None
        assert db.get_ema(conn) == (0.0, 0)
        print(f"  reset: clean slate")

        conn.close()
        print("\nDB LIFECYCLE: ALL PASSED")


if __name__ == "__main__":
    test_full_lifecycle()
