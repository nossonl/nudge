"""Minimal training test — 16 ratings, 2 steps, verify loss + adapter save."""

import tempfile
from pathlib import Path


def test_train_minimal():
    try:
        from mlx_lm import load as mlx_load
    except ImportError:
        print("mlx-lm not installed, skipping"); return

    from reinforceclaw import db
    from reinforceclaw.trainer import train

    model_name = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    with tempfile.TemporaryDirectory() as tmp:
        conn = db.connect(Path(tmp) / "test.db")
        cfg = {
            "model": model_name, "lr": 4e-6, "steps": 2,
            "traj_clip": [0.992, 1.002], "token_clip": [0.5, 2.0],
            "kl_coeff": 0.001, "lora_rank": 8, "grad_accum": 2,
            "grad_clip": 1.0, "batch_min": 16, "replay_ratio": 0.0,
            "ema_decay": 0.99, "pos_weight": 1.2, "adapter_keep": 0,
            "adapter_root": str(Path(tmp) / "adapters"), "_skip_lock_once": True,
        }

        # 16 ratings — mix of good and bad
        for i in range(16):
            rating = 1 if i % 2 == 0 else -1
            db.add_feedback(conn, model_name,
                            f"What is {i}+{i}?", f"The answer is {i*2}", rating)

        print(f"Ratings: {db.count(conn)['total']}")

        # Train
        metrics = train(cfg, conn)
        assert metrics is not None, "train() returned None"

        loss = metrics["avg_loss"]
        print(f"Loss: {loss:.4f}")
        assert isinstance(loss, float), f"loss should be float, got {type(loss)}"
        assert loss == loss, "loss is NaN"  # NaN != NaN

        # Adapter saved?
        adapter = db.latest_candidate(conn)
        assert adapter is not None, "no adapter saved"
        assert Path(adapter["path"]).exists(), f"adapter file missing: {adapter['path']}"
        print(f"Adapter: v{adapter['version']} at {adapter['path']}")

        # EMA should not commit until activation.
        ema, count = db.get_ema(conn)
        assert count == 0, "EMA should not update before activation"
        db.activate_training_round(conn, metrics["version"], metrics["ema_mean"], metrics["ema_count"], metrics["feedback_ids"])
        ema, count = db.get_ema(conn)
        assert count > 0, "EMA count not updated after activation"
        print(f"EMA: {ema:.4f} ({count} samples)")

        conn.close()
        print("PASSED")


if __name__ == "__main__":
    test_train_minimal()
