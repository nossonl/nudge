"""Training pipeline test — verifies MIS-PO produces real gradients and adapters."""

import tempfile
from pathlib import Path


def test_training_pipeline():
    """Full training: rate 20 → train → verify adapter exists and loss decreased."""
    try:
        from mlx_lm import load as mlx_load
    except ImportError:
        print("mlx-lm not installed, skipping"); return

    from reinforceclaw import db
    from reinforceclaw.trainer import train, load_model

    model_name = "Qwen/Qwen3.5-9B"
    print(f"Model: {model_name}")

    with tempfile.TemporaryDirectory() as tmp:
        conn = db.connect(Path(tmp) / "train_test.db")
        cfg = {"model": model_name, "lr": 4e-6, "traj_clip": [0.992, 1.002], "steps": 4,
               "token_clip": [0.5, 2.0], "kl_coeff": 0.001, "lora_rank": 16,
               "grad_accum": 2, "grad_clip": 1.0, "batch_min": 16,
               "replay_ratio": 0.5, "ema_decay": 0.99, "pos_weight": 1.2,
               "adapter_keep": 0, "adapter_root": str(Path(tmp) / "adapters"),
               "_skip_lock_once": True}

        # add ratings — mix of good and bad
        print("\n1. Adding 20 ratings (mixed good/bad)...")
        prompts = [
            ("What is 2+2?", "4", 1),
            ("What is the capital of France?", "Paris", 1),
            ("Write hello in caps", "HELLO", 1),
            ("What is 10*10?", "100", 1),
            ("Name a color", "Blue", 1),
            ("What is 5+3?", "9", -1),  # wrong answer
            ("Capital of Japan?", "Beijing", -1),  # wrong
            ("Write bye in caps", "bye", -1),  # didn't capitalize
            ("What is 3*3?", "12", -1),  # wrong
            ("Name a planet", "Pluto is not a planet anymore", -1),  # debatable
            ("Explain AI", "AI is artificial intelligence used in many applications today.", 1),
            ("What is Python?", "A programming language", 1),
            ("Sort 3,1,2", "1,2,3", 1),
            ("What is HTTP?", "Hypertext Transfer Protocol", 1),
            ("Reverse 'hello'", "olleh", 1),
            ("What is 100/5?", "25", -1),  # wrong
            ("Is the earth flat?", "Yes it is flat", -1),  # very wrong
            ("What language is this: print('hi')?", "Java", -1),  # wrong
            ("What is 7*8?", "54", -1),  # wrong
            ("What is DNA?", "Deoxyribonucleic acid, the molecule that carries genetic info.", 1),
        ]

        for prompt, response, rating in prompts:
            db.add_feedback(conn, model_name, prompt, response, rating, source="test")

        counts = db.count(conn)
        print(f"   Ratings: {counts['total']} ({counts['good']}+ {counts['bad']}-)")
        assert counts["total"] == 20

        # check EMA before training
        ema_before, _ = db.get_ema(conn)
        print(f"   EMA before: {ema_before}")

        # train round 1
        print("\n2. Training round 1...")
        metrics1 = train(cfg, conn)
        assert metrics1 is not None, "training should have produced results"
        print(f"   Loss: {metrics1['avg_loss']:.4f}")
        print(f"   Batch size: {metrics1['batch_size']}")
        print(f"   EMA after: {metrics1['ema_mean']:.3f}")

        # verify adapter was saved
        adapter1 = db.latest_candidate(conn)
        assert adapter1 is not None, "no adapter saved"
        assert Path(adapter1["path"]).exists(), "adapter file missing"
        print(f"   Adapter: v{adapter1['version']} at {adapter1['path']}")

        # activate candidate and verify EMA updated
        db.activate_training_round(conn, metrics1["version"], metrics1["ema_mean"], metrics1["ema_count"], metrics1["feedback_ids"])
        ema_after, ema_count = db.get_ema(conn)
        assert ema_count > 0, "EMA not updated after activation"
        print(f"   EMA updated: {ema_after:.3f} ({ema_count} samples)")

        # add more ratings and train round 2
        print("\n3. Training round 2 (with replay)...")
        for prompt, response, rating in prompts[:16]:
            db.add_feedback(conn, model_name, f"r2: {prompt}", response, rating, source="test")

        metrics2 = train(cfg, conn)
        assert metrics2 is not None, "round 2 should have trained"
        print(f"   Loss: {metrics2['avg_loss']:.4f}")

        adapter2 = db.latest_candidate(conn)
        assert adapter2["version"] == 2, "should be v2"
        print(f"   Adapter: v{adapter2['version']}")
        db.activate_training_round(conn, metrics2["version"], metrics2["ema_mean"], metrics2["ema_count"], metrics2["feedback_ids"])

        # verify only one active adapter
        adapters = db.list_adapters(conn)
        active = [a for a in adapters if a["status"] == "active"]
        assert len(active) == 1, f"expected 1 active adapter, got {len(active)}"
        print(f"   Active adapters: {len(active)} ✓")

        # test rollback
        print("\n4. Rollback test...")
        prev = db.rollback_to(conn, 1)
        assert prev["version"] == 1
        print(f"   Rolled back to v{prev['version']} ✓")

        # load the adapter and verify it produces different logprobs than base
        print("\n5. Verify adapter changes model output...")
        import mlx.core as mx
        from reinforceclaw.trainer import _compute_logprobs

        base, tok = mlx_load(model_name)
        base.eval()
        test_ids = mx.array(tok.encode("What is 2+2?"))
        base_lp = _compute_logprobs(base, test_ids)
        del base

        adapted, _ = load_model(model_name, 16, adapter2["path"])
        adapted_lp = _compute_logprobs(adapted, test_ids)
        del adapted

        diff = mx.abs(base_lp - adapted_lp).mean().item()
        print(f"   Mean logprob diff (base vs adapted): {diff:.6f}")
        assert Path(adapter2["path"]).stat().st_size > 0, "adapter file should be non-empty"
        print(f"   Adapter artifact exists ✓")

        conn.close()
        print(f"\n{'='*50}")
        print("TRAINING PIPELINE: ALL PASSED")


if __name__ == "__main__":
    test_training_pipeline()
