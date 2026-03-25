"""Smoke test — 20 IFEval prompts. Run: python -m tests.test_smoke [model_name]"""
# auto-rates each prompt, trains 3 rounds, measures pass rate delta.
# success = 5+ percentage point improvement. that's the bar.

import sys
import tempfile
from pathlib import Path

PROMPTS = [
    ("Write a haiku about programming.", lambda r: len(r.strip().split("\n")) >= 3),
    ("List exactly 5 benefits of exercise.", lambda r: len([l for l in r.split("\n") if l.strip()]) >= 5),
    ("Explain recursion in exactly 3 sentences.", lambda r: r.count(".") >= 3),
    ("Write a function that reverses a string in Python.", lambda r: "def " in r),
    ("What is 17 * 23?", lambda r: "391" in r),
    ("Name 3 primary colors.", lambda r: any(c in r.lower() for c in ["red", "blue", "yellow"])),
    ("Write the word 'hello' in ALL CAPS.", lambda r: "HELLO" in r),
    ("Count from 1 to 5, one per line.", lambda r: "1" in r and "5" in r),
    ("What is the capital of France?", lambda r: "paris" in r.lower()),
    ("Write a JSON object with keys 'name' and 'age'.", lambda r: '"name"' in r and '"age"' in r),
    ("Translate 'thank you' to Spanish.", lambda r: "gracias" in r.lower()),
    ("Write a 4-line poem that rhymes.", lambda r: len([l for l in r.strip().split("\n") if l.strip()]) >= 4),
    ("List the planets in our solar system.", lambda r: "mars" in r.lower() and "jupiter" in r.lower()),
    ("Explain what an API is in simple terms.", lambda r: len(r) > 50),
    ("Write a SQL query to select all users.", lambda r: "select" in r.lower() and "from" in r.lower()),
    ("What does HTTP stand for?", lambda r: "hypertext" in r.lower()),
    ("Write exactly one paragraph about cats.", lambda r: len(r) > 30),
    ("Convert 100 Celsius to Fahrenheit.", lambda r: "212" in r),
    ("Write a bullet list of 3 programming languages.", lambda r: any(l in r.lower() for l in ["python", "java", "rust"])),
    ("What is 2^10?", lambda r: "1024" in r),
]


def _generate(model, tokenizer, prompt):
    """Greedy decode one prompt. Returns response text."""
    import mlx.core as mx
    if hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    else:
        ids = tokenizer.encode(prompt)
    gen = list(ids)
    for _ in range(256):
        logits = model(mx.array(gen)[None, :])
        tok = mx.argmax(logits[0, -1, :]).item()
        if tok == tokenizer.eos_token_id:
            break
        gen.append(tok)
    return tokenizer.decode(gen[len(ids):])


def _evaluate(model, tokenizer):
    passed = sum(1 for p, check in PROMPTS if check(_generate(model, tokenizer, p)))
    return passed / len(PROMPTS)


def run(model_name=None, rounds=3):
    try:
        from mlx_lm import load as mlx_load
    except ImportError:
        print("mlx-lm not installed. Skipping."); return

    from nudge import db
    from nudge.trainer import train, load_model

    model_name = model_name or "Qwen/Qwen3-0.6B"
    print(f"Model: {model_name}")

    model, tok = mlx_load(model_name)
    baseline = _evaluate(model, tok)
    print(f"Baseline: {baseline:.0%} ({int(baseline*20)}/20)")
    del model

    with tempfile.TemporaryDirectory() as tmp:
        conn = db.connect(Path(tmp) / "smoke.db")
        cfg = {"model": model_name, "lr": 4e-6, "traj_clip": [0.992, 1.002], "steps": 8,
               "token_clip": [0.5, 2.0], "kl_coeff": 0.001, "lora_rank": 16,
               "grad_accum": 4, "grad_clip": 1.0, "batch_min": 16,
               "replay_ratio": 0.5, "ema_decay": 0.99, "pos_weight": 1.2,
               "adapter_keep": 20}

        # auto-rate with base model
        base, btok = mlx_load(model_name)
        for prompt, check in PROMPTS:
            resp = _generate(base, btok, prompt)
            db.add_feedback(conn, model_name, prompt, resp, 1 if check(resp) else -1, source="smoke")
        del base

        for rnd in range(1, rounds + 1):
            print(f"\nRound {rnd}/{rounds}...")
            m = train(cfg, conn)
            if m:
                print(f"  Loss: {m['avg_loss']:.4f}, EMA: {m['ema_mean']:.3f}")
                if rnd < rounds:
                    latest = db.latest_adapter(conn)
                    if latest:
                        tm, tt = load_model(model_name, cfg["lora_rank"], latest["path"])
                        for prompt, check in PROMPTS:
                            resp = _generate(tm, tt, prompt)
                            db.add_feedback(conn, model_name, prompt, resp,
                                            1 if check(resp) else -1, source="smoke")
                        del tm

        latest = db.latest_adapter(conn)
        if latest:
            fm, ft = load_model(model_name, cfg["lora_rank"], latest["path"])
            final = _evaluate(fm, ft)
            del fm
        else:
            final = baseline

        delta = final - baseline
        print(f"\nBaseline: {baseline:.0%} | Final: {final:.0%} | Delta: {delta:+.0%}")
        print("PASS" if delta >= 0.05 else f"Below 5pp threshold ({delta:+.0%})")
        conn.close()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
