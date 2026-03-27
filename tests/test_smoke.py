"""Smoke test — 60 prompts with known answers. 3 training rounds."""

import sys
import tempfile
from pathlib import Path

from reinforceclaw.backend_mlx import mlx_drain


def _free():
    """Force memory cleanup after deleting a model."""
    mlx_drain(collect_garbage=True)


def _drain():
    mlx_drain()

# 60 prompts with verifiable checks.
PROMPTS = [
    # math
    ("What is 17 * 23?", lambda r: "391" in r),
    ("What is 2^10?", lambda r: "1024" in r),
    ("What is 144 / 12?", lambda r: "12" in r),
    ("What is 15% of 200?", lambda r: "30" in r),
    ("What is the square root of 81?", lambda r: "9" in r),
    ("What is 7 * 8?", lambda r: "56" in r),
    ("What is 1000 - 373?", lambda r: "627" in r),
    ("What is 25 * 4?", lambda r: "100" in r),
    ("Convert 100 Celsius to Fahrenheit.", lambda r: "212" in r),
    ("What is 13 + 29?", lambda r: "42" in r),
    # facts
    ("What is the capital of France?", lambda r: "paris" in r.lower()),
    ("What is the capital of Japan?", lambda r: "tokyo" in r.lower()),
    ("What is the capital of Australia?", lambda r: "canberra" in r.lower()),
    ("What is the chemical symbol for gold?", lambda r: "Au" in r),
    ("What is the chemical symbol for water?", lambda r: "H2O" in r or "h2o" in r.lower()),
    ("How many continents are there?", lambda r: "7" in r or "seven" in r.lower()),
    ("What planet is closest to the sun?", lambda r: "mercury" in r.lower()),
    ("What is the largest ocean?", lambda r: "pacific" in r.lower()),
    ("What year did World War 2 end?", lambda r: "1945" in r),
    ("Who wrote Romeo and Juliet?", lambda r: "shakespeare" in r.lower()),
    # code
    ("Write a Python function that reverses a string.", lambda r: "def " in r),
    ("Write a SQL query to select all users.", lambda r: "select" in r.lower() and "from" in r.lower()),
    ("Write a JSON object with keys 'name' and 'age'.", lambda r: '"name"' in r and '"age"' in r),
    ("Write a Python list comprehension that squares numbers 1-5.", lambda r: "[" in r and "for" in r),
    ("Write a bash command to list all files in the current directory.", lambda r: "ls" in r),
    ("Write a Python function to check if a number is prime.", lambda r: "def " in r and "prime" in r.lower()),
    ("Write a regex pattern that matches email addresses.", lambda r: "@" in r),
    ("Write a Python dict with 3 key-value pairs.", lambda r: "{" in r and ":" in r),
    ("Write a for loop in Python that prints 1 to 10.", lambda r: "for" in r and "print" in r),
    ("Write a Python function that returns the factorial of n.", lambda r: "def " in r),
    # formatting
    ("Write a haiku about programming.", lambda r: len(r.strip().split("\n")) >= 3),
    ("List exactly 5 benefits of exercise.", lambda r: len([l for l in r.split("\n") if l.strip()]) >= 5),
    ("Explain recursion in exactly 3 sentences.", lambda r: r.count(".") >= 3),
    ("Write the word 'hello' in ALL CAPS.", lambda r: "HELLO" in r),
    ("Count from 1 to 5, one per line.", lambda r: "1" in r and "5" in r),
    ("Translate 'thank you' to Spanish.", lambda r: "gracias" in r.lower()),
    ("Write a 4-line poem.", lambda r: len([l for l in r.strip().split("\n") if l.strip()]) >= 4),
    ("List the planets in our solar system.", lambda r: "mars" in r.lower() and "jupiter" in r.lower()),
    ("What does HTTP stand for?", lambda r: "hypertext" in r.lower()),
    ("Write a bullet list of 3 programming languages.", lambda r: any(l in r.lower() for l in ["python", "java", "rust"])),
    # reasoning
    ("If a train travels at 60mph for 2.5 hours, how far does it go?", lambda r: "150" in r),
    ("What comes next: 2, 4, 8, 16, __?", lambda r: "32" in r),
    ("If you have 3 apples and buy 5 more, how many do you have?", lambda r: "8" in r),
    ("What is bigger: 0.3 or 0.25?", lambda r: "0.3" in r or ".3" in r),
    ("How many letters are in the word 'elephant'?", lambda r: "8" in r),
    ("If today is Monday, what day is it in 3 days?", lambda r: "thursday" in r.lower()),
    ("What is 20% of 50?", lambda r: "10" in r),
    ("A rectangle is 4m wide and 7m long. What is its area?", lambda r: "28" in r),
    ("Sort these numbers: 5, 2, 8, 1, 9.", lambda r: "1" in r and "9" in r),
    ("What is the next prime after 7?", lambda r: "11" in r),
    # general
    ("Explain what an API is in simple terms.", lambda r: len(r) > 50),
    ("Write exactly one paragraph about cats.", lambda r: len(r) > 30),
    ("Name 3 primary colors.", lambda r: any(c in r.lower() for c in ["red", "blue", "yellow"])),
    ("What is the boiling point of water in Celsius?", lambda r: "100" in r),
    ("Name a programming language created by Google.", lambda r: "go" in r.lower() or "dart" in r.lower()),
    ("What does CPU stand for?", lambda r: "central" in r.lower() and "processing" in r.lower()),
    ("What is the speed of light in km/s (approximately)?", lambda r: "300" in r or "299" in r),
    ("Name the 4 seasons.", lambda r: "spring" in r.lower() and "summer" in r.lower()),
    ("What is the smallest prime number?", lambda r: "2" in r),
    ("How many sides does a hexagon have?", lambda r: "6" in r or "six" in r.lower()),
]
TRAIN_PROMPTS = PROMPTS[:40]
EVAL_PROMPTS = PROMPTS[40:]


def _generate(model, tokenizer, prompt):
    from mlx_lm import generate
    from reinforceclaw.trainer import _apply_chat_template
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = _apply_chat_template(
            tokenizer,
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False,
        )
    else:
        formatted = prompt
    text = generate(model, tokenizer, prompt=formatted, max_tokens=96, verbose=False)
    _drain()
    return text


def _evaluate(model, tokenizer, prompts):
    passed = 0
    for i, (p, check) in enumerate(prompts):
        if check(_generate(model, tokenizer, p)):
            passed += 1
        if (i + 1) % 10 == 0:
            print(f"  eval {i+1}/{len(prompts)}...", flush=True)
    return passed / len(prompts)


def run(model_name=None, rounds=3):
    try:
        from mlx_lm import load as mlx_load
    except ImportError:
        print("mlx-lm not installed."); return

    from reinforceclaw import db
    from reinforceclaw.trainer import train_result, load_model

    model_name = model_name or "Qwen/Qwen2.5-7B-Instruct"
    print(f"Model: {model_name}", flush=True)
    print(f"Train prompts: {len(TRAIN_PROMPTS)} | Eval prompts: {len(EVAL_PROMPTS)}")

    # single model load for baseline + rating
    model, tok = mlx_load(model_name)
    print("Evaluating baseline...", flush=True)
    baseline = _evaluate(model, tok, EVAL_PROMPTS)
    print(f"Baseline: {baseline:.0%} ({int(baseline * len(EVAL_PROMPTS))}/{len(EVAL_PROMPTS)})", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        conn = db.connect(Path(tmp) / "smoke.db")
        cfg = {"model": model_name, "loss_fn": "mis-po", "lr": 8e-6, "traj_clip": [0.996, 1.001], "steps": 4,
               "token_clip": [0.5, 2.0], "kl_coeff": 0.08, "lora_rank": 16,
               "grad_accum": 1, "grad_clip": 1.0, "batch_min": 8, "batch_size": 4,
               "replay_ratio": 0.25, "ema_decay": 0.99, "pos_weight": 1.0,
               "lora_target": "attention", "max_passes": 1.0,
               "adapter_keep": 5,
               "low_priority": False,
               "memory_fraction": 0.70}

        # auto-rate using the SAME model — no second load
        print("\nAuto-rating all prompts...")
        for i, (prompt, check) in enumerate(TRAIN_PROMPTS, 1):
            resp = _generate(model, tok, prompt)
            rating = 1 if check(resp) else -1
            db.add_feedback(conn, model_name, prompt, resp, rating, source="smoke")
            if i % 10 == 0:
                print(f"  rated {i}/{len(TRAIN_PROMPTS)}...", flush=True)
        print(f"Rated: {len(TRAIN_PROMPTS)} prompts", flush=True)

        # free model before training (trainer loads its own)
        del model, tok
        _free()
        import time; time.sleep(3)  # give Metal time to reclaim GPU memory

        for rnd in range(1, rounds + 1):
            print(f"\n--- Round {rnd}/{rounds} ---", flush=True)
            m = train_result(cfg, conn)
            if m.get("status") == "trained":
                print(f"  Loss: {m['avg_loss']:.4f}, EMA: {m['ema_mean']:.3f}", flush=True)
                if rnd < rounds:
                    latest = db.latest_adapter(conn)
                    if latest:
                        tm, tt = load_model(model_name, cfg["lora_rank"], latest["path"])
                        for prompt, check in TRAIN_PROMPTS:
                            resp = _generate(tm, tt, prompt)
                            db.add_feedback(conn, model_name, prompt, resp,
                                            1 if check(resp) else -1, source="smoke")
                        del tm, tt
                        _free()
                        time.sleep(2)
            else:
                print(f"  Skipped: {m.get('reason', 'unknown')}")

        latest = db.latest_adapter(conn)
        if latest:
            fm, ft = load_model(model_name, cfg["lora_rank"], latest["path"])
            final = _evaluate(fm, ft, EVAL_PROMPTS)
            del fm, ft
            _free()
            time.sleep(2)
        else:
            final = baseline

        delta = final - baseline
        print(f"\n{'='*40}")
        print(f"Baseline: {baseline:.0%} ({int(baseline * len(EVAL_PROMPTS))}/{len(EVAL_PROMPTS)})", flush=True)
        print(f"Final:    {final:.0%} ({int(final * len(EVAL_PROMPTS))}/{len(EVAL_PROMPTS)})")
        print(f"Delta:    {delta:+.0%}")
        print(f"{'PASS' if delta >= 0.05 else f'Below 5pp threshold ({delta:+.0%})'}")
        conn.close()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
