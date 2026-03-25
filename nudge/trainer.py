"""MIS-PO trainer. EMA baseline. LoRA only. Adapter versioning. Hot-swap."""
# score = reward - running_average
# running average tracks all past scores so the model always has signal to learn from

import json
import shutil
from pathlib import Path

from . import db

# lazy — MLX only loaded when you actually train
mx = nn = optim = mlx_load = LoRALinear = None


def _ensure_mlx():
    global mx, nn, optim, mlx_load, LoRALinear
    if mx is not None:
        return
    import mlx.core, mlx.nn, mlx.optimizers
    from mlx_lm import load
    try:
        from mlx_lm.tuner.lora import LoRALinear as _L  # mlx-lm >= 0.31
    except ImportError:
        from mlx_lm.tuning.lora import LoRALinear as _L  # older versions
    mx, nn, optim = mlx.core, mlx.nn, mlx.optimizers
    mlx_load, LoRALinear = load, _L


def _adapter_dir():
    d = Path.home() / ".nudge" / "adapters"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _apply_lora(model, rank=16):
    # swap all Linear layers to LoRA, including inside lists
    # bug fix: vanilla recursion skipped list children (e.g. model.layers),
    # meaning LoRA was applied to almost nothing. now we handle lists too.
    def _replace(module):
        for name, child in module.children().items():
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear.from_linear(child, r=rank))
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, nn.Module):
                        _replace(item)
            elif isinstance(child, nn.Module):
                _replace(child)
    _replace(model)
    return model


def load_model(model_name, lora_rank=16, adapter_path=None):
    _ensure_mlx()
    model, tokenizer = mlx_load(model_name)
    model = _apply_lora(model, rank=lora_rank)
    if adapter_path and Path(adapter_path).exists():
        weights = mx.load(adapter_path)
        lora_w = {k: v for k, v in weights.items() if "lora" in k.lower()}
        if lora_w:
            model.load_weights(list(lora_w.items()))
    model.eval()
    return model, tokenizer


def _compute_logprobs(model, input_ids):
    """Forward pass → per-token log probs."""
    logits = model(input_ids[None, :]).squeeze(0)
    lp = nn.log_softmax(logits, axis=-1)
    tok_lp = mx.take_along_axis(lp[:-1], input_ids[1:, None], axis=-1).squeeze(-1)
    mx.eval(tok_lp)
    return tok_lp


def _tokenize(tokenizer, item):
    """Prompt+response → input_ids + response_start index."""
    _ensure_mlx()
    p, r = item["prompt"], item["response"]
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = []
        if item.get("context"):
            try:
                ctx = json.loads(item["context"])
                if isinstance(ctx, dict) and "system" in ctx:
                    msgs.append({"role": "system", "content": ctx["system"]})
            except (json.JSONDecodeError, TypeError):
                pass
        msgs.append({"role": "user", "content": p})
        prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        full_ids = tokenizer.apply_chat_template(
            msgs + [{"role": "assistant", "content": r}], add_generation_prompt=False
        )
    else:
        prompt_ids = tokenizer.encode(p)
        full_ids = tokenizer.encode(p + r)
    return {
        "input_ids": mx.array(full_ids),
        "response_start": max(1, len(prompt_ids)),
        "rating": item["rating"],
        "id": item["id"],
    }


def _make_loss_fn(tokenized, ref_logprobs, cfg, ema_mean):
    # loss function — idx picks which sample from the batch
    tc_lo, tc_hi = cfg["token_clip"]
    tj_lo, tj_hi = cfg["traj_clip"]
    kl_c, pw = cfg["kl_coeff"], cfg["pos_weight"]

    def loss_fn(model, idx: int):
        t = tokenized[idx]
        ids, start, rating = t["input_ids"], t["response_start"], t["rating"]
        ref = ref_logprobs[idx]

        logits = model(ids[None, :]).squeeze(0)
        new_lp = mx.take_along_axis(
            nn.log_softmax(logits, axis=-1)[:-1], ids[1:, None], axis=-1
        ).squeeze(-1)

        # response region only — prompt tokens don't get gradients
        new_r, ref_r = new_lp[start - 1:], ref[start - 1:]
        ratio = mx.exp(new_r - ref_r)

        # token clip: kill tokens that moved too far
        mask = mx.logical_and(ratio >= tc_lo, ratio <= tc_hi).astype(mx.float32)
        clipped = mx.clip(ratio, tc_lo, tc_hi)

        # trajectory clip: geometric mean of all ratios, gate the whole response
        traj = mx.exp(mx.mean(new_r - ref_r))
        gate = mx.logical_and(traj >= tj_lo, traj <= tj_hi).astype(mx.float32)

        # how much better/worse than average was this response
        adv = rating - ema_mean
        if rating > 0:
            adv *= pw  # boost rare positive signal

        n = mx.maximum(mx.sum(mask), mx.array(1.0))
        actor = -mx.sum(clipped * adv * gate * mask) / n

        # KL penalty: forward KL approximation. always non-negative, zero when new==ref
        kl = mx.exp(ref_r - new_r) - (ref_r - new_r) - 1
        kl_loss = mx.sum(kl * mask) / n

        return actor + kl_c * kl_loss

    return loss_fn


def train(config, conn):
    # one round of training
    _ensure_mlx()
    batch_min = config.get("batch_min", 16)
    if db.count_trainable_untrained(conn) < batch_min:
        return None

    model_name = config["model"]
    cfg = {k: config.get(k, d) for k, d in [
        ("lora_rank", 16), ("steps", 8), ("lr", 4e-6),
        ("token_clip", [0.5, 2.0]), ("traj_clip", [0.992, 1.002]),
        ("kl_coeff", 0.001), ("grad_accum", 4), ("grad_clip", 1.0),
        ("ema_decay", 0.99), ("pos_weight", 1.2), ("adapter_keep", 20),
        ("replay_ratio", 0.5),
    ]}

    # build batch: configurable fresh/replay mix
    replay_n = min(int(round(batch_min * cfg["replay_ratio"])), batch_min)
    replay = db.get_replay(conn, limit=replay_n)
    fresh = db.get_untrained(conn, limit=batch_min - len(replay))
    batch = fresh + replay
    fresh_ids = [f["id"] for f in fresh]  # track these BEFORE tokenizing

    # step 1: ref logprobs from base model (no LoRA)
    latest = db.latest_adapter(conn)
    base, tokenizer = mlx_load(model_name)
    base.eval()
    tokenized = [_tokenize(tokenizer, item) for item in batch]
    ref_lps = [_compute_logprobs(base, t["input_ids"]) for t in tokenized]
    del base  # free VRAM

    # step 2: policy model (with LoRA)
    policy, _ = load_model(model_name, cfg["lora_rank"],
                           latest["path"] if latest else None)
    policy.train()
    opt = optim.Adam(learning_rate=cfg["lr"])

    ema_mean, ema_count = db.get_ema(conn)
    loss_fn = _make_loss_fn(tokenized, ref_lps, cfg, ema_mean)

    # nn.value_and_grad captures the model — DON'T pass it again when calling
    vg = nn.value_and_grad(policy, loss_fn)

    total_loss, total_steps = 0.0, 0
    for step in range(cfg["steps"]):
        acc_loss, acc_grads = 0.0, None
        for micro in range(cfg["grad_accum"]):
            idx = (step * cfg["grad_accum"] + micro) % len(tokenized)
            loss, grads = vg(idx)  # NOT vg(policy, idx) — model already captured
            mx.eval(loss)
            acc_loss += loss.item()
            acc_grads = grads if acc_grads is None else mx.tree_map(
                lambda a, b: a + b, acc_grads, grads)

        acc_grads = mx.tree_map(lambda g: g / cfg["grad_accum"], acc_grads)

        # gradient clipping — no cpu-gpu sync, just multiply (1.0 when under limit)
        flat = [g for _, g in nn.utils.tree_flatten(acc_grads)]
        if flat:
            norm = mx.sqrt(sum(mx.sum(g * g) for g in flat))
            scale = mx.minimum(mx.array(cfg["grad_clip"]) / (norm + 1e-6), mx.array(1.0))
            acc_grads = mx.tree_map(lambda g: g * scale, acc_grads)

        opt.update(policy, acc_grads)
        mx.eval(policy.parameters())
        total_loss += acc_loss / cfg["grad_accum"]
        total_steps += 1

    # update EMA from fresh items only — replays already counted
    for item in fresh:
        ema_count += 1
        ema_mean = cfg["ema_decay"] * ema_mean + (1 - cfg["ema_decay"]) * item["rating"]
    db.update_ema(conn, ema_mean, ema_count)

    # save adapter
    parent_v = latest["version"] if latest else None
    new_v = (parent_v or 0) + 1
    save_dir = _adapter_dir() / f"v{new_v}"
    save_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = str(save_dir / "adapter.safetensors")

    lora_w = {k: v for k, v in nn.utils.tree_flatten(policy.trainable_parameters())}
    mx.save_safetensors(adapter_file, lora_w)

    # also save adapter_config.json — vLLM needs this
    adapter_cfg = {"r": cfg["lora_rank"], "lora_alpha": cfg["lora_rank"],
                   "target_modules": "all", "base_model": model_name}
    (save_dir / "adapter_config.json").write_text(json.dumps(adapter_cfg))

    metrics = {"avg_loss": total_loss / max(total_steps, 1),
               "batch_size": len(batch), "steps": cfg["steps"], "ema_mean": ema_mean}
    db.add_adapter(conn, new_v, adapter_file, parent_v, metrics)
    db.mark_trained(conn, fresh_ids, new_v)

    # only cleanup if user set a limit — default keeps everything for rollback
    if cfg.get("adapter_keep"):
        for p in db.cleanup_adapters(conn, keep=cfg["adapter_keep"]):
            d = Path(p).parent
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)

    del policy
    return metrics


def _convert_to_gguf(adapter_dir):
    """Auto-convert safetensors to GGUF for Ollama. Returns GGUF path or None."""
    import subprocess
    safetensors_file = Path(adapter_dir) / "adapter.safetensors"
    gguf_file = Path(adapter_dir) / "adapter.gguf"
    if gguf_file.exists():
        return str(gguf_file)
    if not safetensors_file.exists():
        return None
    # try llama.cpp convert tool — installed via pip install llama-cpp-python or brew
    for cmd in ["convert-lora-to-gguf", "python3 -m llama_cpp.convert_lora"]:
        try:
            result = subprocess.run(
                cmd.split() + [str(safetensors_file), "--outfile", str(gguf_file)],
                capture_output=True, timeout=120,
            )
            if result.returncode == 0 and gguf_file.exists():
                return str(gguf_file)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def hot_swap(server_type, adapter_path, model_name):
    # push adapter to whatever server is running
    import requests
    adapter_dir = str(Path(adapter_path).parent)

    if server_type == "ollama":
        # ollama needs GGUF — auto-convert from safetensors
        gguf = _convert_to_gguf(adapter_dir)
        adapter_ref = gguf if gguf else adapter_dir
        mf = f"FROM {model_name}\nADAPTER {adapter_ref}\n"
        try:
            r = requests.post("http://localhost:11434/api/create",
                              json={"name": f"{model_name}-nudge", "modelfile": mf}, timeout=60)
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    elif server_type == "lmstudio":
        # LM Studio uses OpenAI-compatible API on port 1234
        # load LoRA by placing adapter in ~/.cache/lm-studio/adapters/ and reloading
        try:
            r = requests.post("http://localhost:1234/v1/lora/load",
                              json={"path": adapter_dir}, timeout=30)
            return r.status_code == 200
        except requests.ConnectionError:
            # fallback: just tell them where the adapter is
            print(f"Load adapter in LM Studio from: {adapter_dir}")
            return True

    elif server_type == "vllm":
        try:
            r = requests.post("http://localhost:8000/v1/load_lora_adapter",
                              json={"lora_name": "nudge", "lora_path": adapter_dir}, timeout=30)
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    else:
        print(f"Restart your server with adapter: {adapter_path}")
        return True
