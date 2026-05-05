"""Per-profile training presets based on the Gemma-4-31B Modal sweep."""

from __future__ import annotations

from .profile import ModelProfile

ANCHOR_BALANCED = 5e-6  # Gemma-4-31B dense balanced (our sweep)

_SIZE_MULT = {
    "tiny":  1.6,   # <= 2B active, flatter logits
    "small": 1.4,   # 2-8B
    "mid":   1.2,   # 8-20B
    "large": 1.0,   # 20-45B   <-- anchor
    "xl":    0.8,   # 45-100B dense, peakier logits
}


def _scale_cut(total_b: float) -> float:
    """Softmax sharpening + router/depth variance at very large total sizes."""
    for threshold, mult in ((100, 1.0), (250, 0.60), (500, 0.45)):
        if total_b <= threshold:
            return mult
    return 0.35


def _moe_mult(kind: str, total_b: float, active_b: float) -> float:
    """Sparsity-driven cut on attention-LoRA LR."""
    if kind != "moe":
        return 1.0
    sparsity = total_b / max(active_b, 1.0)
    for threshold, mult in ((4, 0.95), (10, 0.85), (20, 0.70)):
        if sparsity <= threshold:
            return mult
    return 0.55

_PRESETS = {
    "careful": {"lr_mult": 0.6, "kl": 0.0020, "steps": 24, "traj_clip": [0.996, 1.001]},
    "balanced": {"lr_mult": 1.0, "kl": 0.0010, "steps": 32, "traj_clip": [0.994, 1.002]},
    "aggressive": {"lr_mult": 1.6, "kl": 0.0005, "steps": 48, "traj_clip": [0.992, 1.004]},
}
_BUCKETS = {
    "tiny": {"batch_min": 30, "batch_size": 8, "grad_accum": 1, "rank": 16},
    "small": {"batch_min": 30, "batch_size": 6, "grad_accum": 1, "rank": 16},
    "mid": {"batch_min": 32, "batch_size": 4, "grad_accum": 2, "rank": 16},
    "large": {"batch_min": 36, "batch_size": 3, "grad_accum": 2, "rank": 16},
    "xl": {"batch_min": 40, "batch_size": 2, "grad_accum": 4, "rank": 8},
}


def pick(profile: ModelProfile, preset: str = "balanced") -> dict:
    """Return a full training config for this (profile, preset)."""
    preset = preset if preset in _PRESETS else "balanced"
    bucket = profile.size_bucket if profile.size_bucket in _SIZE_MULT else "mid"
    p, b = _PRESETS[preset], _BUCKETS[bucket]
    lr = (ANCHOR_BALANCED
          * _SIZE_MULT[bucket]
          * _scale_cut(profile.total_b)
          * _moe_mult(profile.kind, profile.total_b, profile.active_b)
          * p["lr_mult"])
    kl = p["kl"] * (0.5 if profile.kind == "moe" else 1.0)
    rank = b["rank"] + (8 if profile.kind == "moe" else 0)

    return {
        "lr": _round(lr),
        "kl_coeff": _round(kl, 6),
        "lora_rank": rank,
        "lora_alpha": rank,
        "lora_target": "attention",
        "batch_min": b["batch_min"],
        "batch_size": b["batch_size"],
        "grad_accum": b["grad_accum"],
        "steps": p["steps"],
        "traj_clip": list(p["traj_clip"]),
        "token_clip": [0.5, 2.0],
        "pos_weight": 1.2 if preset != "careful" else 1.0,
        "replay_ratio": 0.0,
        "ema_decay": 0.99,
        "model_profile": profile.as_dict(),
        "tuning_mode": "auto",
    }


def _round(x: float, sig: int = 7) -> float:
    return float(f"{x:.{sig}g}") if x > 0 else 0.0
