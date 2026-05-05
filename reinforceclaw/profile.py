"""Detect model architecture (dense vs MoE), family, and size, then pick configs.

Two-stage detection: read the HF config.json if available (authoritative) and fall
back to name parsing (works for repos we haven't fetched yet). Output feeds
``presets.pick`` to auto-select LR, KL, LoRA rank, batch-min without user tuning.

Works across any provider/platform the user might name a model from:
HuggingFace (``org/Model-7B``), community forks (``unsloth/…``, ``mlx-community/…``),
LM Studio paths, local directories, Ollama tags (``qwen2.5:7b``), llama.cpp GGUF
filenames, and closed APIs. Ollama/GGUF/API targets are flagged
``trainable=False`` rather than crashing the trainer.

Config calibration is grounded in the modal sweeps:
- Dense Gemma-4-31B, lr=8e-6 kl=0.0005 rank=16 attention-only -> +3.9% (baseline winner)
- Dense Gemma-4-31B, lr=5e-6 kl=0.001 (balanced default) -> stable, p=1.000
- MoE variants: ``all_linear`` target never beat attention-only; attention-only with
  lower KL + higher rank is the safer MoE starting point.

Smaller models tolerate a little more LR; larger dense and high-sparsity MoE
models need lower LR. The preset picker uses tested buckets and large-model cuts
instead of a sqrt curve from pre-training, because RL LoRA collapse risk stayed
close to flat around the 31B sweep.
"""

from __future__ import annotations

import json
import os
import re
import heapq
from dataclasses import asdict, dataclass
from pathlib import Path

_MOE_KEYS = ("num_local_experts", "num_experts", "moe_num_experts", "n_routed_experts")
_DETECT_CACHE: dict[str, "ModelProfile"] = {}
_DETECT_CACHE_MAX = 64
_MOE_NAME_HINTS = (
    "moe", "mixtral", "16e", "128e", "phi-3.5-moe", "phi-moe",
    "deepseek-v2", "deepseek-v3", "deepseek-v4", "deepseek-r1-671", "deepseek-coder-v2",
    "qwen3-30b-a", "qwen3-235b-a", "qwen3.5-35b-a", "qwen3.5-122b-a", "qwen3.5-397b-a",
    "kimi-k2", "minimax-text", "minimax-m",
    "nemotron-3-super", "nemotron-3-nano-30b-a",
    "olmoe", "granite-3.0-3b-a", "jamba", "arctic-8x", "step-3.5-flash",
)

_FAMILY_PATTERNS = [
    ("qwen",     ("qwen",)),
    ("llama",    ("llama", "meta-llama", "hermes", "tulu")),
    ("mistral",  ("mistral", "mixtral", "codestral", "magistral", "devstral", "ministral", "pixtral")),
    ("gemma",    ("gemma",)),
    ("phi",      ("phi-", "phi_", "phi4", "microsoft/phi")),
    ("deepseek", ("deepseek",)),
    ("glm",      ("glm-", "zai-org/glm", "chatglm")),
    ("kimi",     ("kimi", "moonshot", "moonlight")),
    ("minimax",  ("minimax",)),
    ("nemotron", ("nemotron",)),
    ("cohere",   ("cohere", "c4ai", "aya",)),
    ("yi",       ("01-ai/yi", "/yi-", "yi-1.", "yi-coder")),
    ("falcon",   ("falcon",)),
    ("olmo",     ("olmo",)),
    ("internlm", ("internlm",)),
    ("granite",  ("granite",)),
    ("jamba",    ("jamba",)),
    ("stablelm", ("stablelm", "stable-lm")),
]

# Vendor prefixes to strip when parsing family/size (they don't change the base model).
_VENDOR_PREFIXES = (
    "unsloth/", "mlx-community/", "lmstudio-community/", "lmstudio/",
    "thebloke/", "bartowski/", "nousresearch/", "togethercomputer/",
    "fireworks-ai/", "groq/", "replicate/", "nvidia/", "bigscience/",
    "huggingfaceh4/", "gradientai/", "abacusai/", "cognitivecomputations/",
    "second-state/", "hugging-quants/", "quantfactory/", "prince-canuma/",
    "hf.co/", "hf://",
)

# Quant / format tags on filenames & repo names that should be stripped before parsing.
# NOTE: kept conservative — matches only suffix segments after a separator.
_QUANT_SUFFIX_RE = re.compile(
    r"[-_](?:"
    r"q\d(?:_[kmos](?:_[lmsxl]|_m|_s)?)?"  # q4, q4_k_m, q5_0, q8_0 …
    r"|iq\d_[a-z]+"                         # iq3_xxs, iq2_m …
    r"|[fb]p?16|fp32|bf16|fp8|int8|int4"
    r"|bnb-?(?:4|8)bit|gptq|awq|exl2|eetq|hqq|sq8"
    r"|gguf|mlx|safetensors|pt|pth|bin"
    r"|4bit|8bit|2bit|3bit"
    r"|instruct|chat|base|it|sft|dpo|orpo|rlhf"  # conversational suffixes (keep family match, drop size-confusers)
    r")(?=-|_|$)",
    re.IGNORECASE,
)

# Known-size fallback: for big-name MoEs whose repo ID doesn't include size
# tokens (Kimi, DeepSeek V3/R1, MiniMax M2, Maverick), so the LR picker can
# route them through the >=100B-total band. Matched against the canonical
# lowered name; first hit wins. (total_b, active_b) in billions.
_KNOWN_SIZES = (
    ("deepseek-v4-pro",    (1600.0, 49.0)),
    ("deepseek-v4-flash",  (284.0, 13.0)),
    ("kimi-k2",           (1000.0, 32.0)),
    ("kimi-k3",           (1500.0, 48.0)),
    ("deepseek-v3",       (671.0, 37.0)),
    ("deepseek-r1-0528",  (671.0, 37.0)),
    ("deepseek-r1",       (671.0, 37.0)),
    ("deepseek-coder-v2", (236.0, 21.0)),
    ("minimax-m2",        (230.0, 10.0)),
    ("minimax-m3",        (350.0, 14.0)),
    ("minimax-text",      (456.0, 45.9)),
    ("llama-4-maverick",  (400.0, 17.0)),   # 128 experts, 17B active
    ("llama-4-scout",     (109.0, 17.0)),   # 16 experts
    ("grok-1",            (314.0, 86.0)),
    ("arctic",            (480.0, 17.0)),
    ("dbrx",              (132.0, 36.0)),
    ("hy3",               (295.0, 21.0)),
    ("step-3.5-flash",    (321.0, 11.0)),
)

# API-only / cloud models we can't train — flag and move on.
_CLOUD_PATTERNS = (
    ("openai",    ("openai/", "gpt-3.5", "gpt-4", "gpt-5", "o1-", "o3-", "o4-")),
    ("anthropic", ("anthropic/", "claude-1", "claude-2", "claude-3", "claude-4", "claude-5", "claude-haiku", "claude-sonnet", "claude-opus")),
    ("google",    ("gemini-", "bison-", "palm-", "google/gemini")),
    ("cohere",    ("cohere/", "command-r", "command-a", "command-nightly")),
    ("aws",       ("bedrock/", "aws/")),
    ("azure",     ("azure-openai/",)),
    ("xai",       ("xai/", "grok-")),
    ("perplexity",("perplexity/", "pplx-")),
)


@dataclass(frozen=True)
class ModelProfile:
    kind: str           # "dense" | "moe" | "cloud"
    family: str         # "qwen" | "llama" | ... | "unknown"
    size_bucket: str    # "tiny" | "small" | "mid" | "large" | "xl" | "unknown"
    total_b: float      # total params in B (best estimate)
    active_b: float     # active params per forward in B (== total_b for dense)
    provider: str       # "local" | "hf" | "ollama" | "mlx" | "unsloth" | "gguf" | "lmstudio" | "cloud-<vendor>"
    trainable: bool     # False for Ollama/GGUF/cloud/API names we can't fine-tune locally
    source: str         # "hf_config" | "name_only" | "mixed" | "cloud_name"

    def as_dict(self) -> dict:
        return asdict(self)


def detect(model_name_or_path: str) -> ModelProfile:
    """Best-effort profile. Never raises. Config trumps name when both available."""
    raw = (model_name_or_path or "").strip()
    if raw in _DETECT_CACHE:
        _DETECT_CACHE[raw] = _DETECT_CACHE.pop(raw)
        return _DETECT_CACHE[raw]
    cloud = _detect_cloud(raw.lower())
    if cloud:
        prof = ModelProfile(kind="cloud", family=cloud, size_bucket="unknown",
                            total_b=0.0, active_b=0.0, provider=f"cloud-{cloud}",
                            trainable=False, source="cloud_name")
        return _cache_profile(raw, prof)

    cfg = _load_hf_config(raw) or {}
    name_bits = _from_name(raw)
    provider = _detect_provider(raw)
    kind = _kind(cfg) or name_bits["kind"]
    family = name_bits["family"]
    param_b = _params_from_config(cfg)
    total_b = param_b or name_bits["total_b"]
    if cfg:
        active_b = _active_from_config(cfg, total_b, kind == "moe") or total_b
    else:
        active_b = name_bits["active_b"] or total_b
    bucket = _bucket(active_b) if active_b else "unknown"
    source = "hf_config" if cfg else "name_only"
    if cfg and not param_b:
        source = "mixed"
    prof = ModelProfile(kind=kind, family=family, size_bucket=bucket,
                        total_b=float(total_b), active_b=float(active_b),
                        provider=provider, trainable=(provider not in {"ollama", "gguf"}), source=source)
    return _cache_profile(raw, prof)


def _cache_profile(key: str, profile: ModelProfile) -> ModelProfile:
    if len(_DETECT_CACHE) >= _DETECT_CACHE_MAX:
        _DETECT_CACHE.pop(next(iter(_DETECT_CACHE)))
    _DETECT_CACHE[key] = profile
    return profile


def _detect_cloud(low: str) -> str | None:
    repo_like = "/" in low
    for vendor, patterns in _CLOUD_PATTERNS:
        for pattern in patterns:
            if repo_like:
                if "/" in pattern and low.startswith(pattern):
                    return vendor
            elif pattern in low:
                return vendor
    return None


def _detect_provider(raw: str) -> str:
    low = raw.lower()
    if re.search(r"(?:^|[-_/.])gguf(?:[-_./]|$)", low):
        return "gguf"
    if ":" in raw and "/" not in raw.split(":", 1)[0]:
        return "ollama"
    if low.startswith(("mlx-community/", "mlx/")):
        return "mlx"
    if low.startswith("unsloth/"):
        return "unsloth"
    if low.startswith(("lmstudio-community/", "lmstudio/")):
        return "lmstudio"
    if Path(raw).is_absolute() or raw.startswith("./") or raw.startswith("~"):
        return "local"
    if "/" in raw:
        return "hf"
    return "local"


def _load_hf_config(model_name_or_path: str) -> dict | None:
    for candidate in _config_candidates(model_name_or_path):
        try:
            return json.loads(Path(candidate).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


def _config_candidates(name: str):
    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    p = Path(name).expanduser()
    if p.is_dir():
        yield p / "config.json"
    cache_roots = [Path.home() / ".cache" / "huggingface"]
    for env_var in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        root = os.environ.get(env_var)
        if root:
            cache_roots.append(Path(root))
    safe_id = name.replace("/", "--").replace(":", "--")
    for root in cache_roots:
        hub_root = root / "hub" / f"models--{safe_id}"
        snapshots = hub_root / "snapshots"
        if snapshots.exists():
            try:
                snaps = heapq.nlargest(5, snapshots.iterdir(), key=mtime)
            except OSError:
                snaps = ()
            for snap in snaps:
                yield snap / "config.json"
    # LM Studio keeps model dirs under ~/.cache/lm-studio/models/<org>/<repo>
    lmstudio = Path.home() / ".cache" / "lm-studio" / "models"
    if lmstudio.exists() and "/" in name:
        yield lmstudio / name / "config.json"


def _kind(cfg: dict) -> str | None:
    if not cfg:
        return None
    if any(cfg.get(k) for k in _MOE_KEYS):
        return "moe"
    arch = " ".join(cfg.get("architectures") or []).lower()
    if "moe" in arch or "mixtral" in arch or "jamba" in arch:
        return "moe"
    if cfg.get("model_type"):
        return "dense"
    return None


def _params_from_config(cfg: dict) -> float:
    """Rough param count in B from config. Good enough for bucketing."""
    if not cfg:
        return 0.0
    h = _int(cfg.get("hidden_size"))
    l = _int(cfg.get("num_hidden_layers"))
    i = _int(cfg.get("intermediate_size"), 4 * h)
    v = _int(cfg.get("vocab_size"))
    if not (h and l):
        return 0.0
    attn = 4 * h * h
    experts = _int(cfg.get("num_local_experts") or cfg.get("num_experts"), 1)
    ffn_per_expert = 3 * h * i
    per_layer = attn + experts * ffn_per_expert
    return (l * per_layer + 2 * v * h) / 1e9


def _active_from_config(cfg: dict, total_b: float, is_moe: bool) -> float:
    if not (cfg and is_moe and total_b):
        return total_b
    experts = _int(cfg.get("num_local_experts") or cfg.get("num_experts"), 1)
    top_k = _int(cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_token"), 2)
    if experts <= 1:
        return total_b
    h = _int(cfg.get("hidden_size"))
    l = _int(cfg.get("num_hidden_layers"))
    i = _int(cfg.get("intermediate_size"), 4 * h)
    v = _int(cfg.get("vocab_size"))
    if not (h and l):
        return total_b
    active_ffn = min(top_k, experts) * 3 * h * i
    active_per_layer = 4 * h * h + active_ffn
    return (l * active_per_layer + 2 * v * h) / 1e9


def _from_name(name: str) -> dict:
    """Parse size + kind from a model id. Robust to vendor forks and quant tags.

    MoE is inferred from (a) a family hint in ``_MOE_NAME_HINTS``, (b) the Mixtral-style
    ``NxMb`` pattern, or (c) the ``…<total>B-A<active>B`` convention used by Qwen,
    DeepSeek, Nemotron, Granite, etc. — catches unreleased/new names for free.
    """
    low = _canonicalize(name or "")
    # Matches the "<total>B/T-A<active>B/M" MoE convention. We normalize to billions.
    active_pair = re.search(r"(\d+(?:\.\d+)?)([bt])[-_]?a(\d+(?:\.\d+)?)([bm])(?![a-z])", low)
    mixture = re.search(r"(\d+)x(\d+(?:\.\d+)?)b", low)
    is_moe = (any(tag in low for tag in _MOE_NAME_HINTS)
              or bool(mixture)
              or bool(active_pair))
    family = next((f for f, pats in _FAMILY_PATTERNS if any(p in low for p in pats)), "unknown")
    if mixture:
        n_experts, expert_b = int(mixture.group(1)), float(mixture.group(2))
        total_b = round(n_experts * expert_b * 0.9, 1)
        active_b = round(2 * expert_b * 0.65, 1)  # top-2 experts + shared attention
    elif active_pair:
        total_b = _size_to_b(active_pair.group(1), active_pair.group(2))
        active_b = float(active_pair.group(3)) / (1000 if active_pair.group(4) == "m" else 1)
    else:
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*([bt])(?![a-z])", low)
        total_b = max((_size_to_b(value, unit) for value, unit in matches), default=0.0)
        active_match = re.search(r"[-_:/]a(\d+(?:\.\d+)?)b(?![a-z])", low)
        active_b = float(active_match.group(1)) if active_match else total_b
    # Known-size override for big-name MoEs whose repo ID under-states size
    # (Maverick lists `17B` which is the active count, not total; Kimi/DeepSeek
    # list no size at all). When the curated entry is larger than what the regex
    # found, adopt it so the LR picker routes through the total-params band.
    for key, (tot, act) in _KNOWN_SIZES:
        if key in low and tot > total_b:
            total_b, active_b = tot, act
            if tot != act:
                is_moe = True
            break
    return {"kind": "moe" if is_moe else "dense", "family": family,
            "total_b": total_b, "active_b": active_b}


def _size_to_b(value: str, unit: str) -> float:
    return float(value) * (1000 if unit == "t" else 1)


def _int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _canonicalize(name: str) -> str:
    """Strip file extensions, vendor prefixes, ollama tags, and quant suffixes."""
    low = name.lower()
    for ext in (".gguf", ".safetensors", ".bin", ".pt", ".pth"):
        if low.endswith(ext):
            low = low[: -len(ext)]
    for prefix in _VENDOR_PREFIXES:
        if low.startswith(prefix):
            low = low[len(prefix):]
            break
    low = low.rsplit("/", 1)[-1]
    if ":" in low:  # ollama tag form: "qwen2.5:7b" -> "qwen2.5-7b"
        low = low.replace(":", "-")
    # Peel off up to 3 trailing quant/format segments ("...-q4_k_m-gguf").
    for _ in range(3):
        stripped = _QUANT_SUFFIX_RE.sub("", low)
        if stripped == low:
            break
        low = stripped
    return low


def _bucket(active_b: float) -> str:
    if active_b <= 2:
        return "tiny"
    if active_b <= 8:
        return "small"
    if active_b <= 20:
        return "mid"
    if active_b <= 45:
        return "large"
    return "xl"
