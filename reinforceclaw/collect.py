"""Prompt generation + human rating helpers."""

from __future__ import annotations

import atexit
import json
from itertools import cycle
from pathlib import Path

import requests

from .backend_mlx import mlx_drain

PROMPT_BANK = {
    "code": [
        "Refactor this Python function to be shorter and clearer without changing behavior.",
        "Find the likely bug in this code and explain the minimal fix.",
        "Write a SQL query that returns the top 5 users by total spend.",
        "Explain what this bash command does and point out any risk.",
        "Write a Python function that deduplicates a list while preserving order.",
        "Given a failing test, suggest the smallest code change that would likely fix it.",
        "Review this API design and name the biggest flaw first.",
        "Write a regex that matches semantic versions like 1.2.3.",
        "Explain why this code is hard to maintain and how to improve it.",
        "Write a small TypeScript type for a user profile object.",
    ],
    "math": [
        "What is 17 multiplied by 23?",
        "If a train goes 60 mph for 2.5 hours, how far does it travel?",
        "What is 15 percent of 240?",
        "Solve: 3x + 7 = 22.",
        "What is the next number: 2, 4, 8, 16, ?",
        "What is the area of a rectangle that is 7 by 9?",
        "What is 0.25 as a percentage?",
        "If you buy 3 items at $19 each, what is the total before tax?",
        "What is the square root of 144?",
        "Sort these numbers from smallest to largest: 9, 2, 14, 7.",
    ],
    "instructions": [
        "Answer in exactly two sentences: what is an API?",
        "Give me three bullet points about why tests matter.",
        "Do not explain, just return a JSON object with keys name and age.",
        "Answer briefly: what does CPU stand for?",
        "Summarize recursion in one paragraph only.",
        "Count from 1 to 5, one per line.",
        "Translate 'thank you' into Spanish and nothing else.",
        "List exactly four programming languages.",
        "Reply with only the word YES if 7 is prime, otherwise NO.",
        "Explain caching in simple terms without using jargon.",
    ],
    "personality": [
        "A user says: you wasted my time. Reply briefly, directly, and responsibly.",
        "A user asks for blunt feedback on a weak product idea. Give it honestly but usefully.",
        "A user is frustrated that your answer ignored their instructions. Respond in a concise accountable way.",
        "A user wants a short explanation with no fluff. Explain what a database is.",
        "A user asks for the fastest path to debug a bug. Answer directly and pragmatically.",
        "A user says your code style is messy. Reply without being defensive.",
        "A user asks you to choose between two technical options. Give a decisive answer and one reason.",
        "A user wants a simple explanation of reinforcement learning with no jargon.",
        "A user asks for risks first, not a sales pitch. Review the idea of training in production.",
        "A user says: stop rambling and answer the question. Show the improved style.",
    ],
}

_LOCAL_MLX = {"key": None, "model": None, "tokenizer": None}


def normalize_topics(raw: str | None) -> list[str]:
    if not raw:
        return list(PROMPT_BANK)
    wanted = []
    for part in raw.split(","):
        topic = part.strip().lower()
        if topic in PROMPT_BANK and topic not in wanted:
            wanted.append(topic)
    return wanted or list(PROMPT_BANK)


def sample_prompts(count: int, topics: list[str]) -> list[dict]:
    pools = [PROMPT_BANK[topic] for topic in topics]
    merged = []
    for group in zip(*[cycle(pool) for pool in pools]):
        for prompt in group:
            merged.append(prompt)
            if len(merged) >= count:
                return [{"topic": topics[i % len(topics)], "prompt": text} for i, text in enumerate(merged)]
    return [{"topic": topics[i % len(topics)], "prompt": text} for i, text in enumerate(merged[:count])]


def load_prompts(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            row = json.loads(line)
            rows.append({"topic": row.get("topic", "custom"), "prompt": row["prompt"]})
        else:
            rows.append({"topic": "custom", "prompt": line})
    return rows


def save_prompts(path: str | Path, prompts: list[dict]) -> None:
    out = []
    for item in prompts:
        out.append(json.dumps({"topic": item["topic"], "prompt": item["prompt"]}, ensure_ascii=True))
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


def _openai_chat(
    base_url: str,
    model: str,
    messages: list[dict],
    api_key: str | None = None,
    timeout: int = 120,
    temperature: float = 0.7,
) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _ollama_chat(base_url: str, model: str, messages: list[dict], timeout: int = 120, temperature: float = 0.7) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"].strip()

def _clear_local_mlx() -> None:
    _LOCAL_MLX.update({"key": None, "model": None, "tokenizer": None})
    mlx_drain(collect_garbage=True)


atexit.register(_clear_local_mlx)


def ollama_available(base_url: str, timeout: int = 2) -> bool:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def _mlx_chat(model: str, messages: list[dict], adapter_path: str | None = None, lora_rank: int = 8, max_tokens: int = 256) -> str:
    from . import trainer

    key = (model, adapter_path or "", int(lora_rank))
    if _LOCAL_MLX["key"] != key:
        _clear_local_mlx()
        from mlx_lm import generate

        loaded_model, tokenizer = trainer.load_model(model, lora_rank=lora_rank, adapter_path=adapter_path)
        _LOCAL_MLX.update({"key": key, "model": loaded_model, "tokenizer": tokenizer, "generate": generate})
    tokenizer = _LOCAL_MLX["tokenizer"]
    prompt = (
        trainer._apply_chat_template(tokenizer, messages, add_generation_prompt=True, tokenize=False)
        if hasattr(tokenizer, "apply_chat_template")
        else "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    )
    text = _LOCAL_MLX["generate"](_LOCAL_MLX["model"], tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False).strip()
    mlx_drain()
    return text


def chat(
    server: str,
    model: str,
    messages: list[dict],
    base_url: str,
    api_key: str | None = None,
    timeout: int = 120,
    temperature: float = 0.7,
    adapter_path: str | None = None,
    lora_rank: int = 8,
    max_tokens: int = 256,
) -> str:
    if server == "mlx":
        return _mlx_chat(model, messages, adapter_path=adapter_path, lora_rank=lora_rank, max_tokens=max_tokens)
    if server == "ollama":
        return _ollama_chat(base_url, model, messages, timeout=timeout, temperature=temperature)
    return _openai_chat(base_url, model, messages, api_key=api_key, timeout=timeout, temperature=temperature)


def continue_conversation(
    model: str,
    transcript: list[dict],
    server: str,
    base_url: str,
    api_key: str | None = None,
) -> str:
    return chat(
        server,
        model,
        [
            {
                "role": "system",
                "content": (
                    "You generate the next user message in a realistic evaluation conversation. "
                    "Return only the next user message. No commentary."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Conversation so far:\n"
                    + "\n".join(f"{m['role'].upper()}: {m['content']}" for m in transcript)
                    + "\n\nWrite the next user turn only."
                ),
            },
        ],
        base_url=base_url,
        api_key=api_key,
        temperature=0.6,
    ).strip()


def judge_response(
    model: str,
    prompt: str,
    response: str,
    server: str,
    base_url: str,
    api_key: str | None = None,
    transcript: list[dict] | None = None,
) -> str:
    content = [
        "You are a strict but practical evaluator for assistant behavior.",
        "Return exactly one word: good, bad, or ignore.",
        "Use ignore if the example is ambiguous, mixed, or not informative.",
        "",
        f"Prompt:\n{prompt}",
        "",
        f"Response:\n{response}",
    ]
    if transcript:
        content.extend(["", "Transcript:", *[f"{m['role'].upper()}: {m['content']}" for m in transcript]])
    raw = chat(
        server,
        model,
        [
            {"role": "system", "content": "Return exactly one of: good, bad, ignore."},
            {"role": "user", "content": "\n".join(content)},
        ],
        base_url=base_url,
        api_key=api_key,
        temperature=0.0,
    ).strip().lower()
    for label in ("good", "bad", "ignore"):
        if label in raw:
            return label
    return "ignore"


def flatten_transcript(messages: list[dict]) -> tuple[str, str]:
    indices = [i for i, msg in enumerate(messages) if msg["role"] == "assistant"]
    if not indices:
        return "", ""
    assistant_idx = max(indices)
    prompt = []
    for msg in messages[:assistant_idx]:
        role = "User" if msg["role"] == "user" else "Assistant"
        prompt.append(f"{role}: {msg['content']}")
    return "\n".join(prompt).strip(), messages[assistant_idx]["content"].strip()


def helper_generate_prompts(
    model: str,
    count: int,
    topics: list[str],
    server: str,
    base_url: str,
    api_key: str | None = None,
) -> list[dict]:
    prompts = []
    styles = cycle(
        (
            "short and direct",
            "slightly tricky but fair",
            "instruction-following focused",
            "style and tone focused",
            "coding focused",
        )
    )
    topic_cycle = cycle(topics)
    for _ in range(count):
        topic = next(topic_cycle)
        style = next(styles)
        content = chat(
            server,
            model,
            [
                {
                    "role": "system",
                    "content": (
                        "You create standalone evaluation prompts for an AI assistant. "
                        "Return only one prompt. No numbering. No explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Write one {style} {topic} prompt for evaluating an assistant. "
                        "It should be answerable without external files or images."
                    ),
                },
            ],
            base_url=base_url,
            api_key=api_key,
            temperature=0.7,
        )
        prompts.append({"topic": topic, "prompt": content.strip()})
    return prompts
