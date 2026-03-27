from reinforceclaw import collect


def test_normalize_topics_filters_unknown_values():
    assert collect.normalize_topics("code,math,unknown") == ["code", "math"]
    assert set(collect.normalize_topics(None)) == set(collect.PROMPT_BANK)


def test_sample_prompts_returns_requested_count():
    prompts = collect.sample_prompts(7, ["code", "math"])
    assert len(prompts) == 7
    assert all("prompt" in item and "topic" in item for item in prompts)


def test_load_and_save_prompts_roundtrip(tmp_path):
    path = tmp_path / "prompts.jsonl"
    prompts = [{"topic": "code", "prompt": "Fix this bug."}, {"topic": "math", "prompt": "What is 2+2?"}]
    collect.save_prompts(path, prompts)
    assert collect.load_prompts(path) == prompts


def test_flatten_transcript_uses_last_assistant_turn():
    prompt, response = collect.flatten_transcript(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Solve 2+2"},
            {"role": "assistant", "content": "4"},
        ]
    )
    assert prompt == "User: Hi\nAssistant: Hello\nUser: Solve 2+2"
    assert response == "4"
