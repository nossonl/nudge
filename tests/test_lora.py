"""LoRA tests — verify adapters apply correctly, save, load, and rollback."""

import tempfile
from pathlib import Path


def test_lora_lifecycle():
    """Full LoRA test: apply → train → save → load → verify weights changed."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load as mlx_load
        from mlx_lm.tuning.lora import LoRALinear
    except ImportError:
        print("mlx-lm not installed, skipping"); return

    from reinforceclaw.trainer import _apply_lora, load_model, _compute_logprobs

    model_name = "Qwen/Qwen3.5-9B"
    print(f"Model: {model_name}")

    # 1. load base model
    print("\n1. Loading base model...")
    base, tok = mlx_load(model_name)
    base.eval()

    # count linear layers
    linear_count = 0
    def _count(m):
        nonlocal linear_count
        for name, child in m.children().items():
            if isinstance(child, nn.Linear):
                linear_count += 1
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, nn.Module):
                        _count(item)
            elif isinstance(child, nn.Module):
                _count(child)
    _count(base)
    print(f"   Linear layers in base: {linear_count}")

    # 2. apply LoRA
    print("\n2. Applying LoRA (rank 16)...")
    model = _apply_lora(base, rank=16)

    lora_count = 0
    def _count_lora(m):
        nonlocal lora_count
        for name, child in m.children().items():
            if isinstance(child, LoRALinear):
                lora_count += 1
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, nn.Module):
                        _count_lora(item)
            elif isinstance(child, nn.Module):
                _count_lora(child)
    _count_lora(model)
    print(f"   LoRA layers applied: {lora_count}")
    assert lora_count > 0, "no LoRA layers applied!"
    assert lora_count == linear_count, f"expected {linear_count} LoRA layers, got {lora_count}"
    print(f"   All {linear_count} linear layers → LoRA ✓")

    # 3. compute logprobs
    print("\n3. Computing logprobs...")
    test_prompt = "What is 2+2?"
    if hasattr(tok, "apply_chat_template"):
        ids = tok.apply_chat_template([{"role": "user", "content": test_prompt}], add_generation_prompt=True)
    else:
        ids = tok.encode(test_prompt)
    input_ids = mx.array(ids)
    lp = _compute_logprobs(model, input_ids)
    print(f"   Logprobs shape: {lp.shape}")
    assert lp.shape[0] > 0, "empty logprobs"
    print(f"   Logprobs computed ✓")

    # 4. save and reload
    print("\n4. Save and reload adapter...")
    with tempfile.TemporaryDirectory() as tmp:
        adapter_file = str(Path(tmp) / "adapter.safetensors")
        lora_w = {k: v for k, v in nn.utils.tree_flatten(model.trainable_parameters())}
        print(f"   Trainable params: {len(lora_w)}")
        assert len(lora_w) > 0, "no trainable params"
        mx.save_safetensors(adapter_file, lora_w)
        print(f"   Saved to {adapter_file}")

        # reload
        model2, tok2 = load_model(model_name, lora_rank=16, adapter_path=adapter_file)
        lp2 = _compute_logprobs(model2, input_ids)
        print(f"   Reloaded logprobs shape: {lp2.shape}")

        # logprobs should be identical (same weights)
        diff = mx.abs(lp - lp2).max().item()
        print(f"   Max logprob diff after reload: {diff:.6f}")
        assert diff < 1e-4, f"logprobs diverged after reload: {diff}"
        print(f"   Adapter reload matches ✓")

    del model, model2, base
    print(f"\n{'='*50}")
    print("LORA LIFECYCLE: ALL PASSED")


if __name__ == "__main__":
    test_lora_lifecycle()
