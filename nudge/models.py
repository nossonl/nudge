"""Supported models. Newest first within each company."""
# any HuggingFace model works via "Other" in the wizard
# want your model here? open a PR

MODELS = {
    "Qwen": [
        # 3.5
        "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-122B-A10B", "Qwen/Qwen3.5-397B-A17B",
        # 3
        "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B",
        "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B",
        # 2.5
        "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-72B-Instruct",
        # coder + reasoning
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/QwQ-32B",
    ],
    "Meta": [
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct",
    ],
    "Mistral": [
        "mistralai/Mistral-Small-4-119B-2603",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "mistralai/Mistral-Large-Instruct-2411",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Codestral-22B-v0.1",
        "mistralai/Devstral-Small-2505",
        "mistralai/Magistral-Small-2506",
    ],
    "Google": [
        "google/gemma-3-1b-it", "google/gemma-3-4b-it",
        "google/gemma-3-12b-it", "google/gemma-3-27b-it",
        "google/gemma-3n-E2B-it", "google/gemma-3n-E4B-it",
        "google/gemma-2-9b-it", "google/gemma-2-27b-it",
    ],
    "Microsoft": [
        "microsoft/Phi-4-mini-instruct", "microsoft/Phi-4-mini-reasoning",
        "microsoft/phi-4", "microsoft/Phi-4-reasoning",
        "microsoft/Phi-3.5-mini-instruct", "microsoft/Phi-3.5-MoE-instruct",
    ],
    "DeepSeek": [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    ],
    "GLM": [
        "zai-org/GLM-5", "zai-org/GLM-4.7-Flash", "zai-org/GLM-4.7",
        "zai-org/GLM-4.6", "zai-org/GLM-4.5", "zai-org/GLM-4.5-Air",
    ],
    "Moonshot": [
        "moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2-Instruct",
        "moonshotai/Kimi-Dev-72B", "moonshotai/Moonlight-16B-A3B-Instruct",
    ],
    "MiniMax": [
        "MiniMaxAI/MiniMax-M2.5", "MiniMaxAI/MiniMax-M2.1",
        "MiniMaxAI/MiniMax-M2", "MiniMaxAI/MiniMax-Text-01-hf",
    ],
    "Nvidia": [
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        "nvidia/OpenReasoning-Nemotron-32B",
    ],
    "Cohere": [
        "CohereLabs/c4ai-command-a-03-2025",
        "CohereLabs/c4ai-command-r7b-12-2024",
        "CohereLabs/aya-expanse-8b", "CohereLabs/aya-expanse-32b",
    ],
    "Yi": ["01-ai/Yi-1.5-9B-Chat", "01-ai/Yi-1.5-34B-Chat", "01-ai/Yi-Coder-9B-Chat"],
    "InternLM": ["internlm/internlm3-8b-instruct", "internlm/internlm2_5-7b-chat"],
    "Falcon": ["tiiuae/Falcon3-7B-Instruct", "tiiuae/Falcon3-10B-Instruct"],
    "AllenAI": ["allenai/Olmo-3-7B-Instruct", "allenai/Olmo-3.1-32B-Instruct"],
}
