# ReinforceClaw
 
Self-improving reinforcement learning for your AI agents. Set it up once, rate responses as you work, and your local model keeps improving in the background.

![ReinforceClaw Diagram](assets/diagram2.png)

## Setup

Install ReinforceClaw from PyPI:

```bash
python3 -m pip install --no-cache-dir reinforceclaw
reinforceclaw init
```

Install the small base package first, then let the wizard add the right ML training backend for your machine. That keeps setup quiet while still supporting MLX on Apple Silicon and CUDA on Linux. If you already know what you want, you can install everything up front with `python3 -m pip install "reinforceclaw[mlx]"` on Apple Silicon or `python3 -m pip install "reinforceclaw[cuda]"` on Linux with CUDA.

Or install ReinforceClaw from npm:

```bash
npm install reinforceclaw
```

The PyPI package is the full local RL system and setup wizard. The npm package is the ReinforceClaw package for JavaScript/OpenClaw plugin installs.

Source checkout:

```bash
git clone https://github.com/nossonl/ReinforceClaw.git
cd ReinforceClaw
python3 -m pip install -e .
reinforceclaw init
```

You can run the wizard yourself or tell your coding agent: "Set up ReinforceClaw for me." Pick your agent, local trainable model, and training preset. After that, just use your agent.

## The problem
 
Ever felt like throwing your computer because the model just will not listen?

You spend time tweaking prompts, editing output, arguing with the model, rewriting instructions, and sometimes it feels like talking to a brick wall. These AI models are not built for you. They are generalized for everybody.

ReinforceClaw puts the control in your hands. With reinforcement learning, you reward what works and punish what does not. Over time, your local model gets optimized toward the way you actually want it to behave.

## Results

I tested this across 80 held-out prompts on both dense and MoE models. These are the results:

| Model | Δ |
|---|---|
| Gemma-4-31B dense | **+3.9%** |
| Gemma-4-31B MoE | **+2.6%** |
 
## How it works
 
Throughout your agent sessions, after each response, you can rate it good, bad, or ignore it. Ignored responses disappear. One tap.
 
```
╭─ Rate this response ───────────╮
│  [1] Good          [2] Bad     │
╰────────────────────────────────╯
```
 
Ratings go into a local database. Once enough build up (**30-40 by default**), a training run kicks off automatically in the background. It trains a LoRA adapter on your local model and prepares it for you.
 
You do not touch training. Once set up, it runs continuously in the background. It just improves.
 
> **⚠️ Rate good responses too.** Bad tends to stand out more, so often you will only rate bad. But only rating bad will not help much, and can even degrade the model. The model needs to understand what you also like, not just what to avoid. Rate both.
 
## Commands
 
Once set up, inside Claude Code, Codex, or any OpenClaw-connected channel, including Telegram, WhatsApp, Slack, and Discord, use `/rl` when typing feedback. In OpenClaw, the setup wizard can also enable real held-message reactions; those rate only the captured AI message they target when OpenClaw forwards the reaction and message id:
 
```
/rl good         # positive reward
/rl bad          # negative reward
/rl undo         # remove last rating
/rl status       # check stats
```

Local terminal commands also support `train`, `rollback`, and `reset`. OpenClaw keeps those admin commands off in channels by default so nobody in Telegram/Slack/WhatsApp can accidentally wipe or change your local training state; owners can opt in with an explicit OpenClaw admin allowlist.
 
From the terminal, same thing but with `reinforceclaw`:
 
```bash
reinforceclaw good   # rates the last captured hook response
reinforceclaw status
reinforceclaw train   # optional
```
 
## Under the hood
 
Uses **MIS-PO** (Metropolis Independence Sampling - Filtered Policy Optimization), the same family of algorithms behind Step 3.5 Flash. Each response is one rollout. Your good/bad rating is the reward. Instead of needing a batch of answers for the same prompt, ReinforceClaw learns from the one answer you actually saw.
 
**Reward:** +1 (good) or -1 (bad).
 
**EMA baseline:** An exponential moving average tracks your rating history. Advantage is computed as:
 
```
advantage = rating - ema_mean
ema_mean = decay * ema_mean + (1 - decay) * rating
```
 
Default decay is `0.99`. If you have been rating mostly good lately, a bad rating stands out more. If you have had a rough streak, a good rating gets more credit.
 
**LoRA adapters:** Base model weights stay frozen. Training only updates low-rank adapter layers. Fast, lightweight, reversible.
 
**Token-level clipping:** For each token, the probability ratio between the new policy and reference policy gets clipped:
 
```
ratio_t = π_new(token_t) / π_ref(token_t)
clipped_t = clip(ratio_t, 0.5, 2.0)
```
 
If a token moves too far, ReinforceClaw uses the clipped ratio for that token's policy update instead of letting one token dominate the step.
 
**Trajectory-level drift control:** The geometric mean of all token ratios across the full response gets capped if the response drift is outside the configured band:
 
```
ρ̄ = exp(mean(log π_new - log π_ref))
trajectory_scale = min(1, ρ̄ / low, high / ρ̄)
```
 
If the whole response drifted too far, the update gets scaled down rather than trusted at full strength.
 
**KL penalty:** Forward-KL divergence keeps the policy from straying too far from the reference model (`β = 0.001`):
 
```
KL = exp(log_ref - log_new) - (log_ref - log_new) - 1
```
 
This is a small penalty for moving too far from the original model. It helps keep training stable.
 
**Replay buffer:** Default is 100% fresh ratings (the sweep found 0% replay beat 30% replay); you can dial in old-sample mixing with `replay_ratio` if you hit drift.
 
**Adapter versioning:** Every training run saves a new adapter version. `rollback` switches back instantly.
 
**Actor loss:**
 
```
L = actor_loss * trajectory_scale + β * mean(KL)
```
 
Positive advantage makes good responses more likely. Negative advantage makes bad responses less likely. Clipping, drift scaling, and KL keep each step small.
 
## Training presets
 
Presets are based on the configs that worked best in our sweeps around 31B models, with separate paths for dense and MoE behavior. RL LoRA learning rates stay close to flat across sizes, so ReinforceClaw does not copy pre-training scaling rules. Pick `custom` in the wizard if you want to change anything.

Dense and MoE models use different config paths:

| Model type | What ReinforceClaw keys on | Default behavior |
|---|---|---|
| Dense | total params = active params | simple size-based preset |
| MoE | total params, active params, and sparsity | lower LR, lower KL, higher LoRA rank |
 
| Preset | Profile | Risk |
|---|---|---|
| careful | Safest, smallest updates | Low |
| **balanced** | Sweep winner - default | Moderate |
| aggressive | Fastest; higher overfit risk | Higher |
 
## Supported models
 
Any local, open-weight trainable transformer model should work as a target: HuggingFace repo, local model directory, MLX model, or CUDA/Transformers/PEFT-compatible model. Dense and MoE models are both supported.

Ollama and GGUF/llama.cpp work for actually using the improved model. The only catch is that RL training needs the matching writable model weights first, so ReinforceClaw trains a LoRA adapter on the matching local/HuggingFace model, then attaches that adapter to your Ollama/GGUF model for inference. In normal use, you still talk to the Ollama model. The base must match because adapters learn changes for specific internal layers; putting the adapter on a different base can fail or behave badly.

Cloud/API models like OpenAI, Anthropic, Gemini, xAI, or others cannot be updated directly because you do not own those remote weights. You can still rate those responses and use the ratings to make a local trainable model you own better with ReinforceClaw.

For Ollama, train against the matching local/HuggingFace weights and serve through the matching Ollama base model. ReinforceClaw creates a separate `<base>-reinforceclaw` model with the adapter attached, so your original model stays untouched.
 
## Supported agents
 
- **Claude Code** - hooks install automatically for terminal and Claude Code Desktop
- **Codex** - hooks install automatically for CLI and the Codex desktop app
- **OpenClaw** - installs as a bundled plugin and works across connected channels like WhatsApp, Telegram, Slack, and Discord

Claude Code terminal and Claude Code Desktop share `~/.claude/settings.json`, so one install covers both. Codex CLI and the Codex desktop app use `~/.codex/hooks.json` plus `~/.codex/config.toml`; `reinforceclaw init` writes both and enables `codex_hooks`.

For OpenClaw, run `reinforceclaw init` and select OpenClaw. ReinforceClaw installs the bundled safe plugin into your existing Gateway, writes only the local host/secret/feedback-mode config, starts its own local bridge service, and exposes `/rl good`, `/rl bad`, `/rl undo`, and `/rl status` in connected channels. Reaction ratings are owner-scoped in groups, and admin commands stay disabled unless you explicitly allowlist them. The plugin never starts shell commands or reads your environment; the local bridge is managed by ReinforceClaw's own launchd/systemd service. You can verify it with:

```bash
openclaw plugins list
openclaw plugins inspect reinforceclaw-feedback
```
 
## Requirements
 
- Python 3.10+
- Apple Silicon with MLX or Linux with CUDA
- A local trainable model
- Recommended for live use: a local inference server such as Ollama, LM Studio, or vLLM. After training, point your agent at the prepared local model/adapter output.

`trust_remote_code` is off by default. Only enable it for a model repo you trust, because it runs that repo's Python code during loading.

Feel free to fork it and build weird things. I would love to see what you make.
 
## License
 
MIT
 
---
 
