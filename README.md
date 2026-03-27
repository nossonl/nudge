# ReinforceClaw
 
Self-improving reinforcement learning for your AI agents. Set it up once, it runs autonomously in the background, and your model keeps getting better every time you use it.

![ReinforceClaw Diagram](assets/diagram.png)

## The problem
 
These AI models aren't built for you. They're generalized for everybody, and everybody has different preferences. They're not optimized for how *you* work. So you spend all this time tweaking prompts/skills, editing outputs, arguing with the model, rewriting instructions, and it still doesn't listen. What if your code could actually look like you hand typed it?
 
With ReinforceClaw, the model learns how you think and is rewarded or held accountable. With reinforcement learning, over time it actually starts thinking the way you do and behaving the way you want it to.
 
## How it works
 
After each response from your agent, you rate it good or bad. One tap.
 
```
╭─ Rate this response ───────────╮
│  [1] Good          [2] Bad     │
╰────────────────────────────────╯
```
 
Ratings go into a local database. Once enough build up (16 by default), a training run kicks off automatically in the background. It trains a LoRA adapter on your local model, hot-swaps it into your running inference server, and your next rollout is already a little closer to what you wanted.
 
You don't touch any of this after setup. It just runs. Every round, the policy improves.
 
> **⚠️ Rate good responses too.** Bad tends to stand out more, so often you'll only rate bad. But only rating bad won't help much. The model needs to understand what you actually like, not just what to avoid. Rate both.
 
## Commands
 
Inside your agent (Claude Code, Codex, OpenClaw), use `/rl`:
 
```
/rl good         # positive reward
/rl bad          # negative reward
/rl undo         # remove last rating
/rl status       # check stats
/rl train        # force a training run
/rl rollback     # revert to previous adapter
/rl reset        # start fresh
```
 
From the terminal, same thing but with `reinforceclaw`:
 
```bash
reinforceclaw good
reinforceclaw status
reinforceclaw train
```
 
## Under the hood
 
Uses **MIS-PO** (Metropolis Independence Sampling - Filtered Policy Optimization), same family of algorithms behind Step 3.5 Flash. Each response you get from the model is a trajectory rollout. Your good/bad rating is the reward. Training runs on single trajectories with binary filtering instead of grouped rollouts, which is what makes this work in real time with one response at a time.
 
**Reward:** +1 (good) or -1 (bad).
 
**EMA baseline:** An exponential moving average tracks your rating history. Advantage is computed as:
 
```
advantage = rating - ema_mean
ema_mean = decay * ema_mean + (1 - decay) * rating
```
 
Default decay is `0.99`. If you've been rating mostly good lately, a bad rating carries more weight.
 
**LoRA adapters:** Base model weights stay frozen. Training only touches low-rank adapter layers (~0.1-1% of parameters). Fast, lightweight, fully reversible.
 
**Token-level clipping:** For each token, the probability ratio between the new policy and reference policy gets clipped:
 
```
ratio_t = π_new(token_t) / π_ref(token_t)
mask_t = (ratio_t >= 0.5) AND (ratio_t <= 2.0)
clipped_t = clip(ratio_t, 0.5, 2.0)
```
 
Tokens that moved too far get excluded from the gradient entirely.
 
**Trajectory-level clipping:** The geometric mean of all token ratios across the full response gets gated:
 
```
ρ̄ = exp(mean(log π_new - log π_ref))
gate = (ρ̄ >= 0.992) AND (ρ̄ <= 1.002)
```
 
If the whole rollout drifted too much, the entire gradient gets zeroed out.
 
**KL penalty:** Forward-KL divergence keeps the policy from straying too far from the reference model (`β = 0.001`):
 
```
KL = exp(log_ref - log_new) - (log_ref - log_new) - 1
```
 
Prevents mode collapse.
 
**Replay buffer:** Each training batch is 50/50 fresh ratings and randomly sampled older ones. Prevents catastrophic forgetting.
 
**Adapter versioning:** Every training run saves a new adapter version. `rollback` reverts instantly.
 
**Actor loss:**
 
```
L = -Σ(clipped_t * advantage * gate * mask_t) / Σ(mask_t) + β * KL
```
 
Positive advantage boosts probability of good responses, negative advantage suppresses bad ones. Clipping and gating keep it stable.
 
## Training presets
 
| Preset | Learning rate | Ratings to notice | Risk |
|---|---|---|---|
| careful | 2e-6 | ~50 | Low, tiny updates |
| **balanced** | 4e-6 | ~25 | Good default |
| aggressive | 1e-5 | ~16 | Fast but volatile |
 
Or custom.
 
## Supported models
 
Any HuggingFace model. The setup has a curated list from Qwen, Meta, Mistral, Google, Microsoft, DeepSeek, and more. Or paste any model ID.
 
## Supported agents
 
- **Claude Code** - hooks install automatically
- **Codex** - hooks install automatically
- **OpenClaw** - installs as a plugin, works across multiple platforms (WhatsApp, Telegram, Slack, Discord, etc.)
- More coming soon
 
## Requirements
 
- Python 3.10+
- Apple Silicon (Metal) or Linux (CPU / CUDA)
- A local inference server: Ollama, LM Studio, vLLM, llama.cpp, etc.
 
## Install
 
```bash
pip install reinforceclaw
reinforceclaw init
```
 
or
 
```bash
curl -sSL https://raw.githubusercontent.com/nossonl/reinforceclaw/main/install.sh | bash
```
 
Pick your agent, model, and training preset. After that, just use your agent. The rest is automatic.
 
## License
 
MIT
 
---
 
reinforceclaw reinforceclaw reinforceclaw
