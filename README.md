# jeffistores-labs

Experimental ML / LLM lab tied to **Jeffi Stores** (industrial-hardware e-commerce). Public learning journey toward a role at Anthropic by **2026-12**.

Goal: every project here either (a) reproduces fundamental ML work for learning, or (b) ships a real ML feature into Jeffi.

## Repo Layout

```
experiments/          One folder per experiment. Self-contained.
  01_micrograd/         Karpathy: build autograd from scratch
  02_makemore/          Karpathy: bigram → MLP → wavenet language models
  03_nanogpt/           Karpathy: GPT from scratch on Tiny Shakespeare
  04_jeffi_descgen/     Fine-tune small LLM on Jeffi product descriptions
notebooks/            Exploratory Jupyter notebooks
src/jeffistores_labs/ Reusable library code (data loaders, eval helpers)
data/                 Local datasets (gitignored — see data/README.md)
scripts/              One-off scripts (download data, sync results)
blog/                 Markdown drafts for blog posts
```

## Two-machine workflow

| Machine | Role |
|---|---|
| **Mac** | Code, git, notes, light Jupyter, blog drafts |
| **Ubuntu + RTX 4080** | All training, fine-tuning, GPU inference |

Source of truth: this GitHub repo. Code flows Mac → GitHub → Ubuntu. Results (loss curves, sample outputs, model metadata) flow Ubuntu → GitHub → Mac. **Model weights are gitignored** — keep them on Ubuntu and/or push to Hugging Face.

## Setup (Ubuntu, RTX 4080 — primary)

```bash
git clone git@github.com:AloysJehwin/jeffistores-labs.git
cd jeffistores-labs

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create env + install deps
uv sync

# Verify CUDA is visible to PyTorch
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Setup (Mac — code/git only)

```bash
cd ~/Documents/GitHub-Personal/jeffistores-labs
uv sync --extra cpu  # CPU-only PyTorch on Mac
```

## Connecting Mac → Ubuntu (VSCode Remote-SSH)

1. Install VSCode "Remote - SSH" extension
2. Add to `~/.ssh/config` on Mac:
   ```
   Host ubuntu-gpu
       HostName <ubuntu-ip-or-tailscale-name>
       User <your-user>
       IdentityFile ~/.ssh/id_ed25519
   ```
3. In VSCode: `Cmd+Shift+P` → "Remote-SSH: Connect to Host" → `ubuntu-gpu`
4. Open the cloned repo on Ubuntu — code locally on Mac, GPU runs remote.

## Daily Loop

1. **Morning (Mac)**: write code, push to GitHub
2. **Evening (Ubuntu via SSH)**: `git pull`, run training, commit results back
3. **Log progress** in [`../Jeffi_Storess_Site/ML_JOURNEY.md`](../Jeffi_Storess_Site/ML_JOURNEY.md) → Daily Log section

## Current Experiment

See `experiments/01_micrograd/` — Day 1.
