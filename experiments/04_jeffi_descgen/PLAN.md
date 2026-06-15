# 04 — Jeffi Description Generator

**One sentence**: Fine-tune a small open LLM (≤ 7B params, QLoRA) on Jeffi's 1,204 product descriptions and ship a demo service that auto-drafts new product copy from spec sheets.

**Anthropic-relevant skills demonstrated**: dataset construction, eval-driven ML (most important), QLoRA / PEFT, retrieval-grounded generation, FastAPI inference service, public writeup.

**Scope guardrails**
- Stays on the Razer (RTX 4080 Laptop, 12 GB VRAM) — no cloud GPU.
- Demo only. Does **not** touch Jeffi prod. No `/admin` integration in this experiment.
- Produces a **portfolio piece + blog post**, not a feature.

---

## Why this isn't a Month 1 project

You're learning ML in depth. Jumping straight to QLoRA fine-tuning means:
- You won't understand *why* the loss curve does what it does
- You won't recognize when the model is broken vs. when the data is broken
- You'll cargo-cult HuggingFace boilerplate

So this project is the **Month 2 capstone**. Month 1 (Karpathy: micrograd → makemore → nanoGPT) builds the muscle to do this properly. Each Karpathy lesson maps to something here:

| Karpathy concept | Used here |
|---|---|
| Backprop intuition (micrograd) | Reading loss curves; debugging gradient explosions |
| Tokenization | Why Phi-3's tokenizer matters for Jeffi's "M8x50mm" SKU strings |
| Self-attention | Understanding what the LoRA adapters are *actually* changing |
| Sampling (temp / top-k / top-p) | Generating diverse-but-coherent descriptions |
| Loss / cross-entropy from scratch | Reading the W&B `train/loss` and `eval/loss` charts |

Plan it now, build it after Karpathy.

---

## Constraints (be honest)

| Constraint | Reality |
|---|---|
| **Dataset size** | 1,204 examples. Tiny. Expect overfitting after 2–3 epochs even with low rank. |
| **VRAM** | 12 GB. 7B-class with 4-bit QLoRA fits with batch_size=1, grad_accum=8. |
| **Eval ground truth** | Each product has *one* description. BLEU/ROUGE will be brutal. Semantic similarity + Claude-as-judge will be more useful. |
| **Domain** | Industrial hardware, Indian English, GST-aware. Generic LLMs hallucinate dimensions and certifications. RAG against the actual product spec rows is the cure. |

---

## Architecture

```
                                    ┌────────────────────┐
                                    │ jeffi_replica DB   │ ← nightly RDS sync
                                    │ (Postgres on Razer)│
                                    └──────────┬─────────┘
                                               │ scripts/01_export_dataset.py
                                               ▼
                          ┌──────────────────────────────────┐
                          │ data/jeffi_descgen_v1.jsonl      │
                          │   {input_spec, output_desc}      │
                          │ + train.jsonl / val.jsonl /      │
                          │   test.jsonl (held-out 100)      │
                          └────┬───────────────────┬─────────┘
                               │                   │
                  ┌────────────▼───┐   ┌───────────▼──────────────┐
                  │ Eval Harness   │   │ QLoRA Trainer            │
                  │ (run BEFORE    │   │ Phi-3-mini-128k (3.8B)   │
                  │  any training) │   │ rank=8, alpha=16,        │
                  │                │   │ lr=2e-4, 3 epochs        │
                  │ Baselines:     │   │ W&B logging              │
                  │ - empty        │   └───────────┬──────────────┘
                  │ - copy spec    │               │
                  │ - 0-shot Phi-3 │               ▼
                  │ - claude-haiku │   ┌──────────────────────────┐
                  │                │   │ adapters/jeffi-descgen-  │
                  │ Metrics:       │   │   v1/   (HF Hub mirror)  │
                  │ - BLEU-4       │   └───────────┬──────────────┘
                  │ - ROUGE-L      │               │
                  │ - cos-sim      │               ▼
                  │ - LLM-judge    │   ┌──────────────────────────┐
                  └────────┬───────┘   │ FastAPI service          │
                           │           │ POST /generate           │
                           └──────────►│ {sku, name, brand, attrs}│
                                       │ → {description}          │
                                       │ runs on Razer:8000       │
                                       └───────────┬──────────────┘
                                                   │ Tailscale only
                                                   ▼
                                       ┌──────────────────────────┐
                                       │ Streamlit demo page      │
                                       │ side-by-side with current│
                                       │ description for each SKU │
                                       └──────────────────────────┘
```

---

## Stages (each ends with a commit + a blog draft)

### Stage 0 — Plan (this doc) ✅

### Stage 1 — Dataset (3 days, post-Karpathy week 4)

**Files to create**:
- `scripts/01_export_dataset.py` — pulls from `jeffi_replica`, builds JSONL
- `notebooks/01_dataset_stats.ipynb` — token-length distribution, brand/category coverage, deduping
- `data/jeffi_descgen_v1/{train,val,test}.jsonl` — 80/10/10 split (gitignored)

**Schema**:
```json
{
  "id": 12345,
  "input": {
    "name": "M8x50mm Hex Bolt Zinc Plated",
    "sku": "BLT-M8-50-ZN",
    "brand": "Unbrako",
    "category": "Fasteners > Bolts > Hex",
    "attributes": {"thread_size": "M8", "length_mm": 50, "finish": "zinc", "grade": "8.8"},
    "mrp": 24.50
  },
  "output": "M8 x 50mm hex head bolt with zinc-plated finish for corrosion resistance..."
}
```

**Decision points**:
- Drop products with descriptions < 80 chars (boilerplate) or > 1500 chars (manuals)
- Strip HTML/markdown from descriptions before training
- Hold out 100 *random* products as `test.jsonl` — never look at these until final eval

### Stage 2 — Eval harness (5 days) ⚠️ HARDEST. DO BEFORE TRAINING.

**Files**:
- `src/jeffistores_labs/descgen/eval.py` — pure Python, no notebooks. Importable.
- `scripts/02_eval_baselines.py` — runs 4 baselines through the harness
- `notebooks/02_eval_results.ipynb` — visualize the baseline numbers

**Metrics** (4 in total — overlap matters):
1. **BLEU-4** — n-gram overlap with reference. Will be low. Useful relatively.
2. **ROUGE-L** — longest-common-subsequence. Better for paraphrase-tolerant comparison.
3. **Semantic similarity** — `all-MiniLM-L6-v2` cosine between generated + reference.
4. **Claude-as-judge** — Claude API rates each (gen, ref) pair on (a) factual consistency, (b) catalog tone, (c) completeness. Spend ~$5–10 here total.

**Baselines to beat** (if our fine-tune doesn't beat at least 2 of these, the project failed):
| Baseline | Description |
|---|---|
| `empty` | Returns "" — sanity floor |
| `copy_input` | Returns the product name — checks if metrics are sane |
| `phi3_zero_shot` | Phi-3-mini with a hand-crafted prompt, no fine-tuning |
| `claude_haiku` | claude-haiku-4-5 with the same prompt |

The fine-tune's job is to **beat zero-shot Phi-3 on BLEU/ROUGE/cos-sim** at a fraction of the inference cost of Claude. That's the headline result.

### Stage 3 — QLoRA fine-tune (3–5 days)

**Files**:
- `scripts/03_train_qlora.py` — single-file training script (HF `trl` `SFTTrainer` + `peft`)
- `configs/phi3_qlora_v1.yaml` — hyperparams checked into git
- W&B project: `jeffi-descgen`

**Hyperparams to start** (will be wrong; iterate):
```yaml
base_model: microsoft/Phi-3.5-mini-instruct
quantization: nf4 (4-bit)
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj]
learning_rate: 2e-4
batch_size: 1
gradient_accumulation_steps: 8
epochs: 3
warmup_ratio: 0.03
max_seq_length: 1024
```

**Sweeps** (W&B):
- `lora_r`: [4, 8, 16, 32]
- `learning_rate`: [1e-4, 2e-4, 5e-4]
- `epochs`: [2, 3, 5]

**Run after every sweep**: full eval harness on the test set. Log results to W&B.

### Stage 4 — Demo service + writeup (1 week)

**Files**:
- `src/jeffistores_labs/descgen/serve.py` — FastAPI service
- `scripts/04_serve.sh` — uvicorn launcher with adapter merged in
- `streamlit_demo.py` — side-by-side comparison UI
- `blog/jeffi-descgen-month-2.md` — full writeup with charts

**Deliverable for the public**:
- GitHub README on `jeffistores-labs` linking to the demo (Tailscale-only, OK)
- Blog post on Hashnode/Substack
- X thread: "I fine-tuned Phi-3 on 1,204 industrial-hardware product descriptions. Here's the loss curve, the eval table, and what surprised me."
- Hugging Face Hub: push the adapter (private OK) so the project links to a real artifact

---

## Anti-goals (things I'm explicitly NOT doing in this experiment)

- ❌ Training from scratch — wrong scale, wrong learning vehicle. nanoGPT in `experiments/03_nanogpt/` covers that.
- ❌ Deploying to Jeffi prod — separate project, requires PR review, A/B test infra, rollback plan.
- ❌ Using more than $20 in cloud / API costs — Claude-judge is the only paid bit, capped.
- ❌ Multiple models / a model zoo — one base model done well > five done lazily.
- ❌ Image / multimodal — that's a Month 4–5 project (CLIP on `product_images`).

---

## Success criteria

When this project is "done":

1. `python scripts/02_eval_baselines.py` and `python scripts/05_eval_finetune.py` produce a **single results table** comparing 5 systems × 4 metrics
2. The fine-tune **beats Phi-3 zero-shot** on at least 3 of 4 metrics
3. A **blog post is published** with the loss curves, eval table, and 3 honest "what didn't work"
4. The **adapter is on Hugging Face Hub** (or local with hash), with model card
5. The **demo service runs on Razer** at `http://razer-gpu:8000/docs` and a 5-min recording exists

---

## Timeline placement in your 6-month roadmap

```
Month 1 (Jun 14 – Jul 13)   ← Karpathy foundations. Project: nanoGPT on Tiny Shakespeare.
Month 2 (Jul 14 – Aug 13)   ← THIS PROJECT. Stages 1 → 4.
Month 3 (Aug 14 – Sep 13)   ← Semantic search + RAG (different experiment folder)
Month 4–6                    ← Open-source contributions, Anthropic interview prep
```

If you finish this project ahead of schedule, move on to the semantic-search project — don't gold-plate this one.

---

## Open questions for me to answer when starting Stage 1

- [ ] Are 1,204 descriptions enough for QLoRA on Phi-3-mini? (Run a 3-epoch test, see eval curve plateau.)
- [ ] Should we include the brand voice as part of the input prompt? (Test both, eval will tell.)
- [ ] Filter or include HTML in source descriptions? (Look at `notebooks/01_dataset_stats.ipynb` first.)
- [ ] Is `Phi-3.5-mini-instruct` (3.8B) right, or should we try `Qwen2.5-3B` or `Llama-3.2-3B`? (Decision in Stage 2 — pick whichever zero-shot baseline is *closest* to our gold; smaller delta = easier fine-tune.)

---

## How to know when to ask for help

- Loss is `nan` after 100 steps → ping me, gradient blow-up
- Eval gets *worse* after fine-tune → ping me, likely data leak or wrong loss masking
- Inference < 5 tok/s → ping me, quantization config is off
- Started 3 weeks ago, still on Stage 1 → ping me, scope-creep check
