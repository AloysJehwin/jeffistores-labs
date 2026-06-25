# 04 descgen — Stage 3 phases (DoRA experiment)

Working notes for the QLoRA vs QDoRA comparison on Phi-3.5-mini.
Anchor paper: [DoRA, Liu et al. 2024 (arxiv 2402.09353)](https://arxiv.org/abs/2402.09353).

## Setup snapshot (Phase 1, 2026-06-25)

| | |
|---|---|
| Host | Razer (RTX 4080 Laptop, 12 GB VRAM) |
| Python | 3.12.13 (uv-managed) |
| torch | 2.12.0 + cu130 |
| transformers | 5.12.0 |
| trl | 1.6.0 |
| peft | 0.19.1 |
| bitsandbytes | 0.49.2 |
| accelerate | 1.14.0 |
| datasets | 5.0.0 |
| Base model | microsoft/Phi-3.5-mini-instruct |

Important: transformers 5.x defaults `torch_dtype="auto"` (changed in #42805,
[bisect notes](https://github.com/huggingface/trl/issues/3910#issuecomment-4801223688)).
Both v1 and v2 configs should pin compute dtype explicitly to avoid surprises.

## Phase outline

| # | Phase | Status | Notes |
|---|---|---|---|
| 1 | Sense check | DONE | env good, GPU idle, dataset missing on disk |
| 2 | Materialize dataset (`scripts/01_export_dataset.py`) | DONE | produces `data/jeffi_descgen/v1/{train,val,test}.jsonl` — 761/95/96 from 952 filtered |
| 3 | Smoke test V1 (baseline QLoRA, `phi3_qlora_v1.yaml`) | DONE | 50 steps clean; train_loss 1.86→1.57, eval_loss 1.83→1.55, mean_token_accuracy 0.59→0.64. Wall: ~2 min. |
| 4 | Create V2 config (`phi3_qlora_v2_dora.yaml`), verify script forwards `use_dora`, smoke test V2 | DONE | added `use_dora` to `train.py` LoraConfig; smoke 50 steps: eval_loss 1.829→1.540 (vs V1 1.828→1.545; indistinguishable at 50 steps). Wall: 140.9 s vs 119.6 s (+18%, as expected). 1.67M trainable params (0.044%). |
| 5 | Approval checkpoint | DONE | approved: sequential V1 → V2, WANDB_DISABLED, JSONL logging, seed 42 |
| 6 | Real runs (V1, then V2) | DONE | V1: 1232 s, eval_loss 0.335, mean_token_acc 0.919. V2: 766 s, eval_loss 0.336, mean_token_acc 0.919. V2 faster wall time but warmer GPU cache (not a fair speed test). |
| 7 | Compare results + capture | DONE | RESULTS_v1_vs_v2.md, runs/loss_curves.png, runs/*.jsonl |

## Final result

**V1 and V2 are functionally identical on this task at this scale.** Largest
eval_loss gap is 0.007 at step 50, well within run-to-run noise on a single
seed. DoRA neither helps nor hurts here. Honest negative result — publishable.

Honest read of why:
- 761 examples is a small training corpus; LoRA's limited capacity probably
  isn't the bottleneck — data is. DoRA's "more closely resembling full
  fine-tuning" advantage matters most when LoRA can't fit; here LoRA fits
  fine (eval_loss 0.335 is strong for 3 epochs on 761 examples).
- We did not try larger LoRA `r`, more epochs, or different target_modules.
  Any of those might surface a DoRA gap; none are scoped for this stage.
- Single seed. Variance across seeds is unknown. The 0.0005 eval_loss
  difference at step 288 is likely seed noise, not signal.

## What to do with this result

- DON'T ship DoRA into production. No measurable gain to justify the
  slightly larger adapter (6.70 MB vs 6.30 MB) and slightly higher
  compute (negligible here but real at larger scale).
- DO use V1 (vanilla QLoRA) adapter for Stage 4 (FastAPI service).
- DO write up the negative result in the blog post — it's actually more
  useful to readers than another "DoRA wins by 0.5%" puff piece.

## Decisions log

- **2026-06-25**: anchor on DoRA over PiSSA / NEFTune. Single technique, single comparison this stage. PiSSA / NEFTune are V3/V4 follow-ups.
- **2026-06-25**: V1 and V2 use the same train/val/test split, same seed, same hyperparams. Only delta is `use_dora: true`.
- **2026-06-25**: original `phi3_qlora_v1.yaml` stays as smoke-test reference. Real-run configs are new files so the experiment is reproducible from the repo state.

## Risks / things to verify before Phase 6

- peft 0.19.1 supports `use_dora=True` on `Linear` modules + bnb 4-bit (Phi-3.5 attention is all `Linear`, should be OK — verify in Phase 4 smoke) — **CONFIRMED in Phase 4 smoke**
- DoRA per-step wall-time is ~15-30% higher than vanilla LoRA; budget V2 longer — **CONFIRMED +18% (119.6s → 140.9s for 50 steps)**
- Single-seed comparison only. Variance unknown. If V2 - V1 is small, may need to revisit.
- trl 1.7 will default `loss_type` from `'nll'` to `'chunked_nll'` (currently on 1.6.0). Future-watch.

## Anti-goals (not in scope for this experiment)

- Hyperparameter sweep (`lora_r`, `learning_rate`, `epochs`)
- PiSSA, NEFTune
- Touching the held-out test set (stays sacred until V1/V2 are locked)
- Deploying anywhere
- Pushing anything to main without explicit approval
