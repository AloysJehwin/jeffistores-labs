# Stage 3 — V1 (vanilla QLoRA) vs V2 (QDoRA) on Jeffi descgen

Same data (761 train / 95 val), same seed (42), same hyperparams. Only delta: `lora.use_dora`.

## Run summary
| metric | V1 (vanilla QLoRA) | V2 (QDoRA) |
|---|---|---|
| train_runtime | 1232 | 766.3 |
| train_samples_per_second | 1.852 | 2.979 |
| train_steps_per_second | 0.234 | 0.376 |
| total_flos | 1.414e+16 | 1.414e+16 |
| train_loss | 0.6828 | 0.6799 |

## Eval loss by step
| step | V1 eval_loss | V2 eval_loss | V1 mean_token_acc | V2 mean_token_acc |
|---|---|---|---|---|
| 50 | 0.8688 | 0.8618 | 0.804 | 0.8051 |
| 100 | 0.5131 | 0.5131 | 0.8751 | 0.8755 |
| 150 | 0.3934 | 0.396 | 0.9048 | 0.9044 |
| 200 | 0.347 | 0.3488 | 0.9169 | 0.9146 |
| 250 | 0.3353 | 0.3365 | 0.9195 | 0.9185 |
| 288 | 0.335 | 0.3355 | 0.9193 | 0.9186 |

## Final eval
- V1 final eval_loss: **0.335**
- V2 final eval_loss: **0.3355**
- V1 final mean_token_accuracy: **0.9193**
- V2 final mean_token_accuracy: **0.9186**

Per-step logs: see `runs/phi3-qlora-v1.jsonl` and `runs/phi3-qdora-v2.jsonl`.
Loss curves: see `runs/loss_curves.png`.

