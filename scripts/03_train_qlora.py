#!/usr/bin/env python
"""Stage 3: QLoRA fine-tune of Phi-3.5-mini on the descgen dataset.

Smoke-test mode:
    uv run python scripts/03_train_qlora.py --smoke

Real run (after Karpathy nanoGPT, when you understand what loss curves mean):
    uv run python scripts/03_train_qlora.py

Override the config (e.g. for a sweep):
    uv run python scripts/03_train_qlora.py --config configs/phi3_qlora_v2.yaml

The smoke test runs 50 optimizer steps and saves a checkpoint. It exists
so you can verify the whole pipeline (load model -> attach LoRA ->
tokenize -> backprop -> save adapter -> evaluate) in ~5 minutes before
committing to a real 1-3 hour training run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jeffistores_labs.descgen.dataset import DATA_DIR, read_jsonl
from jeffistores_labs.descgen.train import TrainingConfig, train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phi3_qlora_v1.yaml"),
        help="YAML training config",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-test mode: applies config.smoke overrides (max_steps=50).",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1

    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"
    if not (train_path.exists() and val_path.exists()):
        print(
            f"ERROR: dataset missing. Run scripts/01_export_dataset.py first.\n"
            f"       expected: {train_path} and {val_path}",
            file=sys.stderr,
        )
        return 1

    cfg = TrainingConfig.from_yaml(args.config)
    print(f"Run name : {cfg.run_name}")
    print(f"Base     : {cfg.base_model}")
    print(f"Smoke    : {args.smoke}")
    print(f"Output   : {cfg.training['output_dir']}")
    print()

    train_examples = read_jsonl(train_path)
    val_examples = read_jsonl(val_path)
    print(f"Loaded {len(train_examples)} train + {len(val_examples)} val examples")
    print()

    adapter_dir = train(cfg, train_examples, val_examples, smoke_test=args.smoke)
    print()
    print("Training complete.")
    print(f"Adapter saved to: {adapter_dir}")
    print()
    print("Next step: evaluate it through the same harness as the baselines:")
    print("  uv run python scripts/05_eval_finetune.py --adapter", adapter_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
