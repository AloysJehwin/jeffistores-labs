#!/usr/bin/env python
"""Compare two run JSONLs (e.g. V1 baseline vs V2 DoRA).

Reads experiments/04_jeffi_descgen/runs/<run_name>.jsonl files written by
the JSONL callback in src/jeffistores_labs/descgen/train.py.

Outputs:
  - experiments/04_jeffi_descgen/RESULTS_v1_vs_v2.md
  - experiments/04_jeffi_descgen/runs/loss_curves.png

Usage (from repo root, on the Razer):
    .venv/bin/python scripts/06_compare_runs.py \\
        --v1 experiments/04_jeffi_descgen/runs/phi3-qlora-v1.jsonl \\
        --v2 experiments/04_jeffi_descgen/runs/phi3-qdora-v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_metrics(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Separate train logs, eval logs, and the final summary."""
    train_rows = [r for r in rows if "loss" in r and "eval_loss" not in r]
    eval_rows = [r for r in rows if "eval_loss" in r]
    summary_rows = [r for r in rows if "train_runtime" in r]
    summary = summary_rows[-1] if summary_rows else {}
    return train_rows, eval_rows, summary


def fmt(x, ndigits=4) -> str:
    if x is None:
        return "-"
    if isinstance(x, (int, float)):
        return f"{x:.{ndigits}g}"
    return str(x)


def make_table(v1_train: list[dict], v2_train: list[dict], v1_eval: list[dict], v2_eval: list[dict]) -> str:
    """Markdown table of eval_loss at matched eval steps."""
    v1_eval_by_step = {r["step"]: r for r in v1_eval}
    v2_eval_by_step = {r["step"]: r for r in v2_eval}
    steps = sorted(set(v1_eval_by_step) | set(v2_eval_by_step))

    lines = [
        "| step | V1 eval_loss | V2 eval_loss | V1 mean_token_acc | V2 mean_token_acc |",
        "|---|---|---|---|---|",
    ]
    for s in steps:
        v1 = v1_eval_by_step.get(s, {})
        v2 = v2_eval_by_step.get(s, {})
        lines.append(
            f"| {s} | {fmt(v1.get('eval_loss'))} | {fmt(v2.get('eval_loss'))} "
            f"| {fmt(v1.get('eval_mean_token_accuracy'))} | {fmt(v2.get('eval_mean_token_accuracy'))} |"
        )
    return "\n".join(lines)


def make_summary_block(v1_sum: dict, v2_sum: dict) -> str:
    keys = ("train_runtime", "train_samples_per_second", "train_steps_per_second", "total_flos", "train_loss")
    lines = [
        "| metric | V1 (vanilla QLoRA) | V2 (QDoRA) |",
        "|---|---|---|",
    ]
    for k in keys:
        lines.append(f"| {k} | {fmt(v1_sum.get(k))} | {fmt(v2_sum.get(k))} |")
    return "\n".join(lines)


def plot_curves(
    v1_train: list[dict], v2_train: list[dict],
    v1_eval: list[dict], v2_eval: list[dict],
    out_path: Path,
) -> None:
    """Two-panel plot: train loss + eval loss for both runs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"WARN: matplotlib not installed; skipping {out_path}", file=sys.stderr)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    def steps_loss(rows, key):
        return [r["step"] for r in rows if key in r], [r[key] for r in rows if key in r]

    s, l = steps_loss(v1_train, "loss"); ax1.plot(s, l, label="V1 vanilla QLoRA", linewidth=1.6)
    s, l = steps_loss(v2_train, "loss"); ax1.plot(s, l, label="V2 QDoRA", linewidth=1.6)
    ax1.set_xlabel("step"); ax1.set_ylabel("train loss"); ax1.set_title("Train loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    s, l = steps_loss(v1_eval, "eval_loss"); ax2.plot(s, l, marker="o", label="V1 vanilla QLoRA", linewidth=1.6)
    s, l = steps_loss(v2_eval, "eval_loss"); ax2.plot(s, l, marker="o", label="V2 QDoRA", linewidth=1.6)
    ax2.set_xlabel("step"); ax2.set_ylabel("eval loss"); ax2.set_title("Eval loss"); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v1", type=Path, required=True)
    p.add_argument("--v2", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("experiments/04_jeffi_descgen/RESULTS_v1_vs_v2.md"))
    p.add_argument("--plot", type=Path, default=Path("experiments/04_jeffi_descgen/runs/loss_curves.png"))
    args = p.parse_args()

    for path in (args.v1, args.v2):
        if not path.exists():
            print(f"ERROR: not found: {path}", file=sys.stderr)
            return 1

    v1_rows = read_jsonl(args.v1)
    v2_rows = read_jsonl(args.v2)
    v1_train, v1_eval, v1_sum = split_metrics(v1_rows)
    v2_train, v2_eval, v2_sum = split_metrics(v2_rows)

    body_parts = [
        "# Stage 3 — V1 (vanilla QLoRA) vs V2 (QDoRA) on Jeffi descgen",
        "",
        "Same data (761 train / 95 val), same seed (42), same hyperparams. Only delta: `lora.use_dora`.",
        "",
        "## Run summary",
        make_summary_block(v1_sum, v2_sum),
        "",
        "## Eval loss by step",
        make_table(v1_train, v2_train, v1_eval, v2_eval),
        "",
        "## Final eval",
        f"- V1 final eval_loss: **{fmt(v1_eval[-1].get('eval_loss') if v1_eval else None)}**",
        f"- V2 final eval_loss: **{fmt(v2_eval[-1].get('eval_loss') if v2_eval else None)}**",
        f"- V1 final mean_token_accuracy: **{fmt(v1_eval[-1].get('eval_mean_token_accuracy') if v1_eval else None)}**",
        f"- V2 final mean_token_accuracy: **{fmt(v2_eval[-1].get('eval_mean_token_accuracy') if v2_eval else None)}**",
        "",
        "Per-step logs: see `runs/phi3-qlora-v1.jsonl` and `runs/phi3-qdora-v2.jsonl`.",
        "Loss curves: see `runs/loss_curves.png`.",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(body_parts) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    plot_curves(v1_train, v2_train, v1_eval, v2_eval, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
