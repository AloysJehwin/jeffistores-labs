#!/usr/bin/env python
"""Stage 3 follow-up: evaluate a fine-tuned adapter through the same eval
harness as the baselines, append a row to baseline_results.json.

Usage:
    uv run python scripts/05_eval_finetune.py --adapter data/jeffi_descgen/v1/checkpoints/phi3-qlora-v1/adapter-final
    uv run python scripts/05_eval_finetune.py --adapter <path> --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from jeffistores_labs.descgen.baselines import FineTunedGenerator
from jeffistores_labs.descgen.dataset import DATA_DIR, read_jsonl
from jeffistores_labs.descgen.eval import evaluate, render_results_table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True, help="Path to saved LoRA adapter dir")
    parser.add_argument("--name", type=str, default=None, help="Override generator name in results")
    parser.add_argument("--limit", type=int, default=None, help="Eval first N test examples only")
    parser.add_argument(
        "--results-json",
        type=Path,
        default=DATA_DIR / "baseline_results.json",
        help="JSON file to append the new row into",
    )
    args = parser.parse_args()

    if not args.adapter.exists():
        print(f"ERROR: adapter not found: {args.adapter}", file=sys.stderr)
        return 1

    test_path = DATA_DIR / "test.jsonl"
    test = read_jsonl(test_path)
    print(f"Loaded {len(test)} test examples from {test_path}")

    gen = FineTunedGenerator(adapter_path=str(args.adapter), name=args.name)
    print(f"Eval'ing generator: {gen.name}")

    t0 = time.time()
    res = evaluate(gen, test, limit=args.limit)
    print(f"\n  {res.summary()}  ({time.time() - t0:.1f}s)")
    print()

    # Append to baseline_results.json so the notebook table picks it up
    if args.results_json.exists():
        existing = json.loads(args.results_json.read_text())
    else:
        existing = {"test_size": len(test), "limit": args.limit, "results": []}

    existing["results"] = [r for r in existing["results"] if r.get("generator") != gen.name]
    existing["results"].append({
        "generator": res.generator_name,
        "n": res.n,
        "bleu": res.bleu,
        "rouge_l_f1": res.rouge_l_f1,
        "cosine_similarity": res.cosine_similarity,
        "length_ratio": res.length_ratio,
        "per_example": res.per_example,
    })
    args.results_json.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    print(render_results_table([
        type(res)(generator_name=r["generator"], n=r["n"], bleu=r["bleu"],
                  rouge_l_f1=r["rouge_l_f1"], cosine_similarity=r["cosine_similarity"],
                  length_ratio=r["length_ratio"])
        for r in existing["results"]
    ]))
    print(f"\nAppended to {args.results_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
