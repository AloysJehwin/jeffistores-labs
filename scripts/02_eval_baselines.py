#!/usr/bin/env python
"""Stage 2: Run all baselines through the eval harness.

Outputs a comparison table to stdout and writes a JSON results file to
data/jeffi_descgen/v1/baseline_results.json.

Run on the Razer (where the GPU + jeffi_replica live):

    uv run python scripts/02_eval_baselines.py
    uv run python scripts/02_eval_baselines.py --limit 20            # smaller eval
    uv run python scripts/02_eval_baselines.py --skip phi3           # skip slow ones
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from jeffistores_labs.descgen.baselines import (
    CopyInputBlockGenerator,
    CopyInputGenerator,
    DBAIDescriptionBaseline,
    EmptyGenerator,
    Phi3ZeroShotGenerator,
)
from jeffistores_labs.descgen.dataset import DATA_DIR, Example, read_jsonl
from jeffistores_labs.descgen.eval import (
    EvalResult,
    evaluate,
    render_results_table,
)


def _run_db_ai_baseline(examples: list[Example], limit: int | None) -> EvalResult:
    """Special-case wrapper since DBAIDescriptionBaseline needs per-example lookup."""
    base = DBAIDescriptionBaseline().with_examples(examples)

    class _Adapter:
        name = base.name

        def __init__(self, examples_list: list[Example]):
            self._iter = iter(examples_list)

        def generate(self, _product):  # noqa: ARG002
            ex = next(self._iter)
            return base.generate_for(ex)

    return evaluate(_Adapter(examples), examples, limit=limit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only evaluate the first N test examples (faster runs).",
    )
    parser.add_argument(
        "--skip", action="append", default=[],
        choices=["empty", "copy_input", "copy_spec_block", "db_ai", "phi3"],
        help="Skip specific baselines (repeatable).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "baseline_results.json",
        help="Where to write the results JSON.",
    )
    args = parser.parse_args()

    test_path = DATA_DIR / "test.jsonl"
    if not test_path.exists():
        print(
            f"ERROR: {test_path} not found. Run scripts/01_export_dataset.py first.",
            file=sys.stderr,
        )
        return 1

    test = read_jsonl(test_path)
    print(f"Loaded {len(test)} test examples from {test_path}")

    results: list[EvalResult] = []

    if "empty" not in args.skip:
        print("\n→ empty (sanity floor)")
        t0 = time.time()
        results.append(evaluate(EmptyGenerator(), test, limit=args.limit))
        print(f"   {results[-1].summary()}  ({time.time() - t0:.1f}s)")

    if "copy_input" not in args.skip:
        print("\n→ copy_input")
        t0 = time.time()
        results.append(evaluate(CopyInputGenerator(), test, limit=args.limit))
        print(f"   {results[-1].summary()}  ({time.time() - t0:.1f}s)")

    if "copy_spec_block" not in args.skip:
        print("\n→ copy_spec_block")
        t0 = time.time()
        results.append(evaluate(CopyInputBlockGenerator(), test, limit=args.limit))
        print(f"   {results[-1].summary()}  ({time.time() - t0:.1f}s)")

    if "db_ai" not in args.skip:
        print("\n→ db_ai_description (existing GPT baseline)")
        t0 = time.time()
        results.append(_run_db_ai_baseline(test, args.limit))
        print(f"   {results[-1].summary()}  ({time.time() - t0:.1f}s)")

    if "phi3" not in args.skip:
        print("\n→ phi3_zero_shot (loads ~2GB onto GPU on first call)")
        t0 = time.time()
        results.append(evaluate(Phi3ZeroShotGenerator(), test, limit=args.limit))
        print(f"   {results[-1].summary()}  ({time.time() - t0:.1f}s)")

    print("\n" + "=" * 80)
    print(render_results_table(results))
    print("=" * 80)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "test_size": len(test),
        "limit": args.limit,
        "results": [
            {
                "generator": r.generator_name,
                "n": r.n,
                "bleu": r.bleu,
                "rouge_l_f1": r.rouge_l_f1,
                "cosine_similarity": r.cosine_similarity,
                "length_ratio": r.length_ratio,
                "per_example": r.per_example,
            }
            for r in results
        ],
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nResults written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
