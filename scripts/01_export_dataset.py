#!/usr/bin/env python
"""Stage 1: Export jeffi_replica products → JSONL train/val/test files.

Run from the repo root on the Razer (where jeffi_replica lives):

    uv run python scripts/01_export_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from jeffistores_labs.descgen.dataset import DATA_DIR, export_all


def main() -> int:
    counts = export_all()
    print()
    print("Export complete →", DATA_DIR)
    print()
    width = max(len(k) for k in counts) + 2
    for k, v in counts.items():
        print(f"  {k:<{width}} {v:>6,}")
    print()
    print("Files:")
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.relative_to(Path.cwd())}  ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
