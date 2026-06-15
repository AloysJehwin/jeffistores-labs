"""Dataset module for the Jeffi description generator.

Defines the schema, the prompt format, and the export pipeline that turns
products in `jeffi_replica` into JSONL training data.

Pipeline:
    products  --(filter + clean)-->  Example  --(format)-->  ChatTurns
    └─→ data/jeffi_descgen_v1/{train,val,test}.jsonl
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..db import query

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MIN_DESC_CHARS = 80     # below this, descriptions are placeholders / boilerplate
MAX_DESC_CHARS = 1500   # above this, descriptions are manuals / pasted PDFs
RANDOM_SEED = 42        # for reproducible splits

# Where the JSONL artifacts land (gitignored — see data/README.md)
DATA_VERSION = "v1"
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "jeffi_descgen" / DATA_VERSION


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------


@dataclass
class ProductInput:
    """Structured input fed to the model. Stable schema — bump DATA_VERSION on change."""

    name: str
    sku: str | None
    brand: str | None
    category: str | None
    material: str | None
    finish: str | None
    size: str | None
    length_cm: float | None
    breadth_cm: float | None
    height_cm: float | None
    weight_grams: float | None
    mrp: float | None
    short_description: str | None  # often a 1-line teaser; fine to feed in


@dataclass
class Example:
    """One training example: a product spec → its catalog description."""

    id: str
    product_input: ProductInput
    target_description: str
    # Reference baselines that are already in the DB — useful for eval, never as labels.
    db_ai_description: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input": asdict(self.product_input),
            "output": self.target_description,
            "db_ai_description": self.db_ai_description,
            "extras": self.extras,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Example":
        return cls(
            id=d["id"],
            product_input=ProductInput(**d["input"]),
            target_description=d["output"],
            db_ai_description=d.get("db_ai_description"),
            extras=d.get("extras", {}),
        )


# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------

_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _clean_text(s: str | None) -> str | None:
    """Strip HTML tags and collapse whitespace. Returns None for empty inputs."""
    if s is None:
        return None
    s = _HTML_TAG.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s if s else None


def _clean_input(p: ProductInput) -> ProductInput:
    return ProductInput(
        name=_clean_text(p.name) or "",
        sku=_clean_text(p.sku),
        brand=_clean_text(p.brand),
        category=_clean_text(p.category),
        material=_clean_text(p.material),
        finish=_clean_text(p.finish),
        size=_clean_text(p.size),
        length_cm=p.length_cm,
        breadth_cm=p.breadth_cm,
        height_cm=p.height_cm,
        weight_grams=p.weight_grams,
        mrp=p.mrp,
        short_description=_clean_text(p.short_description),
    )


# -----------------------------------------------------------------------------
# Prompt formatting — chat template-friendly so it works with both Phi-3 and Claude
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a copywriter for Jeffi Stores, an Indian industrial-hardware e-commerce "
    "site. Given a product's structured spec, write a clear, factual catalog description "
    "in 2–4 sentences. Keep technical details exact. Use Indian English. No marketing "
    "fluff, no emojis, no bullet points."
)


def render_input_block(p: ProductInput) -> str:
    """Render the structured input as a deterministic spec block.

    Stable ordering matters: the model learns to attend to specific positions.
    Missing fields are skipped entirely (rather than 'None'/'null') so the
    model doesn't learn to parrot 'None'.
    """
    lines: list[str] = [f"Name: {p.name}"]
    if p.brand:
        lines.append(f"Brand: {p.brand}")
    if p.category:
        lines.append(f"Category: {p.category}")
    if p.sku:
        lines.append(f"SKU: {p.sku}")
    if p.material:
        lines.append(f"Material: {p.material}")
    if p.finish:
        lines.append(f"Finish: {p.finish}")
    if p.size:
        lines.append(f"Size: {p.size}")
    dims = []
    if p.length_cm:
        dims.append(f"L={p.length_cm}cm")
    if p.breadth_cm:
        dims.append(f"B={p.breadth_cm}cm")
    if p.height_cm:
        dims.append(f"H={p.height_cm}cm")
    if dims:
        lines.append(f"Dimensions: {' x '.join(dims)}")
    if p.weight_grams:
        lines.append(f"Weight: {p.weight_grams:.0f} g")
    if p.mrp:
        lines.append(f"MRP: ₹{p.mrp:.2f}")
    if p.short_description:
        lines.append(f"Tagline: {p.short_description}")
    return "\n".join(lines)


def to_chat_messages(p: ProductInput, target: str | None = None) -> list[dict[str, str]]:
    """Format a product as a list of chat turns. If target is None, only the prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_input_block(p)},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


# -----------------------------------------------------------------------------
# Export pipeline
# -----------------------------------------------------------------------------

EXPORT_SQL = """
SELECT
    p.id::text AS id,
    p.name,
    p.sku,
    b.name AS brand,
    c.name AS category,
    p.material,
    p.finish,
    p.size,
    p.length_cm,
    p.breadth_cm,
    p.height_cm,
    p.weight_grams,
    p.mrp,
    p.short_description,
    p.description,
    p.ai_description
FROM products p
LEFT JOIN brands     b ON p.brand_id    = b.id
LEFT JOIN categories c ON p.category_id = c.id
WHERE p.description IS NOT NULL
  AND length(p.description) >= :min_chars
  AND length(p.description) <= :max_chars
"""


def fetch_raw() -> pd.DataFrame:
    """Pull everything we'll need from `jeffi_replica`."""
    return query(EXPORT_SQL, {"min_chars": MIN_DESC_CHARS, "max_chars": MAX_DESC_CHARS})


def row_to_example(row: pd.Series) -> Example:
    cleaned_desc = _clean_text(row["description"])
    if cleaned_desc is None:
        raise ValueError(f"row {row['id']} has empty description after cleaning")

    p = ProductInput(
        name=row["name"],
        sku=row.get("sku"),
        brand=row.get("brand"),
        category=row.get("category"),
        material=row.get("material"),
        finish=row.get("finish"),
        size=row.get("size"),
        length_cm=row.get("length_cm"),
        breadth_cm=row.get("breadth_cm"),
        height_cm=row.get("height_cm"),
        weight_grams=row.get("weight_grams"),
        mrp=row.get("mrp"),
        short_description=row.get("short_description"),
    )
    p = _clean_input(p)
    return Example(
        id=row["id"],
        product_input=p,
        target_description=cleaned_desc,
        db_ai_description=_clean_text(row.get("ai_description")),
    )


def split_examples(
    examples: list[Example],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = RANDOM_SEED,
) -> tuple[list[Example], list[Example], list[Example]]:
    """Deterministic shuffle + 80/10/10 split. Test set is the held-out 10%."""
    import random

    rng = random.Random(seed)
    pool = list(examples)
    rng.shuffle(pool)
    n = len(pool)
    train_n = int(train_ratio * n)
    val_n = int(val_ratio * n)
    return pool[:train_n], pool[train_n : train_n + val_n], pool[train_n + val_n :]


def write_jsonl(examples: list[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Example]:
    out: list[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Example.from_dict(json.loads(line)))
    return out


def export_all(out_dir: Path | None = None) -> dict[str, int]:
    """Run the full export. Returns counts per split."""
    out_dir = out_dir or DATA_DIR
    df = fetch_raw()
    examples: list[Example] = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            examples.append(row_to_example(row))
        except ValueError:
            skipped += 1
    train, val, test = split_examples(examples)
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(val, out_dir / "val.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")
    return {
        "fetched": len(df),
        "kept": len(examples),
        "skipped": skipped,
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
