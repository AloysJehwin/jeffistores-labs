"""Eval harness for description generation.

Single source of truth for "how good is this generator?". Every baseline,
every fine-tune, every API model goes through this.

Metrics
-------
1. BLEU-4              — n-gram overlap with reference (sacrebleu)
2. ROUGE-L F1          — longest common subsequence (rouge_score)
3. Semantic similarity — cosine sim of sentence-transformer embeddings
4. Length ratio        — len(pred) / len(ref) — sanity check, not graded

Generator protocol
------------------
Anything that implements `Generator(name: str)` with a `.generate(p: ProductInput) -> str`
method works here. See `baselines.py` for empty / copy / phi3-zero-shot / db-ai-description.

Usage
-----
    from jeffistores_labs.descgen.eval import evaluate, Generator
    from jeffistores_labs.descgen.dataset import read_jsonl, DATA_DIR

    test = read_jsonl(DATA_DIR / "test.jsonl")
    results = evaluate(my_generator, test, limit=50)
    print(results.summary())

Why these four metrics specifically
-----------------------------------
- **BLEU-4** is harsh on short references and paraphrases. Will be low across
  the board (5–15 typical for catalog text). Useful *relatively*: which
  generator beats which.
- **ROUGE-L** is more forgiving of word-order shuffles. Better for
  free-form descriptions than BLEU.
- **Cosine similarity** ignores wording entirely; it asks "are these about
  the same thing?". This is the metric that should improve most after
  fine-tuning, because the model learns the *domain*.
- **Length ratio** isn't graded — it's a guardrail. If your generator
  consistently outputs 5× longer or shorter than references, something's wrong.

Anti-patterns avoided
---------------------
- We DO NOT compute the generator's loss on the references. Loss != quality.
- We DO NOT report a single composite score. They each measure different
  things; collapse them and you lose the diagnostic value.
- We DO NOT include exact-match accuracy. Generation isn't classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

from .dataset import Example, ProductInput

# -----------------------------------------------------------------------------
# Generator protocol
# -----------------------------------------------------------------------------


class Generator(Protocol):
    """Anything with a name and a generate(ProductInput) -> str method."""

    name: str

    def generate(self, product: ProductInput) -> str: ...


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------


@dataclass
class Prediction:
    example_id: str
    reference: str
    prediction: str


@dataclass
class EvalResult:
    generator_name: str
    n: int
    bleu: float
    rouge_l_f1: float
    cosine_similarity: float
    length_ratio: float                    # mean of len(pred) / len(ref)
    per_example: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.generator_name:<24} n={self.n:<4} "
            f"BLEU={self.bleu:5.2f}  ROUGE-L={self.rouge_l_f1:5.2f}  "
            f"cos-sim={self.cosine_similarity:5.3f}  len-ratio={self.length_ratio:.2f}"
        )


# -----------------------------------------------------------------------------
# Lazy metric backends — only imported when first used so importing this
# module is cheap (matters for the dataset script + tests).
# -----------------------------------------------------------------------------


def _bleu(predictions: list[str], references: list[str]) -> float:
    import sacrebleu  # type: ignore[import-untyped]

    # sacrebleu expects references as a list of N reference-sets (one per example).
    return float(sacrebleu.corpus_bleu(predictions, [references]).score)


def _rouge_l(predictions: list[str], references: list[str]) -> float:
    from rouge_score import rouge_scorer  # type: ignore[import-untyped]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    f1s = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    return 100.0 * sum(f1s) / max(len(f1s), 1)


_EMBED_MODEL_CACHE = {}


def _cosine_similarity(predictions: list[str], references: list[str]) -> float:
    """Mean cosine sim of sentence-transformer embeddings.

    Uses all-MiniLM-L6-v2 — small (90 MB), fast, good enough for relative ranking.
    """
    from sentence_transformers import SentenceTransformer, util  # type: ignore[import-untyped]

    if "minilm" not in _EMBED_MODEL_CACHE:
        _EMBED_MODEL_CACHE["minilm"] = SentenceTransformer("all-MiniLM-L6-v2")
    model = _EMBED_MODEL_CACHE["minilm"]

    pred_emb = model.encode(predictions, convert_to_tensor=True, show_progress_bar=False)
    ref_emb = model.encode(references, convert_to_tensor=True, show_progress_bar=False)
    sims = util.cos_sim(pred_emb, ref_emb).diagonal()
    return float(sims.mean())


def _length_ratio(predictions: list[str], references: list[str]) -> float:
    ratios = [len(p) / max(len(r), 1) for p, r in zip(predictions, references)]
    return sum(ratios) / max(len(ratios), 1)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def evaluate(
    generator: Generator,
    examples: Iterable[Example],
    *,
    limit: int | None = None,
) -> EvalResult:
    """Run a generator over examples and compute all metrics."""
    rows: list[Prediction] = []
    for i, ex in enumerate(examples):
        if limit is not None and i >= limit:
            break
        pred = generator.generate(ex.product_input)
        rows.append(Prediction(ex.id, ex.target_description, pred))

    preds = [r.prediction for r in rows]
    refs = [r.reference for r in rows]

    return EvalResult(
        generator_name=generator.name,
        n=len(rows),
        bleu=_bleu(preds, refs) if preds else 0.0,
        rouge_l_f1=_rouge_l(preds, refs) if preds else 0.0,
        cosine_similarity=_cosine_similarity(preds, refs) if preds else 0.0,
        length_ratio=_length_ratio(preds, refs) if preds else 0.0,
        per_example=[
            {"id": r.example_id, "reference": r.reference, "prediction": r.prediction}
            for r in rows
        ],
    )


def evaluate_many(
    generators: list[Generator],
    examples: list[Example],
    *,
    limit: int | None = None,
) -> list[EvalResult]:
    """Convenience: run multiple generators on the same set, return results table."""
    return [evaluate(g, examples, limit=limit) for g in generators]


def render_results_table(results: list[EvalResult]) -> str:
    """Plain-text comparison table for terminal output."""
    if not results:
        return "(no results)"
    header = f"{'generator':<24} {'n':<5} {'BLEU':>6} {'ROUGE-L':>8} {'cos-sim':>9} {'len-ratio':>10}"
    sep = "-" * len(header)
    rows = [
        f"{r.generator_name:<24} {r.n:<5} {r.bleu:>6.2f} {r.rouge_l_f1:>8.2f} "
        f"{r.cosine_similarity:>9.3f} {r.length_ratio:>10.2f}"
        for r in results
    ]
    return "\n".join([header, sep, *rows])
