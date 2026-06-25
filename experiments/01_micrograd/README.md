# 01 — micrograd

Build a tiny autograd engine from scratch following Karpathy's *Spelled-out intro
to neural networks and backprop*. The goal is not to "get through the video" —
it is to be able to (a) derive backprop on a small graph on a whiteboard, and
(b) debug a real loss curve later because you know what the gradient is doing.

**Reference (canonical):** [Karpathy — The spelled-out intro to neural networks
and backprop](https://www.youtube.com/watch?v=VMj-3S1tku0) (~2.5 hrs)

**Anti-goal:** copy-pasting `Value` + writing "I built micrograd" on the resume.
That counts for nothing. The scaffold below forces you to derive each step.

---

## Schedule

| Day | Notebook / file | What you finish | Time |
|---|---|---|---|
| 1 | `01_value_and_autograd.ipynb` | A working `Value` class with `+`, `*`, `tanh`, and a manual `backward()` on a 3-node graph. **Match PyTorch gradients exactly.** | 2 hrs |
| 2 | `02_topological_backward.ipynb` | Replace manual backward with proper topo-sort. Add `exp`, `**`, `/`, `relu`. Gradient-check 5 random expressions against autograd. | 2 hrs |
| 3 | `03_neuron_layer_mlp.ipynb` | Build `Neuron`, `Layer`, `MLP` classes. Train a 2-3-1 MLP on the 4-point XOR-ish dataset from the lecture. Loss < 0.01 in < 200 steps. | 2 hrs |
| 4 | `micrograd_solo.py` + `tests/test_micrograd.py` | **Re-implement from scratch, no copy-paste from notebooks.** Pytest must cover: (a) gradient-check on 3 expressions, (b) MLP trains on XOR, (c) graph DAG topology is correct. | 3 hrs |

Total: ~9 hrs over ~4 days.

## What "done" looks like (objective checks)

You can declare micrograd done when **all** of these are true:

- [ ] You can take a 4-node expression like `((a + b) * tanh(c) - d).backward()` on a whiteboard and predict every `.grad` correctly *before* running the code.
- [ ] `pytest experiments/01_micrograd/tests/` is green.
- [ ] Your `MLP` reaches loss < 0.01 on Karpathy's XOR-ish dataset in <200 steps with `lr=0.05`.
- [ ] You can explain in one sentence why we use `tanh` and not `sigmoid` in this lecture, and why the `_prev` set is needed.
- [ ] You wrote a 200-word note in `NOTES.md` of what surprised you. (Skipping this is the strongest sign you didn't actually engage.)

## Files in this folder

- `README.md` — this file
- `01_value_and_autograd.ipynb` — code-along (today's stub)
- `02_topological_backward.ipynb` — TODO
- `03_neuron_layer_mlp.ipynb` — TODO
- `micrograd_solo.py` — TODO (Day 4: solo re-implementation)
- `tests/test_micrograd.py` — TODO (Day 4)
- `NOTES.md` — append-only as you learn

## Common traps Karpathy doesn't flag

These bite everyone going through the lecture. Pre-warning so you don't lose half a day:

1. **The `+=` bug in `backward()`.** If a node is used twice in a graph (e.g. `b = a + a`), naive assignment `self.grad = ...` overwrites the second contribution. Always **accumulate** with `+=`. The clue: small graphs that work fine in isolation suddenly give wrong gradients when a variable is reused.

2. **`_prev` as a `set` of `Value` objects.** `Value` is unhashable unless you define `__hash__` / `__eq__`. Karpathy's version skips this by default — Python's default object identity works. If you go too fancy and override `__eq__` for numeric comparison, you'll break the set semantics.

3. **The topo sort is *post-order* of the DFS, then reversed.** Easy to write it as pre-order and get gradients that mostly work on small graphs and silently corrupt on bigger ones.

4. **`Value.__radd__` etc.** are needed for expressions like `2 + v`. Karpathy adds them as a one-liner but the lecture skips past it.

5. **You will get bored of typing `Value(...)`.** Resist the urge to refactor early. The exercise is the typing.

## Sanity check before you start (Day 1, first 5 min)

You need the venv working. The current notebook fails with `No module named 'torch'`. Fix:

```bash
cd ~/Documents/GitHub-Personal/jeffistores-labs
.venv/bin/python -c "import torch; print(torch.__version__)"
```

If that errors, you need to activate the env in your Jupyter kernel:

```bash
.venv/bin/python -m ipykernel install --user --name jeffistores-labs --display-name "jeffistores-labs (.venv)"
```

Then in the notebook, kernel picker → "jeffistores-labs (.venv)".

## When to move on

When all checkboxes above are ticked, go to `experiments/02_makemore/`. Do *not*
keep polishing micrograd — diminishing returns. The lecture has more
hand-holding than makemore does; that's by design.

## Notes (append as you learn)

— blank, write here —
