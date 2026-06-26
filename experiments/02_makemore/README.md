# 02 — makemore

Character-level language modeling from scratch, following Karpathy's
[makemore series](https://github.com/karpathy/makemore). Same task at every
stage — given a partial name, predict the next character — different
architectures. The progression mirrors the history of language modeling:
probability tables → linear neural net → MLP with embeddings → BatchNorm →
WaveNet hierarchical context.

The dataset: 32,032 baby names in `data/names.txt`. Tiny but rich. The first
character of names is heavily skewed (lots of 'A', 'M', 'J'); intra-name
character transitions are highly non-uniform. Real linguistic structure to
fit, small enough to train in minutes.

## Why this stage matters

`micrograd` proved you understand backprop on scalars. **`makemore` proves
you understand model design on tensors.** From Stage 2.2 onward we use
PyTorch — your hand-rolled `Value` is gone. What survives is the mental
model: every tensor op has a forward pass and a local derivative; backward
is `.backward()` walking the graph.

The lesson at each stage is *the loss number and the samples*. A trained
bigram model produces `babi`, `kayly`, `azhi` — recognizably name-shaped.
A trained MLP produces `kira`, `axton`, `lillenne` — varied, plausible.
The numbers tell you which architecture moved the needle and which didn't.

## Reference videos (Karpathy)

- [The spelled-out intro to language modeling: bigrams](https://www.youtube.com/watch?v=PaCmpygFfXo) — stages 2.1 and 2.2
- [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) — stage 2.3
- [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) — stage 2.4 (warm-up)
- [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) — stage 2.4 (manual backprop)
- [Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) — stage 2.5

## Stages

| Notebook | Stage | What it ships | Target NLL |
|---|---|---|---|
| `01_bigram.ipynb` | **2.1** — counting bigrams | 27×27 frequency table, sampler, baseline loss | **~2.46** |
| `02_bigram_nn.ipynb` | **2.2** — neural bigram | Same task via a 27×27 weight matrix + softmax. Should match 2.1's NLL. | ~2.46 |
| `03_mlp.ipynb` | **2.3** — Bengio MLP | Embedding table + context concat + tanh + softmax. Block size 3. | **~2.10** |
| `04_batchnorm.ipynb` | **2.4** — BatchNorm + manual backprop | 6-layer net, hand-derive every backward pass, validate against autograd | ~2.05 |
| `05_wavenet.ipynb` | **2.5** — WaveNet-style depth | Hierarchical context (Karpathy's tree of MLPs). Spatial bias matters. | ~1.99 |

Each notebook ends with **10 sampled names** so the architecture's effect is qualitative, not just numerical.

## Common traps (calling out before they bite)

1. **Bigram NLL is averaged over *characters*, not names.** A 4-character name with NLL=2.0 contributes log p of -2.0 per char, summed over 4 chars. Standard. Avoid summing per name and forgetting to divide.

2. **The `.` start/end token doubles as both.** A name `emma` becomes `.emma.` → 5 transitions: `.→e`, `e→m`, `m→m`, `m→a`, `a→.`. One vocabulary, one token, two roles.

3. **PyTorch's `cross_entropy` already includes softmax + log.** Don't double-softmax. The forward pass of stage 2.2+ takes raw logits.

4. **Don't forget `requires_grad=True` when you handcraft a weight tensor.** Easy to forget when you're not using `nn.Module`. Karpathy hits this in lecture 2.

## Anti-goals (per ROADMAP.md)

- Skipping straight to 2.4/2.5 because they're more "impressive." Each stage's number only matters relative to the one before.
- Training on a larger dataset. The names dataset is calibrated to fit on CPU in minutes per stage.
- Replacing handcrafted modules with `nn.Linear` etc. Stage 2.4 is **the** lecture where building from scratch matters.

## What "done" looks like

For each stage:

- [ ] Loss on a held-out 10% dev set hits the target above
- [ ] 10 sampled names look qualitatively different from the previous stage
- [ ] One line appended to `NOTES.md` describing what surprised you

Block A is done when all five stages above are ticked AND you can articulate (without notes) what each architectural change buys you. Then move to `03_nanogpt/`.
