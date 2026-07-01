# micrograd — notes

Append-only log. One line per insight. Honest is better than impressive.

Format:
- `YYYY-MM-DD [day-N]: thing that clicked or didn't`

---

2026-06-26 [day-1]: The `+=` in `_backward` isn't optional — it's what makes diamond-shaped graphs work. Without it, only the last path through a shared node contributes a gradient and you silently lose half your update.

2026-06-26 [day-1]: Topological sort via post-order DFS then reversed. Post-order means a node is appended *after* all its children — reversing gives you root-first, which is exactly the order you need to walk backwards.

2026-06-27 [day-2]: Each `_backward` closure is captured at forward time, not call time. That's why it works: `out.grad` is read *when the closure runs*, not when it was defined, so the accumulated gradient is already there.

2026-06-27 [day-2]: `sum(generator, start_value)` with a `Value` as start is the clean trick for the dot product in Neuron — avoids a separate zero-init accumulator and keeps the whole computation inside the autograd graph.

2026-06-28 [day-3]: Forgetting `p.grad = 0.0` before `loss.backward()` is the most common training bug. Gradients accumulate across steps via `+=`, so step 2 sees step-1 gradients added in — loss bounces or diverges and there's no error message.

2026-06-28 [day-3]: Symmetric weight init matters. Every neuron initialized to 0 computes the same output → same gradient → same update every step. The layer never differentiates and learning stalls. Uniform(-1, 1) breaks this by construction.

2026-06-30 [iris]: The output layer must be *linear* (no tanh). Tanh squashes outputs to (-1, 1) which wrecks softmax — you get near-uniform probabilities on every example, loss barely moves past ln(3) ≈ 1.1, and the model can't learn class separation.

2026-06-30 [iris]: Cross-entropy needs numerical stability. Subtracting max(logits) before exp prevents overflow without changing the result (the constant cancels in numerator and denominator). Without it, large logits → exp → inf → nan loss.

2026-07-02 [jeffi-products]: LR=0.05 diverged after step 50 on the jeffi 5-class problem — loss climbed from 1.65 back to 4.49 while accuracy flatlined at 48%. Same LR that worked on Iris (120 examples, 3 classes) overshoots on a harder surface (702 examples, 5 classes with heavy overlap). Fix: lr=0.01.

2026-07-02 [jeffi-products]: Screws, Bolts, and Nuts have nearly identical median dimensions (20×15×4cm, 500g) — the confusion matrix will be a mess for those three. Drill Bits (thin, light, 100g) and Non-Sparking Tools separate more cleanly. Physical dims alone can't distinguish fastener subtypes; you'd need material or thread-pitch data.
