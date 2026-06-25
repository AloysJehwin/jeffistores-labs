# ROADMAP — to Anthropic by 2026-12

Living document. Supersedes the original Month 1 / Month 2 narrative in the
README. Update weekly; honest revisions, not optimistic ones.

**Last revised:** 2026-06-25
**Target date:** December 2026
**Time remaining:** ~6 months

---

## Where this plan is honest about

The original plan assumed I would do fundamentals (Karpathy chain) before the
applied project (descgen). That isn't what happened. I shipped descgen Stages
1-3 in Month 1, before finishing micrograd → makemore → nanoGPT. As of today:

- `experiments/01_micrograd/`: scaffold + one half-done notebook
- `experiments/02_makemore/`: empty
- `experiments/03_nanogpt/`: empty
- `experiments/04_jeffi_descgen/`: Stages 1-3 done, Stage 4 not started

This is the **inverted risk profile**: portfolio-rich, fundamentals-light.
Most candidates have the opposite problem. It's correctable, but only if I
swing back to the Karpathy work *before* doing more applied projects.

Anthropic technical interviews test fundamentals (derive backprop, implement
attention, explain a loss curve). A portfolio gets you the interview; the
fundamentals get the offer. Both are required.

---

## Compass

Three things must be true by **2026-11** to make a credible application:

1. **Fundamentals can be defended on a whiteboard.** I can derive backprop,
   sketch a transformer block, explain why my loss curve does what it does,
   on demand, without notes. Confidence here is non-negotiable.
2. **At least one shipped, public, technically substantive artifact.** This
   project (or its successor) running publicly, with an honest writeup,
   results table, and code in a public repo. Not a screenshot.
3. **A non-trivial OSS footprint at huggingface/* or anthropics/*.** At
   least two merged PRs of real engineering substance, OR one merged PR + a
   pattern of high-quality diagnoses/comments on issues. Quantity is not
   the metric; signal is.

Everything in this roadmap funnels into those three.

---

## Schedule (revised, honest)

### Block A — fundamentals catch-up (Jun 26 – Jul 23, 4 weeks)

Goal: finish Karpathy's chain. Stage 4 of descgen is paused.

| Week | Target |
|---|---|
| Jun 26 – Jul 2 | `01_micrograd` end-to-end. Pytest green. `NOTES.md` honest. |
| Jul 3 – Jul 9 | `02_makemore` stages 1-3 (bigram, MLP, deepen). |
| Jul 10 – Jul 16 | `02_makemore` stages 4-5 (BatchNorm, WaveNet) + `03_nanogpt` setup. |
| Jul 17 – Jul 23 | `03_nanogpt`: train on Tiny Shakespeare, get a sample, write a 500-word note on what surprised me. |

Success check at end of Block A: I can sit down at a whiteboard, derive
backprop through a 3-node graph, and sketch single-head attention with
matrix shapes and the softmax dimension correct. If I can't, repeat the
weak chunk before moving on.

### Block B — descgen Stage 4 (Jul 24 – Aug 13, 3 weeks)

Goal: ship the descgen artifact publicly.

| Week | Target |
|---|---|
| Jul 24 – Jul 30 | Multi-seed runs (V1 + V2 × 3 seeds) to lock the DoRA negative result with variance bars. Then `scripts/05_eval_finetune.py` against `test.jsonl` for V1 + 4 baselines. |
| Jul 31 – Aug 6 | Claude-as-judge harness. ~$5-10 in API. Single 5×4 results table. |
| Aug 7 – Aug 13 | FastAPI + Streamlit demo (modest, runs on Razer behind Tailscale). Push V1 adapter to HF Hub with a real model card. Draft blog post. |

The blog post is **drafted in this block, polished in the next**. Don't
publish yet — Block C's work makes the writing better.

### Block C — RAG / semantic search (Aug 14 – Sep 13, 4 weeks)

Goal: a second, smaller project that demonstrates breadth.

| Week | Target |
|---|---|
| Aug 14 – Aug 20 | Pick a concrete RAG question (e.g. "best replacement for an obscure SKU"). Build the retrieval index against `jeffi_replica`. |
| Aug 21 – Aug 27 | Eval harness for RAG (retrieval@k, faithfulness via Claude judge). |
| Aug 28 – Sep 3 | Hybrid retrieval (BM25 + embedding), evaluate. Honest comparison. |
| Sep 4 – Sep 13 | Writeup + integrate the RAG service into the same FastAPI app as descgen. |

Polish and publish the descgen blog post at the end of this block — it
benefits from being written after seeing the eval-first discipline of
RAG.

### Block D — OSS push + interview prep (Sep 14 – Nov 30, ~11 weeks)

Goal: two more substantive OSS contributions + interview-shape exercises.

| Weeks | Target |
|---|---|
| Sep 14 – Oct 5 | Watch `huggingface/peft`, `huggingface/transformers`, `huggingface/trl` issue queues. Pick **real bugs you hit in your own RAG / descgen work** (not strangers' bugs). Open at least one PR. |
| Oct 6 – Oct 26 | LeetCode-style algorithms refresh + ML systems-design problems. NOT memorization — solving novel problems out loud. ~1 hr/day. |
| Oct 27 – Nov 16 | Mock interviews. Anthropic-flavored: implement attention from scratch with no internet, debug a broken training run, design an eval harness for X. Ideally with another engineer; alternatively self-recorded. |
| Nov 17 – Nov 30 | Final polish: portfolio README, public artifact links, application materials. |

### Block E — application (Dec 1 – Dec 15)

Goal: submit. Have referrals lined up first.

---

## Anti-goals (things this roadmap explicitly does NOT do)

- **Master more than 2 papers/techniques.** Quality over breadth. Pick DoRA's
  family or one alignment paper (e.g. Constitutional AI replication) — not
  five.
- **Pad the OSS PR count.** A merged 500-line PR with a real fix is worth more
  than 10 typo PRs. Quantity is suspicious.
- **Daily progress posts on X / LinkedIn.** Already addressed in
  `posting_strategy.md`. Empty progress is anti-signal.
- **Multiple capstone projects.** descgen + RAG is enough. A third would
  spread effort thin and give nothing depth.
- **Touch Jeffi prod from any of these experiments.** Demo only, behind
  Tailscale. Production integration is its own separate project that needs
  review + rollback + A-B test infra.

---

## Networking (the part the original plan omitted)

Anthropic hires on referrals more than cold applications. By **Block C
(Aug-Sep)** I should have started building presence with people who actually
build at Anthropic — not just DevRel:

- Quality replies on technical blog posts (Anthropic engineering blog,
  individual model-team / safety-team posts)
- Useful comments on PRs in `anthropics/*` and on huggingface/* where Anthropic
  contributors are active
- Attend at least one Anthropic / safety meetup / paper-discussion if any are
  reachable from where I am

Not stalking. Just being a recognizable, useful presence. The PR #429 and the
trl #3910 bisect comment are starts; the goal is a consistent pattern of high-
quality engineering replies.

---

## Decision log (revise weekly)

- **2026-06-25** — paused descgen Stage 4 in favor of Block A (Karpathy
  catch-up). Reason: Stage 3 surfaced that I could ship a QLoRA pipeline
  without being able to defend it at depth. Fundamentals first, polish later.
- **2026-06-25** — opened first PR to `anthropics/claude-quickstarts` (#429)
  and bisect-diagnosis on `huggingface/trl#3910`. Treat OSS as
  byproduct-of-real-work going forward, not a separate quest.
- **2026-06-25** — repo is currently private. Revisit at start of Block C —
  may make a separate public portfolio repo so the working scratch space
  can stay private.

---

## Where this roadmap may be wrong

I'm flagging this honestly, because the roadmap is more useful when it's
revisable:

1. **4 weeks for Karpathy might not be enough.** If micrograd takes 1.5 weeks
   instead of 1, makemore stages 4-5 might compress into a weekend.
   Compressing is OK *only* if the success check at end of Block A passes.
   If it doesn't, extend by 2 weeks and slip everything.
2. **Multi-seed at Block B's start might surface that the DoRA negative
   result was actually significant.** That changes the blog post. Be ready.
3. **The OSS push (Block D) might happen organically before Sep 14.** That's
   fine — but it shouldn't displace fundamentals work in Block A.

Review this doc weekly. Bias toward acknowledging slippage early.
