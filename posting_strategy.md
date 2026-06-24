# Public Posting Strategy

**Goal**: become *visible* to ML researchers, Anthropic engineers, and AI-eng managers — without becoming an "AI Twitter influencer" type. The point is depth and substance, not engagement.

**Time horizon**: aligned with the 6-month roadmap to Anthropic (`ML_JOURNEY.md`).

---

## Why I'm not posting daily

Three reasons (revisit if any change):

1. **Daily slop dilutes signal.** People at Anthropic / DeepMind / labs are not on X for hot takes — they post when they have something genuinely worth saying. Mimic that.
2. **Posting takes hours per week** that should go to Karpathy, the QLoRA fine-tune, paper reading.
3. **Empty progress posts** ("Day 5 of learning ML!") actively hurt the signal — they read as performative rather than substantive.

**Cadence guardrail**: post when there's a real artifact (commit, eval result, insight, paper reproduction). Aim for **1–4 substantive posts per month**, not daily. If you don't have something to say, don't say anything.

---

## What's allowed to post

A post must satisfy at least **one** of these:

- [ ] **Real result**: a chart, an eval table, a benchmark, a measured speedup
- [ ] **Surprising finding**: something contrarian backed by data
- [ ] **Concrete artifact**: a working tool, a public adapter on HF Hub, a blog post URL
- [ ] **Honest failure**: "I expected X, got Y, here's what I learned" — these are gold
- [ ] **Paper take**: a sharp 1–2 paragraph distillation of someone else's work, with your read

If a draft post doesn't tick at least one box, kill it.

---

## What is NEVER allowed

These are professionally costly. If a draft contains any of them, delete:

- ❌ **AI-generated text** posted as your own thoughts. Anthropic engineers can smell this from orbit.
- ❌ **Hype words**: "game-changing", "10x", "the future of", "revolutionary", "AGI is here"
- ❌ **Vague learning logs** ("Day 5 of learning ML, Karpathy's video #3 is amazing!")
- ❌ **Engagement bait**: "What do you think?", reply-asking, follow-for-more
- ❌ **Cross-posting the same thing to 4 platforms** with no adaptation
- ❌ **Day-counter posts** unless paired with a real artifact ("Day 30 — fine-tuned adapter beats baselines on 4/4 metrics" is fine; "Day 30 of my ML journey!" is not)
- ❌ **Anything political, snarky, or hot-takey** about other AI labs / engineers / projects
- ❌ **Posting before a result is *complete*** — the X feed will not remember your "spoiler" once the result lands

---

## Platforms

| Platform | Use for | Cadence |
|---|---|---|
| **X (Twitter)** | Real-time signal to ML community. **Primary hiring channel for AI labs.** | 1–4 substantive posts/month |
| **Personal blog (Hashnode or Substack)** | Long-form writeups: "I built X and here's what I learned." Anthropic interviewers read these. | 1 post per major milestone (~monthly) |
| **LinkedIn** | Indian / SAP-adjacent network, useful for Forward Deployed Engineer roles. Adapted from blog/X — never just cross-post. | 1 post per milestone |
| **GitHub** | The artifact itself. Most important — code, README, model cards, notebooks with baked outputs. | Daily commits to `jeffistores-labs` |
| **Hugging Face Hub** | Push fine-tuned adapters + model cards. Real ML-portfolio currency. | Each successful training run |
| **Instagram** | ❌ Skip. Wrong audience. |

---

## Voice & style

- **First person, factual**. "I trained Phi-3 on 952 industrial-hardware product descriptions." Not: "🚀 Just shipped a custom LLM!"
- **Numbers in every post**. BLEU, cos-sim, % improvement, training time, parameter count. Numbers are credibility.
- **Show the work**: link to the GitHub commit, the notebook, the W&B run, the HF adapter. Posts without artifacts are claims.
- **Concede things**. "Surprised me — I expected this to be true but the data says no." Honesty signals seriousness.
- **No emojis except for charts/figures**. (Personal preference — feels free to relax this. But never emoji-spam.)

---

## 30-day rolling plan

**Days 1–28 (Karpathy month)**: **silent**. Code, commit, learn. No public posts.

**Day ~30** (after nanoGPT done + descgen baselines published): **first post**.
- X thread: "I built an eval harness for product description generation, then ran 5 baselines. Two findings surprised me." (See `drafts/x_baseline_findings.md`)
- LinkedIn version: longer, more context-setting, ends with "What I'm working on next." (See `drafts/linkedin_baseline_findings.md`)
- GitHub: pin the `jeffistores-labs` repo with a polished README

**Day ~50** (after real QLoRA training run beats baselines): **second post**.
- X: "Update — fine-tuned Phi-3-mini on the same dataset, here's the eval table."
- Blog: full writeup with loss curves, 3 things that didn't work, hyperparam table
- HF Hub: push the adapter publicly with a model card

**Day ~75** (semantic search project shipped): **third post**.

**Day ~120 (start applying)**: switch from "building in public" to "ready to interview." Pin the strongest 2 posts to your X profile.

---

## Success metrics (set honestly, ignore vanity)

| Metric | Target by Day 180 | Why |
|---|---|---|
| **Substantive posts** | 6–10 | Quality, not volume |
| **GitHub stars** on `jeffistores-labs` | 10–50 | Validates the work has substance, not theatrics |
| **HF Hub adapter downloads** | 50+ | Someone actually used what you built |
| **DMs from researchers/recruiters** | ≥1 | The whole point |
| **Blog post readers** (analytics) | 200+ per post | Proves substance over velocity |
| **Followers** | Don't track | Vanity metric. Will follow naturally if other metrics hit. |

If you hit follower count without hitting the other metrics, **you've drifted** — pull back, refocus on substance.

---

## Anti-burnout rules

1. **One post draft per week max** in writing time, regardless of cadence
2. **Never post within 30 minutes of waking up** — sleep on it
3. **Never reply-engage in arguments** on X. Mute liberally. Block freely.
4. **Posting time-budget**: 1h/week max. Karpathy/QLoRA work always wins the calendar fight.

---

## How to post

1. Write the draft in `drafts/` here in the repo (markdown)
2. Sit on it for 24 hours
3. Re-read. Cut 30%.
4. Post. Add link to artifact (GitHub commit / blog / HF Hub).
5. Commit the draft after posting (so future-you can audit what you've shared publicly)
6. **Don't** edit the post after publishing unless there's a factual error
7. **Do not** check engagement metrics for 24h — work, then look once

---

## Owned channels (links)

- GitHub: https://github.com/AloysJehwin
- jeffistores-labs: https://github.com/AloysJehwin/jeffistores-labs
- X handle: _(set up clean profile when first post is ready)_
- Blog: _(create on Hashnode or Substack at first post)_
- Hugging Face: _(create at first adapter push)_

---

## Living document

This file is wrong somewhere. Edit it when you find out where. Don't apologize, just fix it and commit.
