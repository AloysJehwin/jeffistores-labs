# AI content pipeline — Razer + Polly

Generates short narrated videos from a YAML script. Razer GPU does the image and
video generation (SDXL + Stable Video Diffusion); AWS Polly Lambda does the voiceover.

## Components

- `gen_image.py` — SDXL text-to-image (Razer GPU)
- `gen_video.py` — Stable Video Diffusion image-to-video (Razer GPU)
- `gen_audio.py` — Polly via `PollyTextToSpeech` Lambda (us-east-1)
- `make_post.py` — orchestrator: takes a YAML script, produces `final.mp4`
- `runs/<title>/` — per-run artifacts (gitignored)

## Constraints learned from Phase 0

- Polly Lambda timeout = 3 seconds. `gen_audio.synth_long()` chunks text by sentence.
- Lambda payload cap = 6 MB base64. Per chunk should stay under ~30 sec of speech.
- SVD weights require `enable_model_cpu_offload()` to fit on the 4080 (12 GB).
- SDXL first-run weights download ~7 GB; cached at `~/.cache/huggingface/`.
- SVD weights ~10 GB; cached likewise.

## Usage (on Razer)

```bash
cd ~/Documents/GitHub-Personal/jeffistores-labs
.venv/bin/python scripts/content/make_post.py scripts/content/example_script.yaml
```

## YAML schema

```yaml
title: my-post
voice_id: Matthew         # default; overrideable per scene
scenes:
  - prompt: "clean line illustration of a version-vs-loss line chart with cliff"
    narration: "If you upgraded huggingface trl past zero point nineteen..."
    motion: 80            # SVD motion_bucket_id (0-255); lower = subtler
    seed: 42              # optional, for reproducibility
```

## What this is NOT for

- Long-form video (>~60 sec). Use a real editor for that.
- Photoreal human avatars. SDXL + SVD will produce uncanny humans.
- Anything safety-critical. Polly mispronounces tech terms (try SSML or phonetic spelling).
