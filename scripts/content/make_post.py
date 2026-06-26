"""Orchestrate one full post: script.yaml -> images -> clips -> audio -> final mp4.

YAML schema:

    title: trl-3910-bisect
    voice_id: Matthew      # Polly voice; default Matthew
    scenes:
      - prompt: "clean line illustration of a version-vs-loss line chart with cliff"
        narration: "If you upgraded huggingface trl past zero point nineteen..."
        motion: 80           # SVD motion_bucket_id (low for diagrams)
      - static_image: experiments/04_jeffi_descgen/runs/post_image_B_curve.png
        narration: "..."
        motion: 30
      - prompt: "..."
        narration: "..."

A scene that provides `static_image` skips SDXL and uses that file directly.
Useful for charts/diagrams that SVD shouldn't try to "animate" but which we
still want SVD to apply subtle motion to (motion: 0 disables SVD entirely
and just holds the still frame for the narration duration).

Output:
    runs/<title>/scene_NN.png
    runs/<title>/scene_NN.mp4
    runs/<title>/narration_NN.mp3
    runs/<title>/final.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import yaml
from imageio_ffmpeg import get_ffmpeg_exe

from gen_audio import synth
from gen_image import generate as gen_image
from gen_video import animate as gen_video

FFMPEG = get_ffmpeg_exe()
FFPROBE = shutil.which("ffprobe") or str(Path(FFMPEG).with_name("ffprobe"))
RUNS = Path(__file__).resolve().parent / "runs"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _release_gpu(pipeline_module: str) -> None:
    """Free a cached pipeline from VRAM so the next stage has room.

    SDXL + SVD together (~17GB combined fp16) don't fit on a 12GB 4080. We run
    image gen first for all scenes, release SDXL, then run video gen for all
    scenes. This function does the release for a single pipeline module name.
    """
    import gc
    import torch
    import importlib
    mod = importlib.import_module(pipeline_module)
    if hasattr(mod, "_pipe"):
        mod._pipe.cache_clear()  # type: ignore[attr-defined]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def probe_duration(media: Path) -> float:
    """Return media duration in seconds via ffprobe. Falls back to ffmpeg parse."""
    try:
        out = subprocess.check_output(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(media)],
            stderr=subprocess.STDOUT,
        )
        return float(json.loads(out)["format"]["duration"])
    except (FileNotFoundError, subprocess.CalledProcessError, KeyError, ValueError):
        # Fallback: parse ffmpeg stderr "Duration: HH:MM:SS.ss"
        res = subprocess.run([FFMPEG, "-i", str(media)], capture_output=True, text=True)
        for line in res.stderr.split("\n"):
            if "Duration:" in line:
                dur = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = dur.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
        raise RuntimeError(f"could not probe duration of {media}")


def resolve_image_path(p: str | Path) -> Path:
    """Allow YAML to reference paths relative to the repo root."""
    pp = Path(p)
    if pp.is_absolute():
        return pp
    if (REPO_ROOT / pp).exists():
        return REPO_ROOT / pp
    return pp


def make_scene(scene: dict, scene_dir: Path, idx: int) -> tuple[Path, Path]:
    """Render one scene: (image | static) -> clip + narration mp3."""
    png = scene_dir / f"scene_{idx:02d}.png"
    mp4 = scene_dir / f"scene_{idx:02d}.mp4"
    mp3 = scene_dir / f"narration_{idx:02d}.mp3"

    # Image source: either provided as static, or generated via SDXL
    if "static_image" in scene:
        src = resolve_image_path(scene["static_image"])
        if not src.exists():
            raise FileNotFoundError(f"scene {idx}: static_image {src} not found")
        if not png.exists():
            shutil.copy(src, png)
    else:
        if not png.exists():
            print(f"  [{idx}] generating image: {scene['prompt'][:60].strip()}...")
            img = gen_image(scene["prompt"], seed=scene.get("seed"))
            img.save(png)

    if not mp4.exists():
        print(f"  [{idx}] animating to mp4...")
        gen_video(png, mp4, motion_bucket_id=scene.get("motion", 127),
                  seed=scene.get("seed"))

    if not mp3.exists():
        print(f"  [{idx}] synthesizing narration...")
        synth(scene["narration"].strip(), voice_id=scene.get("voice_id", "Matthew"),
              out_path=mp3)

    return mp4, mp3


def concat_clips(clip_audio_pairs: list[tuple[Path, Path]], out_path: Path) -> Path:
    """Pad each clip to the length of its audio, then concat with audio overlaid."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []

    for i, (clip, audio) in enumerate(clip_audio_pairs):
        secs = probe_duration(audio)
        # Loop video to match audio length, then mux audio
        part = out_path.parent / f"_part_{i:02d}.mp4"
        subprocess.run([
            FFMPEG, "-y", "-stream_loop", "-1", "-i", str(clip),
            "-i", str(audio),
            "-c:v", "libx264", "-c:a", "aac",
            "-t", f"{secs:.2f}",
            "-pix_fmt", "yuv420p",
            "-shortest",
            str(part),
        ], check=True, capture_output=True)
        parts.append(part)

    # Concat all parts via concat demuxer
    concat_list = out_path.parent / "_concat.txt"
    concat_list.write_text("\n".join(f"file '{p.name}'" for p in parts))
    subprocess.run([
        FFMPEG, "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy", str(out_path),
    ], check=True, capture_output=True, cwd=str(out_path.parent))

    for p in parts:
        p.unlink(missing_ok=True)
    concat_list.unlink(missing_ok=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path, help="YAML script file")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.script.read_text())
    title = cfg["title"]
    scene_dir = RUNS / title
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: all images (SDXL, or copy static_image), serial.
    for i, scene in enumerate(cfg["scenes"]):
        png = scene_dir / f"scene_{i:02d}.png"
        if png.exists():
            continue
        if "static_image" in scene:
            src = resolve_image_path(scene["static_image"])
            if not src.exists():
                raise FileNotFoundError(f"scene {i}: static_image {src} not found")
            shutil.copy(src, png)
        else:
            print(f"[image {i}] {scene['prompt'][:60].strip()}...")
            img = gen_image(scene["prompt"], seed=scene.get("seed"))
            img.save(png)

    # Release SDXL before loading SVD — they don't fit together on 12GB.
    _release_gpu("gen_image")

    # Phase 2: all narrations (Polly), cheap.
    for i, scene in enumerate(cfg["scenes"]):
        mp3 = scene_dir / f"narration_{i:02d}.mp3"
        if mp3.exists():
            continue
        print(f"[audio {i}]")
        synth(scene["narration"].strip(),
              voice_id=scene.get("voice_id", cfg.get("voice_id", "Matthew")),
              out_path=mp3)

    # Phase 3: all video animations (SVD), serial.
    for i, scene in enumerate(cfg["scenes"]):
        mp4 = scene_dir / f"scene_{i:02d}.mp4"
        png = scene_dir / f"scene_{i:02d}.png"
        if mp4.exists():
            continue
        print(f"[video {i}]")
        gen_video(png, mp4, motion_bucket_id=scene.get("motion", 127),
                  seed=scene.get("seed"))

    _release_gpu("gen_video")

    # Phase 4: collect (clip, audio) pairs and concat with mux.
    pairs: list[tuple[Path, Path]] = []
    for i in range(len(cfg["scenes"])):
        pairs.append((scene_dir / f"scene_{i:02d}.mp4",
                      scene_dir / f"narration_{i:02d}.mp3"))

    final = scene_dir / "final.mp4"
    print(f"concatenating {len(pairs)} scenes -> {final}")
    concat_clips(pairs, final)
    print(f"wrote {final} ({final.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
