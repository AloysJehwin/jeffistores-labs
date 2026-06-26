"""Generate short video clips via Stable Video Diffusion (image-to-video).

SVD takes a still image (1024x576) and animates it for ~25 frames (~4 sec at 6fps).
Quality is best on landscape, near-natural photo-ish images. Pure text/diagrams
don't animate well — SVD will try to "panate" them, which looks weird.

Usage:
  from gen_video import animate
  animate(Path("/tmp/scene.png"), Path("/tmp/scene.mp4"))
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

DEFAULT_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"


@lru_cache(maxsize=1)
def _pipe(model_id: str = DEFAULT_MODEL):
    import torch
    from diffusers import StableVideoDiffusionPipeline

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.enable_model_cpu_offload()      # offload UNet pieces to CPU between steps
    pipe.unet.enable_forward_chunking()  # chunk attention to lower peak
    # SVD's AutoencoderKLTemporalDecoder doesn't implement enable_tiling().
    # We rely on small decode_chunk_size in animate() to bound VAE-decode peak.
    return pipe


def animate(
    image_path: Path,
    out_path: Path,
    *,
    num_frames: int = 14,            # 4080 12GB sweet spot; 25 OOMs on full decode
    fps: int = 7,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    decode_chunk_size: int = 4,      # decode 4 frames at a time (vs all 14 at once)
    seed: int | None = None,
) -> Path:
    import torch
    from PIL import Image as PILImage
    from diffusers.utils import export_to_video

    img = PILImage.open(image_path).convert("RGB").resize((1024, 576))
    p = _pipe()
    g = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
    frames = p(
        img,
        num_frames=num_frames,
        decode_chunk_size=decode_chunk_size,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        generator=g,
    ).frames[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out_path), fps=fps)
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", default="/tmp/svd_out.mp4", type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--motion", type=int, default=127)
    args = parser.parse_args()
    p = animate(args.image, args.out, seed=args.seed, motion_bucket_id=args.motion)
    print(f"wrote {p}  ({p.stat().st_size // 1024} KB)")
