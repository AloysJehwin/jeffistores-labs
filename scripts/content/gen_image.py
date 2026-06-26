"""Generate images via Stable Diffusion XL on the Razer GPU.

Loads SDXL once into a module-global pipeline so callers can request many
images per session without re-loading the ~7GB of weights.

Usage:
  from gen_image import generate
  img = generate("a clean technical diagram of a transformer block")
  img.save("/tmp/out.png")
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_NEGATIVE = (
    "blurry, low quality, distorted, watermark, signature, text artifacts, "
    "deformed, ugly, oversaturated"
)


@lru_cache(maxsize=1)
def _pipe(model_id: str = DEFAULT_MODEL):
    """Load SDXL once. Cached for the process lifetime."""
    import torch
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    # Save VRAM on the 4080 12GB: enable attention slicing.
    pipe.enable_attention_slicing()
    return pipe


def generate(
    prompt: str,
    *,
    negative_prompt: str = DEFAULT_NEGATIVE,
    width: int = 1024,
    height: int = 1024,
    steps: int = 25,
    guidance: float = 5.5,
    seed: int | None = None,
) -> "Image.Image":
    import torch
    p = _pipe()
    g = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
    return p(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=g,
    ).images[0]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--out", default="/tmp/sdxl_out.png", type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=25)
    args = parser.parse_args()
    img = generate(args.prompt, seed=args.seed, steps=args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"wrote {args.out}")
