"""Generate audio narration via the PollyTextToSpeech API Gateway endpoint.

The Lambda is fronted by an HTTP API at:
    POST https://38ebtgecfh.execute-api.us-east-1.amazonaws.com/polly
with AWS_IAM auth (SigV4 signed with the caller's AWS credentials).

Request body (JSON):
    {"text": str, "voice_id": str, "text_type": "text"|"ssml"}

Response (when Content-Type is audio/mpeg):
    raw mp3 bytes  (API Gateway decodes the Lambda's base64 reply for binary
                    Content-Type so we receive ready-to-write mp3)

Constraints:
  - Lambda timeout = 3s. Chunk long text by sentence (use chunk_text).
  - Lambda response cap = 6 MB. Stay under ~30s of audio per chunk.

Usage:
  from gen_audio import synth, synth_long
  synth("Hello world", out_path=Path("/tmp/hello.mp3"))
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
import urllib.request

API_URL = os.environ.get(
    "POLLY_API_URL",
    "https://38ebtgecfh.execute-api.us-east-1.amazonaws.com/default/polly",
)
REGION = "us-east-1"
SERVICE = "execute-api"
DEFAULT_VOICE = "Matthew"


def _signed_post(body_json: str) -> bytes:
    """POST to the API Gateway with SigV4. Returns raw response body bytes."""
    creds = boto3.Session().get_credentials().get_frozen_credentials()
    request = AWSRequest(
        method="POST",
        url=API_URL,
        data=body_json,
        headers={"Content-Type": "application/json"},
    )
    SigV4Auth(creds, SERVICE, REGION).add_auth(request)

    req = urllib.request.Request(
        API_URL, data=body_json.encode("utf-8"),
        headers=dict(request.headers), method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


def synth(text: str, *, voice_id: str = DEFAULT_VOICE, out_path: Path) -> Path:
    """Synthesize a short string (<=~20 words). Raises on non-2xx."""
    body = json.dumps({"text": text, "voice_id": voice_id, "text_type": "text"})
    try:
        audio_bytes = _signed_post(body)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Polly endpoint {e.code}: {e.read()[:300]!r}") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)
    return out_path


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = 180) -> list[str]:
    """Split narration into chunks that fit comfortably under the 3s Lambda timeout."""
    sentences = _SENTENCE.split(text.strip())
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if not s:
            continue
        if len(buf) + 1 + len(s) <= max_chars:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return chunks


def synth_long(text: str, *, voice_id: str = DEFAULT_VOICE, out_dir: Path) -> list[Path]:
    """Chunk a long narration and synthesize each piece to a separate mp3.

    Returns list of mp3 paths in order. Concatenation is the caller's job.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunk_text(text)
    paths: list[Path] = []
    for i, chunk in enumerate(chunks):
        p = out_dir / f"chunk_{i:03d}.mp3"
        synth(chunk, voice_id=voice_id, out_path=p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("--voice", default=DEFAULT_VOICE)
    p.add_argument("--out", default="/tmp/polly_out.mp3", type=Path)
    args = p.parse_args()
    out = synth(args.text, voice_id=args.voice, out_path=args.out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
