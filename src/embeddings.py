from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from openai import OpenAI

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nv-embedqa-e5-v5"
MAX_CHARS = 4096


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY not set")
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=key)


def embed(text: str, input_type: str = "passage", model: str = DEFAULT_MODEL) -> np.ndarray:
    text = (text or "").strip()[:MAX_CHARS]
    if not text:
        raise ValueError("empty text for embedding")
    r = _client().embeddings.create(
        model=model,
        input=[text],
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    return np.array(r.data[0].embedding, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def session_text_for_embedding(session: dict) -> str:
    """Собрать репрезентативный текст сессии для embedding'а."""
    summary = session.get("summary") or {}
    parts: list[str] = []
    if summary.get("tldr"):
        parts.append(str(summary["tldr"]))
    for kp in summary.get("key_points", []) or []:
        parts.append(str(kp))
    for t in summary.get("topics", []) or []:
        parts.append(str(t))
    transcript = session.get("transcript", "") or ""
    if transcript:
        parts.append(transcript[:2000])
    return "\n".join(parts)
