from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from functools import lru_cache

from openai import OpenAI

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

SYSTEM_PROMPT = """Ты — ассистент, который анализирует транскрипты голосовых заметок и встреч на русском языке.
Твоя задача — извлечь из текста структурированную информацию.

Отвечай СТРОГО валидным JSON без markdown-обёртки, по схеме:
{
  "tldr": "одно-двух-предложенное резюме",
  "key_points": ["важная мысль 1", "важная мысль 2", ...],
  "action_items": ["задача 1", "задача 2", ...],
  "topics": ["тема 1", "тема 2", ...],
  "sentiment": "positive | neutral | negative",
  "language": "ru"
}

Правила:
- key_points: 3-7 пунктов, каждая мысль самодостаточна
- action_items: только явные задачи/дела (пустой список если их нет)
- topics: 1-5 тем верхнего уровня
- НЕ выдумывай факты, которых нет в транскрипте
- Если транскрипт короткий или бессвязный — отрази это в tldr"""


@dataclass
class Summary:
    tldr: str
    key_points: list[str]
    action_items: list[str]
    topics: list[str]
    sentiment: str
    language: str
    raw: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set — добавь ключ в .env")
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


def summarize(transcript: str, model: str | None = None) -> Summary:
    if not transcript.strip():
        raise ValueError("empty transcript")

    model_name = model or os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct")
    completion = _client().chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Транскрипт:\n\n{transcript}"},
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content or "{}"
    data = json.loads(raw)
    return Summary(
        tldr=data.get("tldr", ""),
        key_points=data.get("key_points", []),
        action_items=data.get("action_items", []),
        topics=data.get("topics", []),
        sentiment=data.get("sentiment", "neutral"),
        language=data.get("language", "ru"),
        raw=raw,
    )
