from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from openai import OpenAI

from .asr import Transcript
from .summarizer import NVIDIA_BASE_URL


@dataclass
class SpeakerStats:
    role: str
    word_count: int
    seconds: float
    segment_count: int
    avg_segment_seconds: float
    words_per_minute: float
    share: float  # доля от суммарного времени речи (0..1)


@dataclass
class TalkStats:
    speakers: list[SpeakerStats]
    total_audio_seconds: float
    total_speech_seconds: float
    silence_seconds: float
    overlap_seconds: float
    first_speaker: str | None
    longest_pause_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeepAnalysis:
    speaker_styles: dict
    emotion_timeline: list[dict]
    conflict_markers: list[dict]
    agreement_markers: list[dict]
    communication_quality: str
    power_balance: str
    next_steps: list[str]
    interesting_quotes: list[dict]
    risks: list[str] = field(default_factory=list)
    raw: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d


def compute_talk_stats(transcripts: dict[str, Transcript]) -> TalkStats:
    speakers: list[SpeakerStats] = []
    all_segments: list[tuple[float, float, str]] = []
    total_audio = 0.0

    role_totals: dict[str, float] = {}
    for role, t in transcripts.items():
        word_count = sum(len(s.text.split()) for s in t.segments if s.text)
        seconds = sum(max(0.0, s.end - s.start) for s in t.segments if s.text)
        segment_count = sum(1 for s in t.segments if s.text)
        avg_seg = seconds / segment_count if segment_count > 0 else 0.0
        wpm = (word_count / seconds * 60.0) if seconds > 0 else 0.0
        role_totals[role] = seconds
        speakers.append(
            SpeakerStats(
                role=role,
                word_count=word_count,
                seconds=seconds,
                segment_count=segment_count,
                avg_segment_seconds=avg_seg,
                words_per_minute=wpm,
                share=0.0,
            )
        )
        total_audio = max(total_audio, t.duration)
        for s in t.segments:
            if s.text:
                all_segments.append((s.start, s.end, role))

    total_speech = sum(role_totals.values())
    for sp in speakers:
        sp.share = (role_totals[sp.role] / total_speech) if total_speech > 0 else 0.0

    all_segments.sort(key=lambda x: x[0])

    overlap = 0.0
    for i, (a_start, a_end, a_role) in enumerate(all_segments):
        for b_start, b_end, b_role in all_segments[i + 1 :]:
            if b_start >= a_end:
                break
            if a_role != b_role:
                overlap += max(0.0, min(a_end, b_end) - max(a_start, b_start))

    # union of all speech intervals -> silence
    merged: list[tuple[float, float]] = []
    for start, end, _ in all_segments:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    union_speech = sum(e - s for s, e in merged)
    silence = max(0.0, total_audio - union_speech)

    longest_pause = 0.0
    for i in range(1, len(all_segments)):
        gap = all_segments[i][0] - all_segments[i - 1][1]
        longest_pause = max(longest_pause, gap)

    return TalkStats(
        speakers=speakers,
        total_audio_seconds=total_audio,
        total_speech_seconds=total_speech,
        silence_seconds=silence,
        overlap_seconds=overlap,
        first_speaker=all_segments[0][2] if all_segments else None,
        longest_pause_seconds=longest_pause,
    )


DEEP_ANALYSIS_PROMPT = """Ты — опытный аналитик коммуникации. Проанализируй транскрипт звонка глубоко.

Возвращай СТРОГО валидный JSON по схеме:
{
  "speaker_styles": {
    "Я": "1-2 предложения о стиле общения (тон, темп, манера)",
    "Собеседник": "то же для собеседника"
  },
  "emotion_timeline": [
    {"time_marker": "начало | середина | конец", "role": "Я | Собеседник", "emotion": "нейтрально | заинтересованно | раздражение | радость | сомнение | напор | усталость", "intensity": 1}
  ],
  "conflict_markers": [
    {"trigger": "что вызвало напряжение", "quote": "цитата из транскрипта"}
  ],
  "agreement_markers": [
    {"about": "о чём договорились", "quote": "цитата"}
  ],
  "communication_quality": "конструктивно | напряжённо | поверхностно | продуктивно | сумбурно",
  "power_balance": "сбалансированно | Я доминировал | Собеседник доминировал",
  "risks": ["возможные риски/недосказанности — что может выстрелить позже"],
  "next_steps": ["конкретные рекомендации для следующего шага со стороны Я"],
  "interesting_quotes": [
    {"role": "Я | Собеседник", "quote": "цитата", "reason": "почему важна"}
  ]
}

Правила:
- intensity: 1-5 (1 — слабо, 5 — очень сильно)
- emotion_timeline: 3-8 точек
- conflict_markers, agreement_markers: 0-5 каждое
- next_steps: 2-5 конкретных пунктов
- interesting_quotes: 2-4 цитаты, цитата дословная из транскрипта
- risks: 0-4 пункта
- НЕ выдумывай — если чего-то нет в звонке, оставляй пустой список
- Цитаты — реальные фрагменты из транскрипта"""


def deep_analyze(dialog: str, model: str | None = None) -> DeepAnalysis:
    if not dialog.strip():
        raise ValueError("empty dialog")

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")

    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
    model_name = model or os.getenv("LLM_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1")

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": DEEP_ANALYSIS_PROMPT},
            {"role": "user", "content": f"Транскрипт звонка:\n\n{dialog}"},
        ],
        temperature=0.3,
        top_p=0.9,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content or "{}"
    data = json.loads(raw)
    return DeepAnalysis(
        speaker_styles=data.get("speaker_styles", {}),
        emotion_timeline=data.get("emotion_timeline", []),
        conflict_markers=data.get("conflict_markers", []),
        agreement_markers=data.get("agreement_markers", []),
        communication_quality=data.get("communication_quality", ""),
        power_balance=data.get("power_balance", ""),
        next_steps=data.get("next_steps", []),
        interesting_quotes=data.get("interesting_quotes", []),
        risks=data.get("risks", []),
        raw=raw,
    )
