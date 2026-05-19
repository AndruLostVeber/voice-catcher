from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    language: str
    segments: list[Segment]
    duration: float


@lru_cache(maxsize=1)
def _load_model(name: str) -> WhisperModel:
    device = os.getenv("WHISPER_DEVICE", "cpu").lower()
    if device == "cuda":
        try:
            return WhisperModel(name, device="cuda", compute_type="float16")
        except Exception as e:
            print(f"[asr] CUDA load failed ({e}); falling back to CPU")
    return WhisperModel(name, device="cpu", compute_type="int8")


def transcribe(
    audio_path: str | Path,
    model_name: str | None = None,
    language: str | None = "ru",
) -> Transcript:
    name = model_name or os.getenv("WHISPER_MODEL", "small")
    model = _load_model(name)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
    )
    segments = [Segment(start=s.start, end=s.end, text=s.text.strip()) for s in segments_iter]
    text = " ".join(s.text for s in segments).strip()
    return Transcript(
        text=text,
        language=info.language,
        segments=segments,
        duration=info.duration,
    )


def merge_dialog(transcripts: dict[str, Transcript]) -> tuple[str, list[tuple[float, str, str]]]:
    """Слить транскрипты нескольких ролей в хронологический диалог.

    Возвращает (текст-диалог, список (start, role, text) для UI).
    """
    items: list[tuple[float, str, str]] = []
    for role, t in transcripts.items():
        for seg in t.segments:
            if seg.text:
                items.append((seg.start, role, seg.text))
    items.sort(key=lambda x: x[0])

    lines: list[str] = []
    prev_role: str | None = None
    buf: list[str] = []
    for _, role, text in items:
        if role != prev_role:
            if buf:
                lines.append(f"[{prev_role}] " + " ".join(buf))
                buf = []
            prev_role = role
        buf.append(text)
    if buf and prev_role is not None:
        lines.append(f"[{prev_role}] " + " ".join(buf))

    return "\n".join(lines), items
