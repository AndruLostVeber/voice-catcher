from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"


def _ensure_dir() -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR


def save_session(
    transcript_text: str,
    summary: dict,
    audio_path: str | None = None,
    duration: float | None = None,
) -> Path:
    _ensure_dir()
    ts = datetime.now()
    record = {
        "id": ts.strftime("%Y%m%d_%H%M%S"),
        "created_at": ts.isoformat(timespec="seconds"),
        "duration": duration,
        "audio_path": audio_path,
        "transcript": transcript_text,
        "summary": summary,
    }
    path = SESSIONS_DIR / f"{record['id']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_sessions() -> list[dict]:
    _ensure_dir()
    out: list[dict] = []
    for path in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def load_session(session_id: str) -> dict | None:
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def delete_session(session_id: str) -> bool:
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return False
    path.unlink()
    return True
