from __future__ import annotations


def session_to_markdown(session: dict) -> str:
    s = session.get("summary", {}) or {}
    lines: list[str] = []

    title = s.get("tldr") or session.get("created_at", "Сессия")
    lines.append(f"# {title}")
    lines.append("")

    meta_parts = []
    if session.get("created_at"):
        meta_parts.append(f"**Дата:** {session['created_at']}")
    if session.get("duration"):
        meta_parts.append(f"**Длительность:** {session['duration']:.0f}с")
    if session.get("audio_path"):
        meta_parts.append(f"**Источник:** `{session['audio_path']}`")
    if meta_parts:
        lines.append(" · ".join(meta_parts))
        lines.append("")

    if s.get("tldr"):
        lines.append("## TL;DR")
        lines.append("")
        lines.append(s["tldr"])
        lines.append("")

    if s.get("key_points"):
        lines.append("## Ключевые мысли")
        lines.append("")
        for kp in s["key_points"]:
            lines.append(f"- {kp}")
        lines.append("")

    if s.get("action_items"):
        lines.append("## Задачи")
        lines.append("")
        for a in s["action_items"]:
            if isinstance(a, dict):
                who = a.get("who", "?")
                task = a.get("task", "")
                lines.append(f"- **{who}:** {task}")
            else:
                lines.append(f"- {a}")
        lines.append("")

    if s.get("decisions"):
        lines.append("## Договорённости")
        lines.append("")
        for d in s["decisions"]:
            lines.append(f"- {d}")
        lines.append("")

    if s.get("open_questions"):
        lines.append("## Открытые вопросы")
        lines.append("")
        for q in s["open_questions"]:
            lines.append(f"- {q}")
        lines.append("")

    if s.get("topics"):
        lines.append("**Темы:** " + ", ".join(s["topics"]))
        lines.append("")

    if s.get("sentiment"):
        lines.append(f"**Тональность:** {s['sentiment']}")
        lines.append("")

    transcript = session.get("transcript", "")
    if transcript:
        lines.append("---")
        lines.append("")
        lines.append("## Транскрипт")
        lines.append("")
        lines.append(transcript)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def filename_for_session(session: dict) -> str:
    sid = session.get("id", "session")
    return f"voice-notes_{sid}.md"
