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

    stats = session.get("talk_stats")
    if stats:
        lines.append("---")
        lines.append("")
        lines.append("## Статистика разговора")
        lines.append("")
        lines.append(
            f"- Запись: {stats.get('total_audio_seconds', 0):.0f}с  "
            f"· Речь: {stats.get('total_speech_seconds', 0):.0f}с  "
            f"· Тишина: {stats.get('silence_seconds', 0):.0f}с  "
            f"· Перекрытие: {stats.get('overlap_seconds', 0):.0f}с"
        )
        for sp in stats.get("speakers", []):
            lines.append(
                f"- **{sp.get('role', '?')}**: "
                f"{sp.get('word_count', 0)} слов, "
                f"{sp.get('seconds', 0):.1f}с ({sp.get('share', 0) * 100:.0f}%), "
                f"темп {sp.get('words_per_minute', 0):.0f} слов/мин"
            )
        if stats.get("first_speaker"):
            lines.append(
                f"- Первым заговорил: **{stats['first_speaker']}**, "
                f"самая длинная пауза: {stats.get('longest_pause_seconds', 0):.1f}с"
            )
        lines.append("")

    deep = session.get("deep_analysis")
    if deep:
        lines.append("---")
        lines.append("")
        lines.append("## Глубокий анализ")
        lines.append("")
        if deep.get("communication_quality"):
            lines.append(f"**Качество коммуникации:** {deep['communication_quality']}")
        if deep.get("power_balance"):
            lines.append(f"**Баланс:** {deep['power_balance']}")
        lines.append("")

        if deep.get("speaker_styles"):
            lines.append("### Стили общения")
            lines.append("")
            for role, desc in deep["speaker_styles"].items():
                lines.append(f"- **{role}**: {desc}")
            lines.append("")

        if deep.get("interesting_quotes"):
            lines.append("### Интересные цитаты")
            lines.append("")
            for q in deep["interesting_quotes"]:
                lines.append(
                    f"> **{q.get('role', '?')}**: «{q.get('quote', '')}»  "
                    f"\n> *— {q.get('reason', '')}*"
                )
                lines.append("")

        if deep.get("next_steps"):
            lines.append("### Следующие шаги")
            lines.append("")
            for s in deep["next_steps"]:
                lines.append(f"- {s}")
            lines.append("")

        if deep.get("risks"):
            lines.append("### Риски и недосказанности")
            lines.append("")
            for r in deep["risks"]:
                lines.append(f"- {r}")
            lines.append("")

        if deep.get("conflict_markers"):
            lines.append("### Сигналы напряжения")
            lines.append("")
            for m in deep["conflict_markers"]:
                lines.append(f"- {m.get('trigger', '')} — «{m.get('quote', '')}»")
            lines.append("")

        if deep.get("agreement_markers"):
            lines.append("### Сигналы согласия")
            lines.append("")
            for m in deep["agreement_markers"]:
                lines.append(f"- {m.get('about', '')} — «{m.get('quote', '')}»")
            lines.append("")

        if deep.get("emotion_timeline"):
            lines.append("### Эмоциональный таймлайн")
            lines.append("")
            for e in deep["emotion_timeline"]:
                bar = "█" * int(e.get("intensity", 1))
                lines.append(
                    f"- `{e.get('time_marker', '')}` "
                    f"**{e.get('role', '?')}** — {e.get('emotion', '')} {bar}"
                )
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
