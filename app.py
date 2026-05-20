from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.analyzer import compute_talk_stats, deep_analyze
from src.asr import merge_dialog, transcribe
from src.call_recorder import CallRecorder, get_loopback_info
from src.exporter import filename_for_session, session_to_markdown
from src.notify import notify
from src.recorder import Recorder, list_input_devices
from src.storage import delete_session, list_sessions, save_session
from src.summarizer import summarize, summarize_dialog

RECORDINGS_DIR = Path(__file__).parent / "data" / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR = Path(__file__).parent / "data" / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENT_EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😟"}
ROLE_COLORS = {"Я": "🟦", "Собеседник": "🟪"}
ROLE_PALETTE = {"Я": "#4C9AFF", "Собеседник": "#A78BFA"}


def _autosave_session(session: dict) -> Path | None:
    if not session:
        return None
    try:
        md = session_to_markdown(session)
        out = EXPORTS_DIR / filename_for_session(session)
        out.write_text(md, encoding="utf-8")
        return out
    except Exception as e:
        st.warning(f"Не удалось автосохранить Markdown: {e}")
        return None


def init_state():
    st.session_state.setdefault("recorder", None)
    st.session_state.setdefault("is_recording", False)
    st.session_state.setdefault("call_recorder", None)
    st.session_state.setdefault("is_in_call", False)
    st.session_state.setdefault("transcript", None)
    st.session_state.setdefault("summary", None)
    st.session_state.setdefault("dialog_text", None)
    st.session_state.setdefault("dialog_items", None)
    st.session_state.setdefault("dialog_summary", None)
    st.session_state.setdefault("talk_stats", None)
    st.session_state.setdefault("deep_analysis", None)
    st.session_state.setdefault("enable_deep_analysis", True)
    st.session_state.setdefault("enable_notifications", True)
    st.session_state.setdefault("autosave_markdown", True)
    st.session_state.setdefault("duration", None)


def render_sidebar():
    st.sidebar.title("⚙️ Настройки")

    whisper_model = st.sidebar.selectbox(
        "Модель распознавания",
        options=["tiny", "base", "small", "medium", "large-v3"],
        index=2,
        help="small — быстро и качественно для русского.",
    )
    st.session_state["whisper_model"] = whisper_model

    llm_options = {
        "nvidia/llama-3.3-nemotron-super-49b-v1": "Nemotron Super 49B — баланс качества и скорости",
        "meta/llama-3.3-70b-instruct": "Llama 3.3 70B — отличный русский",
        "meta/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick — новейшая, MoE",
        "mistralai/mixtral-8x22b-instruct-v0.1": "Mixtral 8x22B — мультиязычный MoE",
        "openai/gpt-oss-120b": "GPT-OSS 120B — мощная для глубокого анализа",
        "openai/gpt-oss-20b": "GPT-OSS 20B — быстрая",
    }
    default_llm = os.getenv("LLM_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1")
    keys = list(llm_options.keys())
    default_idx = keys.index(default_llm) if default_llm in keys else 0
    llm_model = st.sidebar.selectbox(
        "LLM для саммари",
        options=keys,
        index=default_idx,
        format_func=lambda k: llm_options[k].split(" — ")[0],
        help="\n".join(f"**{k}** — {v}" for k, v in llm_options.items()),
    )
    st.session_state["llm_model"] = llm_model
    st.sidebar.caption(llm_options[llm_model].split(" — ", 1)[-1])

    st.session_state["enable_deep_analysis"] = st.sidebar.toggle(
        "🔬 Глубокий анализ звонков",
        value=st.session_state.get("enable_deep_analysis", True),
        help="Второй LLM-вызов: стили общения, эмоции, цитаты, рекомендации. Тратит ещё один кредит.",
    )
    st.session_state["enable_notifications"] = st.sidebar.toggle(
        "🔔 Уведомления Windows",
        value=st.session_state.get("enable_notifications", True),
        help="Звуковое и системное уведомление, когда обработка завершилась.",
    )
    st.session_state["autosave_markdown"] = st.sidebar.toggle(
        "💾 Автосохранение Markdown",
        value=st.session_state.get("autosave_markdown", True),
        help="После обработки автоматически сохранять .md в data/exports/.",
    )

    devices = list_input_devices()
    if devices:
        st.sidebar.caption(f"🎙 Микрофонов: {len(devices)}")

    loopback = get_loopback_info()
    if loopback:
        st.sidebar.caption(f"🔊 Loopback: {loopback['name']} ({loopback['rate']}Hz)")
    else:
        st.sidebar.warning("WASAPI loopback недоступен")


def process_audio(audio_path: Path):
    with st.status("Обрабатываю...", expanded=True) as status:
        st.write("🎧 Распознаю речь...")
        t0 = time.time()
        transcript = transcribe(
            audio_path,
            model_name=st.session_state.get("whisper_model"),
            language="ru",
        )
        st.write(f"✅ Распознано за {time.time() - t0:.1f}с | {len(transcript.text)} символов")

        st.write("🧠 Делаю саммари...")
        t0 = time.time()
        summary = summarize(transcript.text, model=st.session_state.get("llm_model"))
        st.write(f"✅ Готово за {time.time() - t0:.1f}с")

        status.update(label="Готово", state="complete")

    st.session_state["transcript"] = transcript
    st.session_state["summary"] = summary
    st.session_state["duration"] = transcript.duration
    st.session_state["dialog_summary"] = None

    st.session_state["last_session"] = save_session(
        transcript_text=transcript.text,
        summary=summary.to_dict(),
        audio_path=str(audio_path),
        duration=transcript.duration,
    )
    if st.session_state.get("autosave_markdown", True):
        md_path = _autosave_session(st.session_state["last_session"])
        if md_path:
            st.session_state["last_session"]["_md_path"] = str(md_path)
    if st.session_state.get("enable_notifications", True):
        notify("Заметка готова", summary.tldr or "Транскрипт и саммари готовы")


MIN_WAV_BYTES = 1024  # меньше — это пустой header или почти тишина


def _check_audio_file(path: Path, label: str) -> str | None:
    if not path.exists():
        return f"{label}: файл не создан"
    size = path.stat().st_size
    if size < MIN_WAV_BYTES:
        return f"{label}: пустая запись ({size} байт)"
    return None


def process_call(system_path: Path, mic_path: Path, duration: float):
    sys_err = _check_audio_file(system_path, "Системный звук")
    mic_err = _check_audio_file(mic_path, "Микрофон")
    if sys_err and mic_err:
        st.error(
            "Обе дорожки пустые. Возможные причины:\n\n"
            f"- {sys_err}\n- {mic_err}\n\n"
            "Проверь: (1) в Windows Sound Settings выбран правильный Default Output "
            "и через него реально играет звук; (2) Default Input — нужный микрофон не замьючен; "
            "(3) во время записи действительно был звук с обеих сторон."
        )
        return

    with st.status("Обрабатываю звонок...", expanded=True) as status:
        st.write("🎧 Распознаю реплики собеседника (system)...")
        if sys_err:
            st.write(f"  ⚠️ {sys_err}")
            sys_transcript = None
        else:
            t0 = time.time()
            sys_transcript = transcribe(
                system_path,
                model_name=st.session_state.get("whisper_model"),
                language="ru",
            )
            st.write(f"  ✅ {time.time() - t0:.1f}с | {len(sys_transcript.text)} символов")

        st.write("🎙 Распознаю свои реплики (mic)...")
        if mic_err:
            st.write(f"  ⚠️ {mic_err}")
            mic_transcript = None
        else:
            t0 = time.time()
            mic_transcript = transcribe(
                mic_path,
                model_name=st.session_state.get("whisper_model"),
                language="ru",
            )
            st.write(f"  ✅ {time.time() - t0:.1f}с | {len(mic_transcript.text)} символов")

        st.write("🔀 Объединяю диалог по таймлайну...")
        sources: dict = {}
        if mic_transcript and mic_transcript.text:
            sources["Я"] = mic_transcript
        if sys_transcript and sys_transcript.text:
            sources["Собеседник"] = sys_transcript

        if not sources:
            status.update(label="Пустой диалог", state="error")
            st.warning(
                "Whisper не распознал речь ни в одной дорожке. Файлы существуют, но в них тишина "
                "или неразборчивый звук. Проверь Default Output/Input в Windows."
            )
            return

        dialog_text, items = merge_dialog(sources)

        st.write("📊 Считаю статистику говорящих...")
        talk_stats = compute_talk_stats(sources)

        st.write("🧠 Делаю саммари диалога...")
        t0 = time.time()
        summary = summarize_dialog(dialog_text, model=st.session_state.get("llm_model"))
        st.write(f"  ✅ {time.time() - t0:.1f}с")

        deep = None
        if st.session_state.get("enable_deep_analysis", True):
            st.write("🔬 Глубокий анализ (стили, эмоции, рекомендации)...")
            t0 = time.time()
            try:
                deep = deep_analyze(dialog_text, model=st.session_state.get("llm_model"))
                st.write(f"  ✅ {time.time() - t0:.1f}с")
            except Exception as e:
                st.write(f"  ⚠️ глубокий анализ пропущен: {e}")

        status.update(label="Готово", state="complete")

    st.session_state["dialog_text"] = dialog_text
    st.session_state["dialog_items"] = items
    st.session_state["dialog_summary"] = summary
    st.session_state["talk_stats"] = talk_stats
    st.session_state["deep_analysis"] = deep
    st.session_state["duration"] = duration
    st.session_state["summary"] = None

    st.session_state["last_session"] = save_session(
        transcript_text=dialog_text,
        summary=summary.to_dict(),
        audio_path=f"{system_path.name} + {mic_path.name}",
        duration=duration,
        talk_stats=talk_stats.to_dict() if talk_stats else None,
        deep_analysis=deep.to_dict() if deep else None,
        kind="call",
    )
    if st.session_state.get("autosave_markdown", True):
        md_path = _autosave_session(st.session_state["last_session"])
        if md_path:
            st.session_state["last_session"]["_md_path"] = str(md_path)
    if st.session_state.get("enable_notifications", True):
        notify("Звонок проанализирован", summary.tldr or "Саммари и анализ готовы")


def render_record_tab():
    st.subheader("🎙 Запись с микрофона")
    col1, col2 = st.columns([1, 1])
    with col1:
        if not st.session_state["is_recording"]:
            if st.button("● Начать запись", type="primary", use_container_width=True):
                rec = Recorder()
                rec.start()
                st.session_state["recorder"] = rec
                st.session_state["is_recording"] = True
                st.rerun()
        else:
            if st.button("■ Остановить", type="secondary", use_container_width=True):
                rec: Recorder = st.session_state["recorder"]
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = RECORDINGS_DIR / f"rec_{ts}.wav"
                rec.stop(out_path)
                st.session_state["is_recording"] = False
                st.session_state["recorder"] = None
                st.success(f"Запись сохранена: {out_path.name}")
                process_audio(out_path)
                st.rerun()
    with col2:
        if st.session_state["is_recording"]:
            rec: Recorder = st.session_state["recorder"]
            st.metric("⏱ Длительность", f"{rec.duration:.0f}с")
            time.sleep(1)
            st.rerun()
        else:
            st.metric("⏱ Готов", "—")


def render_call_tab():
    st.subheader("📞 Запись звонка")
    st.caption(
        "Захватываю системный звук (собеседник) + микрофон (ты) одновременно. "
        "Работает с любым приложением: MAX, Telegram, Zoom, Discord."
    )

    loopback = get_loopback_info()
    if not loopback:
        st.error("WASAPI loopback не найден. Убедись, что в Windows выбрано устройство вывода по умолчанию.")
        return

    st.info(
        f"🔊 Будет записан системный звук с: **{loopback['name']}**\n\n"
        "⚠️ Предупреди собеседника о записи — это требование закона и хороший тон."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if not st.session_state["is_in_call"]:
            if st.button("📞 Начать запись звонка", type="primary", use_container_width=True):
                try:
                    cr = CallRecorder()
                    system_path, mic_path = cr.start(RECORDINGS_DIR)
                    st.session_state["call_recorder"] = cr
                    st.session_state["call_paths"] = (system_path, mic_path)
                    st.session_state["is_in_call"] = True
                    st.rerun()
                except RuntimeError as e:
                    st.error(str(e))
        else:
            if st.button("■ Завершить звонок", type="secondary", use_container_width=True):
                cr: CallRecorder = st.session_state["call_recorder"]
                duration = cr.stop()
                system_path, mic_path = st.session_state["call_paths"]
                st.session_state["is_in_call"] = False
                st.session_state["call_recorder"] = None
                if cr.errors:
                    for err in cr.errors:
                        st.warning(err)
                st.success(f"Звонок записан: {duration:.0f}с")
                process_call(system_path, mic_path, duration)
                st.rerun()
    with col2:
        if st.session_state["is_in_call"]:
            cr: CallRecorder = st.session_state["call_recorder"]
            st.metric("🔴 В разговоре", f"{cr.duration:.0f}с")
            time.sleep(1)
            st.rerun()
        else:
            st.metric("⏱ Готов", "—")


def render_upload_tab():
    st.subheader("📁 Загрузка аудиофайла")
    uploaded = st.file_uploader(
        "Выбери .wav / .mp3 / .m4a / .ogg",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
    )
    if uploaded is not None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RECORDINGS_DIR / f"upload_{ts}_{uploaded.name}"
        out_path.write_bytes(uploaded.getbuffer())
        st.audio(str(out_path))
        if st.button("🚀 Обработать", type="primary"):
            process_audio(out_path)
            st.rerun()


def render_dialog_result():
    summary = st.session_state["dialog_summary"]
    dialog_text = st.session_state["dialog_text"]
    items = st.session_state["dialog_items"]

    st.divider()
    header_col, dl_col = st.columns([3, 1])
    with header_col:
        st.subheader("📞 Результат звонка")
    with dl_col:
        sess = st.session_state.get("last_session")
        if sess:
            st.download_button(
                "⬇️ Markdown",
                data=session_to_markdown(sess),
                file_name=filename_for_session(sess),
                mime="text/markdown",
                use_container_width=True,
            )

    emoji = SENTIMENT_EMOJI.get(summary.sentiment, "")
    st.info(f"**TL;DR** {emoji}\n\n{summary.tldr}")

    call_paths = st.session_state.get("call_paths")
    if call_paths:
        system_path, mic_path = call_paths
        audio_cols = st.columns(2)
        with audio_cols[0]:
            st.caption("🟪 Собеседник (system)")
            if Path(system_path).exists():
                st.audio(str(system_path))
        with audio_cols[1]:
            st.caption("🟦 Я (микрофон)")
            if Path(mic_path).exists():
                st.audio(str(mic_path))

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**🔑 Ключевые мысли**")
        for kp in summary.key_points:
            st.markdown(f"- {kp}")

        if summary.decisions:
            st.markdown("**🤝 Договорённости**")
            for d in summary.decisions:
                st.markdown(f"- {d}")
    with cols[1]:
        st.markdown("**✅ Задачи**")
        if summary.action_items:
            for a in summary.action_items:
                who = a.get("who", "?")
                task = a.get("task", "")
                badge = "🟦" if who == "Я" else ("🟪" if who == "Собеседник" else "🟨")
                st.markdown(f"- {badge} **{who}:** {task}")
        else:
            st.caption("явных задач не обнаружено")

        if summary.open_questions:
            st.markdown("**❓ Открытые вопросы**")
            for q in summary.open_questions:
                st.markdown(f"- {q}")

    if summary.topics:
        st.markdown("**🏷 Темы:** " + " · ".join(f"`{t}`" for t in summary.topics))

    talk_stats = st.session_state.get("talk_stats")
    if talk_stats:
        st.divider()
        st.markdown("### 📊 Статистика разговора")
        metric_cols = st.columns(4)
        metric_cols[0].metric("⏱ Запись", f"{talk_stats.total_audio_seconds:.0f}с")
        metric_cols[1].metric("🗣 Речь", f"{talk_stats.total_speech_seconds:.0f}с")
        metric_cols[2].metric("🤫 Тишина", f"{talk_stats.silence_seconds:.0f}с")
        metric_cols[3].metric("🌀 Перекрытие", f"{talk_stats.overlap_seconds:.0f}с")

        chart_cols = st.columns([1, 1])
        with chart_cols[0]:
            df_time = pd.DataFrame(
                [{"Кто": sp.role, "секунды": sp.seconds} for sp in talk_stats.speakers]
            )
            if not df_time.empty:
                pie = (
                    alt.Chart(df_time)
                    .mark_arc(innerRadius=45)
                    .encode(
                        theta=alt.Theta("секунды:Q"),
                        color=alt.Color(
                            "Кто:N",
                            scale=alt.Scale(
                                domain=list(ROLE_PALETTE.keys()),
                                range=list(ROLE_PALETTE.values()),
                            ),
                            legend=alt.Legend(orient="bottom"),
                        ),
                        tooltip=["Кто", "секунды"],
                    )
                    .properties(height=180, title="Время речи")
                )
                st.altair_chart(pie, use_container_width=True)
        with chart_cols[1]:
            df_words = pd.DataFrame(
                [
                    {"Кто": sp.role, "Метрика": "Слова", "Значение": sp.word_count}
                    for sp in talk_stats.speakers
                ]
                + [
                    {
                        "Кто": sp.role,
                        "Метрика": "Темп (WPM)",
                        "Значение": round(sp.words_per_minute),
                    }
                    for sp in talk_stats.speakers
                ]
            )
            if not df_words.empty:
                bars = (
                    alt.Chart(df_words)
                    .mark_bar()
                    .encode(
                        x=alt.X("Кто:N", title=None),
                        y=alt.Y("Значение:Q"),
                        color=alt.Color(
                            "Кто:N",
                            scale=alt.Scale(
                                domain=list(ROLE_PALETTE.keys()),
                                range=list(ROLE_PALETTE.values()),
                            ),
                            legend=None,
                        ),
                        column=alt.Column("Метрика:N", header=alt.Header(title=None)),
                        tooltip=["Кто", "Метрика", "Значение"],
                    )
                    .properties(height=180)
                )
                st.altair_chart(bars, use_container_width=False)

        sp_cols = st.columns(len(talk_stats.speakers) or 1)
        for col, sp in zip(sp_cols, talk_stats.speakers):
            badge = ROLE_COLORS.get(sp.role, "⬜")
            with col:
                st.markdown(f"**{badge} {sp.role}**")
                st.markdown(
                    f"- Слов: **{sp.word_count}**\n"
                    f"- Время: **{sp.seconds:.1f}с** ({sp.share * 100:.0f}%)\n"
                    f"- Темп: **{sp.words_per_minute:.0f}** слов/мин\n"
                    f"- Реплик: {sp.segment_count}, средняя {sp.avg_segment_seconds:.1f}с"
                )
        if talk_stats.first_speaker:
            st.caption(
                f"Первым заговорил: **{talk_stats.first_speaker}** · "
                f"Самая длинная пауза: {talk_stats.longest_pause_seconds:.1f}с"
            )

    deep = st.session_state.get("deep_analysis")
    if deep:
        st.divider()
        st.markdown("### 🔬 Глубокий анализ")

        meta_cols = st.columns(2)
        with meta_cols[0]:
            if deep.communication_quality:
                st.markdown(f"**Качество коммуникации:** `{deep.communication_quality}`")
        with meta_cols[1]:
            if deep.power_balance:
                st.markdown(f"**Баланс:** `{deep.power_balance}`")

        if deep.speaker_styles:
            style_cols = st.columns(len(deep.speaker_styles) or 1)
            for col, (role, desc) in zip(style_cols, deep.speaker_styles.items()):
                badge = ROLE_COLORS.get(role, "⬜")
                with col:
                    st.markdown(f"**🎭 Стиль {badge} {role}**")
                    st.caption(desc)

        if deep.interesting_quotes:
            st.markdown("**💬 Интересные цитаты**")
            for q in deep.interesting_quotes:
                role = q.get("role", "?")
                badge = ROLE_COLORS.get(role, "⬜")
                quote = q.get("quote", "")
                reason = q.get("reason", "")
                st.markdown(f"> {badge} **{role}:** «{quote}»  \n*— {reason}*")

        adv_cols = st.columns(2)
        with adv_cols[0]:
            if deep.next_steps:
                st.markdown("**🎯 Следующие шаги**")
                for s in deep.next_steps:
                    st.markdown(f"- {s}")
        with adv_cols[1]:
            if deep.risks:
                st.markdown("**⚡ Риски / недосказанности**")
                for r in deep.risks:
                    st.markdown(f"- {r}")

        marker_cols = st.columns(2)
        with marker_cols[0]:
            if deep.conflict_markers:
                st.markdown("**⚠️ Сигналы напряжения**")
                for m in deep.conflict_markers:
                    st.markdown(f"- *{m.get('trigger', '')}*  \n  > «{m.get('quote', '')}»")
        with marker_cols[1]:
            if deep.agreement_markers:
                st.markdown("**✅ Сигналы согласия**")
                for m in deep.agreement_markers:
                    st.markdown(f"- *{m.get('about', '')}*  \n  > «{m.get('quote', '')}»")

        if deep.emotion_timeline:
            st.markdown("**😊 Эмоциональный таймлайн**")
            time_order = {"начало": 0, "середина": 1, "конец": 2}
            df_em = pd.DataFrame(
                [
                    {
                        "phase": e.get("time_marker", ""),
                        "phase_order": time_order.get(e.get("time_marker", ""), 99),
                        "role": e.get("role", "?"),
                        "intensity": int(e.get("intensity", 1) or 1),
                        "emotion": e.get("emotion", ""),
                    }
                    for e in deep.emotion_timeline
                    if e.get("role") in ROLE_PALETTE
                ]
            )
            if not df_em.empty:
                line = (
                    alt.Chart(df_em)
                    .mark_line(point=alt.OverlayMarkDef(size=120, filled=True))
                    .encode(
                        x=alt.X(
                            "phase:N",
                            sort=["начало", "середина", "конец"],
                            title="Фаза разговора",
                        ),
                        y=alt.Y(
                            "intensity:Q",
                            title="Интенсивность",
                            scale=alt.Scale(domain=[0, 5]),
                        ),
                        color=alt.Color(
                            "role:N",
                            scale=alt.Scale(
                                domain=list(ROLE_PALETTE.keys()),
                                range=list(ROLE_PALETTE.values()),
                            ),
                            legend=alt.Legend(orient="top", title=None),
                        ),
                        tooltip=["role", "phase", "emotion", "intensity"],
                    )
                    .properties(height=220)
                )
                st.altair_chart(line, use_container_width=True)
            with st.expander("Детали"):
                for e in deep.emotion_timeline:
                    role = e.get("role", "?")
                    badge = ROLE_COLORS.get(role, "⬜")
                    bar = "█" * int(e.get("intensity", 1) or 1)
                    st.markdown(
                        f"`{e.get('time_marker', '')}` {badge} **{role}** — "
                        f"{e.get('emotion', '')} {bar}"
                    )

    with st.expander("💬 Диалог (по сегментам)"):
        for start, role, text in items or []:
            badge = ROLE_COLORS.get(role, "⬜")
            st.markdown(f"{badge} `{start:.1f}с` **[{role}]** {text}")

    with st.expander("📄 Полный транскрипт"):
        st.text_area("dialog", dialog_text or "", height=240, label_visibility="collapsed")


def render_note_result():
    summary = st.session_state["summary"]
    transcript = st.session_state["transcript"]

    st.divider()
    header_col, dl_col = st.columns([3, 1])
    with header_col:
        st.subheader("📝 Результат")
    with dl_col:
        sess = st.session_state.get("last_session")
        if sess:
            st.download_button(
                "⬇️ Markdown",
                data=session_to_markdown(sess),
                file_name=filename_for_session(sess),
                mime="text/markdown",
                use_container_width=True,
            )

    emoji = SENTIMENT_EMOJI.get(summary.sentiment, "")
    st.info(f"**TL;DR** {emoji}\n\n{summary.tldr}")

    sess = st.session_state.get("last_session") or {}
    audio_path = sess.get("audio_path")
    if audio_path and Path(audio_path).exists():
        st.audio(audio_path)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**🔑 Ключевые мысли**")
        for kp in summary.key_points:
            st.markdown(f"- {kp}")
    with cols[1]:
        st.markdown("**✅ Задачи**")
        if summary.action_items:
            for a in summary.action_items:
                st.markdown(f"- {a}")
        else:
            st.caption("явных задач не обнаружено")

    if summary.topics:
        st.markdown("**🏷 Темы:** " + " · ".join(f"`{t}`" for t in summary.topics))

    with st.expander("📄 Полный транскрипт"):
        st.text_area("note", transcript.text, height=200, label_visibility="collapsed")
        with st.expander("По сегментам"):
            for seg in transcript.segments:
                st.markdown(f"`[{seg.start:.1f}-{seg.end:.1f}]` {seg.text}")


def render_result():
    if st.session_state.get("dialog_summary") is not None:
        render_dialog_result()
    elif st.session_state.get("summary") is not None:
        render_note_result()


def _session_matches(s: dict, query: str) -> bool:
    if not query:
        return True
    q = query.lower()
    haystack_parts = [
        s.get("transcript", "") or "",
        (s.get("summary", {}) or {}).get("tldr", "") or "",
    ]
    summary = s.get("summary", {}) or {}
    for kp in summary.get("key_points", []) or []:
        haystack_parts.append(str(kp))
    for a in summary.get("action_items", []) or []:
        if isinstance(a, dict):
            haystack_parts.append(str(a.get("task", "")))
        else:
            haystack_parts.append(str(a))
    for t in summary.get("topics", []) or []:
        haystack_parts.append(str(t))
    return q in "\n".join(haystack_parts).lower()


def render_history_tab():
    st.subheader("🗂 История сессий")
    sessions = list_sessions()
    if not sessions:
        st.caption("Пока пусто. Запиши или загрузи что-нибудь.")
        return

    total_dur = sum(float(s.get("duration") or 0) for s in sessions)
    kinds = [s.get("kind", "note") for s in sessions]
    n_calls = sum(1 for k in kinds if k == "call")
    n_notes = len(sessions) - n_calls

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Всего сессий", len(sessions))
    mcol2.metric("Звонков", n_calls)
    mcol3.metric("Заметок", n_notes)
    mcol4.metric("Общее время", f"{total_dur / 60:.0f} мин")

    fcol1, fcol2, fcol3 = st.columns([3, 1, 1])
    with fcol1:
        query = st.text_input(
            "🔍 Поиск по транскриптам, ключевым мыслям, темам",
            placeholder="например: бюджет, встреча, важно...",
            label_visibility="collapsed",
        )
    with fcol2:
        kind_filter = st.selectbox(
            "Тип",
            options=["Все", "Звонки", "Заметки"],
            label_visibility="collapsed",
        )
    with fcol3:
        sort_by = st.selectbox(
            "Сортировка",
            options=["Сначала новые", "Сначала старые", "По длительности"],
            label_visibility="collapsed",
        )

    filtered = sessions
    if kind_filter == "Звонки":
        filtered = [s for s in filtered if s.get("kind") == "call"]
    elif kind_filter == "Заметки":
        filtered = [s for s in filtered if s.get("kind", "note") == "note"]
    filtered = [s for s in filtered if _session_matches(s, query)]

    if sort_by == "Сначала старые":
        filtered.sort(key=lambda s: s.get("created_at", ""))
    elif sort_by == "По длительности":
        filtered.sort(key=lambda s: float(s.get("duration") or 0), reverse=True)
    else:
        filtered.sort(key=lambda s: s.get("created_at", ""), reverse=True)

    if not filtered:
        st.caption("Ничего не найдено.")
        return

    st.caption(f"Показано: {len(filtered)} из {len(sessions)}")

    for s in filtered:
        summary = s.get("summary", {})
        kind_badge = "📞" if s.get("kind") == "call" else "📝"
        with st.expander(f"{kind_badge} {s['created_at']} — {summary.get('tldr', '(без саммари)')[:80]}"):
            st.caption(f"ID: `{s['id']}` | Длительность: {s.get('duration', 0):.1f}с")

            audio_path = s.get("audio_path", "") or ""
            if " + " in audio_path:
                parts = [p.strip() for p in audio_path.split(" + ")]
                for label, fname in zip(("Собеседник", "Я"), parts):
                    p = RECORDINGS_DIR / fname
                    if p.exists():
                        st.caption(label)
                        st.audio(str(p))
            elif audio_path and Path(audio_path).exists():
                st.audio(audio_path)

            if summary.get("key_points"):
                st.markdown("**Ключевые мысли:**")
                for kp in summary["key_points"]:
                    st.markdown(f"- {kp}")
            if summary.get("action_items"):
                st.markdown("**Задачи:**")
                for a in summary["action_items"]:
                    if isinstance(a, dict):
                        st.markdown(f"- **{a.get('who', '?')}:** {a.get('task', '')}")
                    else:
                        st.markdown(f"- {a}")
            if summary.get("decisions"):
                st.markdown("**Договорённости:**")
                for d in summary["decisions"]:
                    st.markdown(f"- {d}")
            with st.expander("Транскрипт"):
                st.text(s.get("transcript", ""))
            col_dl, col_del = st.columns([1, 1])
            with col_dl:
                st.download_button(
                    "⬇️ Markdown",
                    data=session_to_markdown(s),
                    file_name=filename_for_session(s),
                    mime="text/markdown",
                    key=f"dl_{s['id']}",
                    use_container_width=True,
                )
            with col_del:
                if st.button("🗑 Удалить", key=f"del_{s['id']}", use_container_width=True):
                    delete_session(s["id"])
                    st.rerun()


def main():
    st.set_page_config(page_title="Voice Notes AI", page_icon="🎙", layout="wide")
    init_state()

    st.title("🎙 Voice Notes AI")
    st.caption("Заметки и звонки → транскрипт (Whisper) → саммари (NVIDIA LLM)")

    render_sidebar()

    tab_rec, tab_call, tab_upl, tab_hist = st.tabs(
        ["🎙 Заметка", "📞 Звонок", "📁 Загрузка", "🗂 История"]
    )
    with tab_rec:
        render_record_tab()
    with tab_call:
        render_call_tab()
    with tab_upl:
        render_upload_tab()
    with tab_hist:
        render_history_tab()

    render_result()


if __name__ == "__main__":
    main()
