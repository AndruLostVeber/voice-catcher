from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.asr import merge_dialog, transcribe
from src.call_recorder import CallRecorder, get_loopback_info
from src.recorder import Recorder, list_input_devices
from src.storage import delete_session, list_sessions, save_session
from src.summarizer import summarize, summarize_dialog

RECORDINGS_DIR = Path(__file__).parent / "data" / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENT_EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😟"}
ROLE_COLORS = {"Я": "🟦", "Собеседник": "🟪"}


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

    llm_model = st.sidebar.selectbox(
        "LLM для саммари",
        options=[
            "meta/llama-3.3-70b-instruct",
            "nvidia/llama-3.1-nemotron-70b-instruct",
            "qwen/qwen2.5-7b-instruct",
        ],
        index=0,
    )
    st.session_state["llm_model"] = llm_model

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

    save_session(
        transcript_text=transcript.text,
        summary=summary.to_dict(),
        audio_path=str(audio_path),
        duration=transcript.duration,
    )


def process_call(system_path: Path, mic_path: Path, duration: float):
    with st.status("Обрабатываю звонок...", expanded=True) as status:
        st.write("🎧 Распознаю реплики собеседника (system)...")
        t0 = time.time()
        sys_transcript = transcribe(
            system_path,
            model_name=st.session_state.get("whisper_model"),
            language="ru",
        )
        st.write(f"  ✅ {time.time() - t0:.1f}с | {len(sys_transcript.text)} символов")

        st.write("🎙 Распознаю свои реплики (mic)...")
        t0 = time.time()
        mic_transcript = transcribe(
            mic_path,
            model_name=st.session_state.get("whisper_model"),
            language="ru",
        )
        st.write(f"  ✅ {time.time() - t0:.1f}с | {len(mic_transcript.text)} символов")

        st.write("🔀 Объединяю диалог по таймлайну...")
        dialog_text, items = merge_dialog({"Я": mic_transcript, "Собеседник": sys_transcript})

        st.write("🧠 Делаю саммари диалога...")
        t0 = time.time()
        summary = summarize_dialog(dialog_text, model=st.session_state.get("llm_model"))
        st.write(f"✅ Готово за {time.time() - t0:.1f}с")

        status.update(label="Готово", state="complete")

    st.session_state["dialog_text"] = dialog_text
    st.session_state["dialog_items"] = items
    st.session_state["dialog_summary"] = summary
    st.session_state["duration"] = duration
    st.session_state["summary"] = None

    save_session(
        transcript_text=dialog_text,
        summary=summary.to_dict(),
        audio_path=f"{system_path.name} + {mic_path.name}",
        duration=duration,
    )


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
    st.subheader("📞 Результат звонка")

    emoji = SENTIMENT_EMOJI.get(summary.sentiment, "")
    st.info(f"**TL;DR** {emoji}\n\n{summary.tldr}")

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
    st.subheader("📝 Результат")

    emoji = SENTIMENT_EMOJI.get(summary.sentiment, "")
    st.info(f"**TL;DR** {emoji}\n\n{summary.tldr}")

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


def render_history_tab():
    st.subheader("🗂 История сессий")
    sessions = list_sessions()
    if not sessions:
        st.caption("Пока пусто. Запиши или загрузи что-нибудь.")
        return

    for s in sessions:
        summary = s.get("summary", {})
        with st.expander(f"📌 {s['created_at']} — {summary.get('tldr', '(без саммари)')[:80]}"):
            st.caption(f"ID: `{s['id']}` | Длительность: {s.get('duration', 0):.1f}с")
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
            if st.button("🗑 Удалить", key=f"del_{s['id']}"):
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
