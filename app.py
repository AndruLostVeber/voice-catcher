from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.asr import transcribe
from src.recorder import Recorder, list_input_devices
from src.storage import delete_session, list_sessions, save_session
from src.summarizer import summarize

RECORDINGS_DIR = Path(__file__).parent / "data" / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENT_EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😟"}


def init_state():
    st.session_state.setdefault("recorder", None)
    st.session_state.setdefault("is_recording", False)
    st.session_state.setdefault("last_audio_path", None)
    st.session_state.setdefault("transcript", None)
    st.session_state.setdefault("summary", None)
    st.session_state.setdefault("duration", None)


def render_sidebar():
    st.sidebar.title("⚙️ Настройки")

    whisper_model = st.sidebar.selectbox(
        "Модель распознавания",
        options=["tiny", "base", "small", "medium", "large-v3"],
        index=2,
        help="small — быстро и качественно для русского. large-v3 — точнее, но медленнее.",
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
        st.sidebar.caption(f"🎙 Микрофонов найдено: {len(devices)}")
        with st.sidebar.expander("Устройства ввода"):
            for d in devices:
                st.write(f"• {d['name']} ({d['channels']}ch)")


def process_audio(audio_path: Path):
    with st.status("Обрабатываю...", expanded=True) as status:
        st.write("🎧 Распознаю речь (faster-whisper)...")
        t0 = time.time()
        transcript = transcribe(
            audio_path,
            model_name=st.session_state.get("whisper_model"),
            language="ru",
        )
        st.write(f"✅ Распознано за {time.time() - t0:.1f}с | {len(transcript.text)} символов")

        st.write("🧠 Делаю саммари (NVIDIA " + st.session_state.get("llm_model", "llama") + ")...")
        t0 = time.time()
        summary = summarize(transcript.text, model=st.session_state.get("llm_model"))
        st.write(f"✅ Готово за {time.time() - t0:.1f}с")

        status.update(label="Готово", state="complete")

    st.session_state["transcript"] = transcript
    st.session_state["summary"] = summary
    st.session_state["duration"] = transcript.duration
    st.session_state["last_audio_path"] = str(audio_path)

    save_session(
        transcript_text=transcript.text,
        summary=summary.to_dict(),
        audio_path=str(audio_path),
        duration=transcript.duration,
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
            st.metric("⏱ Готов к записи", "—")


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


def render_result():
    if st.session_state["summary"] is None:
        return

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
        st.text_area("", transcript.text, height=200, label_visibility="collapsed")
        with st.expander("По сегментам"):
            for seg in transcript.segments:
                st.markdown(f"`[{seg.start:.1f}-{seg.end:.1f}]` {seg.text}")


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
                    st.markdown(f"- {a}")
            with st.expander("Транскрипт"):
                st.text(s.get("transcript", ""))
            if st.button("🗑 Удалить", key=f"del_{s['id']}"):
                delete_session(s["id"])
                st.rerun()


def main():
    st.set_page_config(page_title="Voice Notes AI", page_icon="🎙", layout="wide")
    init_state()

    st.title("🎙 Voice Notes AI")
    st.caption("Запись → транскрипт (faster-whisper) → саммари (NVIDIA LLM)")

    render_sidebar()

    tab_rec, tab_upl, tab_hist = st.tabs(["🎙 Запись", "📁 Загрузка", "🗂 История"])
    with tab_rec:
        render_record_tab()
    with tab_upl:
        render_upload_tab()
    with tab_hist:
        render_history_tab()

    render_result()


if __name__ == "__main__":
    main()
