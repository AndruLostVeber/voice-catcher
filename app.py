from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.analyzer import compute_talk_stats, deep_analyze
from src.asr import merge_dialog, transcribe
from src.call_recorder import (
    CallRecorder,
    get_loopback_info,
    list_loopback_devices,
    list_microphones,
)
from src.embeddings import cosine_similarity, embed, session_text_for_embedding
from src.exporter import filename_for_session, session_to_markdown
from src.notify import notify
from src.recorder import Recorder, list_input_devices
from src.storage import (
    delete_session,
    list_sessions,
    load_session,
    save_session,
    update_session,
)
from src.summarizer import summarize, summarize_dialog
from src.theme import inject as inject_css

RECORDINGS_DIR = Path(__file__).parent / "data" / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR = Path(__file__).parent / "data" / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENT_EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😟"}
ROLE_COLORS = {"Я": "🟦", "Собеседник": "🟪"}
ROLE_PALETTE = {"Я": "#4C9AFF", "Собеседник": "#A78BFA"}


def format_duration(seconds: float | int | None) -> str:
    s = int(seconds or 0)
    if s < 60:
        return f"{s}с"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m}:{sec:02d}"
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def _load_session_into_state(session_id: str) -> bool:
    sess = load_session(session_id)
    if not sess:
        return False
    from src.analyzer import DeepAnalysis, SpeakerStats, TalkStats
    from src.summarizer import DialogSummary, Summary

    st.session_state["last_session"] = sess
    st.session_state["duration"] = sess.get("duration")
    st.session_state["call_paths"] = None

    audio_path = sess.get("audio_path") or ""
    if sess.get("kind") == "call" and " + " in audio_path:
        parts = [p.strip() for p in audio_path.split(" + ")]
        if len(parts) == 2:
            st.session_state["call_paths"] = (
                str(RECORDINGS_DIR / parts[0]),
                str(RECORDINGS_DIR / parts[1]),
            )

    summary_dict = sess.get("summary", {}) or {}
    if sess.get("kind") == "call":
        st.session_state["summary"] = None
        st.session_state["transcript"] = None
        st.session_state["dialog_text"] = sess.get("transcript", "")
        st.session_state["dialog_items"] = None
        st.session_state["dialog_summary"] = DialogSummary(
            tldr=summary_dict.get("tldr", ""),
            key_points=summary_dict.get("key_points", []),
            action_items=summary_dict.get("action_items", []),
            decisions=summary_dict.get("decisions", []),
            open_questions=summary_dict.get("open_questions", []),
            topics=summary_dict.get("topics", []),
            sentiment=summary_dict.get("sentiment", "neutral"),
            language=summary_dict.get("language", "ru"),
            raw="",
        )
        ts = sess.get("talk_stats") or {}
        if ts:
            st.session_state["talk_stats"] = TalkStats(
                speakers=[SpeakerStats(**sp) for sp in ts.get("speakers", [])],
                total_audio_seconds=ts.get("total_audio_seconds", 0.0),
                total_speech_seconds=ts.get("total_speech_seconds", 0.0),
                silence_seconds=ts.get("silence_seconds", 0.0),
                overlap_seconds=ts.get("overlap_seconds", 0.0),
                first_speaker=ts.get("first_speaker"),
                longest_pause_seconds=ts.get("longest_pause_seconds", 0.0),
            )
        else:
            st.session_state["talk_stats"] = None
        deep = sess.get("deep_analysis") or {}
        if deep:
            st.session_state["deep_analysis"] = DeepAnalysis(
                speaker_styles=deep.get("speaker_styles", {}),
                emotion_timeline=deep.get("emotion_timeline", []),
                conflict_markers=deep.get("conflict_markers", []),
                agreement_markers=deep.get("agreement_markers", []),
                communication_quality=deep.get("communication_quality", ""),
                power_balance=deep.get("power_balance", ""),
                next_steps=deep.get("next_steps", []),
                interesting_quotes=deep.get("interesting_quotes", []),
                risks=deep.get("risks", []),
                raw="",
            )
        else:
            st.session_state["deep_analysis"] = None
    else:
        st.session_state["dialog_summary"] = None
        st.session_state["talk_stats"] = None
        st.session_state["deep_analysis"] = None
        st.session_state["summary"] = Summary(
            tldr=summary_dict.get("tldr", ""),
            key_points=summary_dict.get("key_points", []),
            action_items=summary_dict.get("action_items", []),
            topics=summary_dict.get("topics", []),
            sentiment=summary_dict.get("sentiment", "neutral"),
            language=summary_dict.get("language", "ru"),
            raw="",
        )
        from src.asr import Transcript

        st.session_state["transcript"] = Transcript(
            text=sess.get("transcript", ""),
            language=summary_dict.get("language", "ru"),
            segments=[],
            duration=float(sess.get("duration") or 0),
        )
    return True


def _reanalyze_session(session: dict) -> dict | None:
    """Пересчитать саммари (и глубокий анализ для звонков) текущей моделью.
    Сохраняет в storage и возвращает обновлённый dict."""
    transcript_text = session.get("transcript", "") or ""
    if not transcript_text.strip():
        st.warning("Транскрипт пустой — нечего переанализировать.")
        return None

    model = st.session_state.get("llm_model")
    do_deep = (
        session.get("kind") == "call"
        and st.session_state.get("enable_deep_analysis", True)
    )

    with st.status("Переанализирую...", expanded=True) as status:
        st.write(f"🧠 LLM: `{model}`")
        try:
            with ThreadPoolExecutor(max_workers=2) as ex:
                if session.get("kind") == "call":
                    f_sum = ex.submit(summarize_dialog, transcript_text, model)
                else:
                    f_sum = ex.submit(summarize, transcript_text, model)
                f_deep = ex.submit(deep_analyze, transcript_text, model) if do_deep else None
                new_summary = f_sum.result()
                new_deep = f_deep.result() if f_deep else None
        except Exception as e:
            status.update(label="Ошибка", state="error")
            st.error(f"Не удалось переанализировать: {e}")
            return None

        updates = {"summary": new_summary.to_dict()}
        if new_deep is not None:
            updates["deep_analysis"] = new_deep.to_dict()
        updated = update_session(session["id"], updates)
        status.update(label="Готово", state="complete")

    if updated and st.session_state.get("autosave_markdown", True):
        _autosave_session(updated)
    if st.session_state.get("enable_notifications", True):
        notify("Переанализ готов", new_summary.tldr or "Сессия обновлена")
    return updated


def _open_folder(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return True
    except Exception:
        return False


def _bulk_zip_bytes(sessions: list[dict]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for s in sessions:
            try:
                md = session_to_markdown(s)
                zf.writestr(filename_for_session(s), md)
            except Exception:
                pass
            try:
                payload = {k: v for k, v in s.items() if not k.startswith("_")}
                zf.writestr(
                    f"json/{s.get('id', 'session')}.json",
                    json.dumps(payload, ensure_ascii=False, indent=2),
                )
            except Exception:
                pass
    return buf.getvalue()


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

    lang_options = {
        "ru": "🇷🇺 Русский",
        "auto": "🌐 Авто (Whisper определит сам)",
        "en": "🇬🇧 English",
        "uk": "🇺🇦 Українська",
        "de": "🇩🇪 Deutsch",
        "es": "🇪🇸 Español",
    }
    lang = st.sidebar.selectbox(
        "Язык распознавания",
        options=list(lang_options.keys()),
        index=0,
        format_func=lambda k: lang_options[k],
        help="Auto — Whisper сам определит, но иногда ошибается. Явный язык точнее.",
    )
    st.session_state["whisper_language"] = None if lang == "auto" else lang

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
            language=st.session_state.get("whisper_language", "ru"),
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
        st.write("🎧🎙 Распознаю обе дорожки параллельно...")
        t0 = time.time()
        whisper_model = st.session_state.get("whisper_model")

        whisper_lang = st.session_state.get("whisper_language", "ru")

        def _safe_transcribe(path: Path, err: str | None):
            if err:
                return None
            return transcribe(path, model_name=whisper_model, language=whisper_lang)

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_sys = ex.submit(_safe_transcribe, system_path, sys_err)
            f_mic = ex.submit(_safe_transcribe, mic_path, mic_err)
            sys_transcript = f_sys.result()
            mic_transcript = f_mic.result()

        sys_len = len(sys_transcript.text) if sys_transcript else 0
        mic_len = len(mic_transcript.text) if mic_transcript else 0
        st.write(
            f"  ✅ {time.time() - t0:.1f}с | system: {sys_len} симв., mic: {mic_len} симв."
        )
        if sys_err:
            st.write(f"  ⚠️ system: {sys_err}")
        if mic_err:
            st.write(f"  ⚠️ mic: {mic_err}")

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

        do_deep = st.session_state.get("enable_deep_analysis", True)
        if do_deep:
            st.write("🧠 Саммари + 🔬 глубокий анализ параллельно...")
        else:
            st.write("🧠 Делаю саммари диалога...")
        t0 = time.time()
        deep = None
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_summary = ex.submit(
                summarize_dialog, dialog_text, st.session_state.get("llm_model")
            )
            f_deep = (
                ex.submit(deep_analyze, dialog_text, st.session_state.get("llm_model"))
                if do_deep
                else None
            )
            summary = f_summary.result()
            if f_deep is not None:
                try:
                    deep = f_deep.result()
                except Exception as e:
                    st.write(f"  ⚠️ глубокий анализ пропущен: {e}")
        st.write(f"  ✅ {time.time() - t0:.1f}с")

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
                duration_rec = rec.duration
                st.session_state["is_recording"] = False
                st.session_state["recorder"] = None
                st.success(f"Запись сохранена: {out_path.name} · {format_duration(duration_rec)}")
                process_audio(out_path)
                st.rerun()
    with col2:
        if st.session_state["is_recording"]:
            rec: Recorder = st.session_state["recorder"]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;">'
                f'<span class="vc-recording-dot"></span>'
                f'<span style="font-size:1.1rem;font-weight:600;">Идёт запись · {format_duration(rec.duration)}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
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

    loopbacks = list_loopback_devices()
    mics = list_microphones()
    default_loopback = get_loopback_info()

    if not loopbacks:
        st.error("WASAPI loopback устройства не найдены.")
        return

    with st.expander("🎚 Аудиоустройства", expanded=not st.session_state.get("is_in_call")):
        lb_names = [f"[{d['index']}] {d['name']} · {d['rate']}Hz · {d['channels']}ch" for d in loopbacks]
        default_lb_idx = 0
        if default_loopback:
            for i, d in enumerate(loopbacks):
                if d["index"] == default_loopback["index"]:
                    default_lb_idx = i
                    break
        lb_choice = st.selectbox(
            "🔊 Системный звук (через что слышит собеседник)",
            options=list(range(len(loopbacks))),
            index=default_lb_idx,
            format_func=lambda i: lb_names[i],
            help="Это устройство загружено как Default Output в Windows. Звук, играющий через него, и будет записан.",
        )
        st.session_state["chosen_loopback"] = loopbacks[lb_choice]["index"]

        mic_names = [f"[{d['index']}] {d['name']} · {d['rate']}Hz · {d['channels']}ch" for d in mics]
        mic_choice = st.selectbox(
            "🎙 Микрофон",
            options=list(range(len(mics))) if mics else [0],
            format_func=lambda i: mic_names[i] if mics else "—",
            help="Запись твоего голоса.",
        )
        st.session_state["chosen_mic"] = mics[mic_choice]["index"] if mics else None

    st.info(
        f"🔊 Системный звук: **{loopbacks[lb_choice]['name']}** · "
        f"🎙 Микрофон: **{mics[mic_choice]['name'] if mics else '?'}**\n\n"
        "⚠️ Предупреди собеседника о записи — это требование закона и хороший тон."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if not st.session_state["is_in_call"]:
            if st.button("📞 Начать запись звонка", type="primary", use_container_width=True):
                try:
                    cr = CallRecorder()
                    system_path, mic_path = cr.start(
                        RECORDINGS_DIR,
                        loopback_index=st.session_state.get("chosen_loopback"),
                        mic_index=st.session_state.get("chosen_mic"),
                    )
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
                st.success(f"Звонок записан: {format_duration(duration)}")
                process_call(system_path, mic_path, duration)
                st.rerun()
    with col2:
        if st.session_state["is_in_call"]:
            cr: CallRecorder = st.session_state["call_recorder"]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;">'
                f'<span class="vc-recording-dot"></span>'
                f'<span style="font-size:1.1rem;font-weight:600;">В разговоре · {format_duration(cr.duration)}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
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
    header_col, btn1_col, btn2_col = st.columns([3, 1, 1])
    with header_col:
        st.subheader("📞 Результат звонка")
    with btn1_col:
        sess = st.session_state.get("last_session")
        if sess:
            st.download_button(
                "⬇️ Markdown",
                data=session_to_markdown(sess),
                file_name=filename_for_session(sess),
                mime="text/markdown",
                use_container_width=True,
            )
    with btn2_col:
        if sess and sess.get("id"):
            if st.button(
                "🔄 Переанализ",
                key="reanalyze_dialog",
                use_container_width=True,
                help="Пересчитать саммари и анализ текущей моделью",
            ):
                upd = _reanalyze_session(sess)
                if upd and _load_session_into_state(upd["id"]):
                    st.toast("Сессия переанализирована", icon="✅")
                    st.rerun()

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
        metric_cols[0].metric("⏱ Запись", format_duration(talk_stats.total_audio_seconds))
        metric_cols[1].metric("🗣 Речь", format_duration(talk_stats.total_speech_seconds))
        metric_cols[2].metric("🤫 Тишина", format_duration(talk_stats.silence_seconds))
        metric_cols[3].metric("🌀 Перекрытие", format_duration(talk_stats.overlap_seconds))

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
    header_col, btn1_col, btn2_col = st.columns([3, 1, 1])
    with header_col:
        st.subheader("📝 Результат")
    with btn1_col:
        sess = st.session_state.get("last_session")
        if sess:
            st.download_button(
                "⬇️ Markdown",
                data=session_to_markdown(sess),
                file_name=filename_for_session(sess),
                mime="text/markdown",
                use_container_width=True,
            )
    with btn2_col:
        if sess and sess.get("id"):
            if st.button(
                "🔄 Переанализ",
                key="reanalyze_note",
                use_container_width=True,
                help="Пересчитать саммари текущей моделью",
            ):
                upd = _reanalyze_session(sess)
                if upd and _load_session_into_state(upd["id"]):
                    st.toast("Заметка переанализирована", icon="✅")
                    st.rerun()

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


def _get_session_embedding(session: dict) -> "np.ndarray | None":  # type: ignore[name-defined]
    cache = st.session_state.setdefault("_embed_cache", {})
    sid = session.get("id")
    if sid in cache:
        return cache[sid]
    text = session_text_for_embedding(session)
    if not text.strip():
        cache[sid] = None
        return None
    try:
        vec = embed(text, input_type="passage")
        cache[sid] = vec
        return vec
    except Exception as e:
        print(f"[embed] {sid}: {e}")
        cache[sid] = None
        return None


def _semantic_search(sessions: list[dict], query: str, top_k: int = 10) -> list[tuple[float, dict]]:
    try:
        qv = embed(query, input_type="query")
    except Exception as e:
        st.warning(f"Не удалось получить embedding запроса: {e}")
        return [(0.0, s) for s in sessions]
    scored: list[tuple[float, dict]] = []
    with st.spinner(f"Семантический поиск по {len(sessions)} сессиям..."):
        for s in sessions:
            sv = _get_session_embedding(s)
            if sv is None:
                continue
            scored.append((cosine_similarity(qv, sv), s))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:top_k]


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
        st.markdown("### 👋 Здесь пока пусто")
        st.caption("Начни с одного из сценариев — результат появится в этой вкладке.")
        cards = st.columns(3)
        with cards[0]:
            st.markdown(
                """
                <div style="background:#161A22;padding:18px;border-radius:14px;border:1px solid #232a36;height:100%;">
                <h4 style="margin:0 0 6px;">🎙 Голосовая заметка</h4>
                <p style="color:#9CA3AF;font-size:0.9rem;margin:0;">
                Вкладка <b>«Заметка»</b> → нажми ● Начать запись → говори → ■ Остановить.
                Получишь TL;DR, ключевые мысли и задачи.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cards[1]:
            st.markdown(
                """
                <div style="background:#161A22;padding:18px;border-radius:14px;border:1px solid #232a36;height:100%;">
                <h4 style="margin:0 0 6px;">📞 Запись звонка</h4>
                <p style="color:#9CA3AF;font-size:0.9rem;margin:0;">
                Вкладка <b>«Звонок»</b> → выбери Default Output (наушники) → запусти разговор в MAX/Zoom/...
                Получишь диалог Я/Собеседник + глубокий анализ.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cards[2]:
            st.markdown(
                """
                <div style="background:#161A22;padding:18px;border-radius:14px;border:1px solid #232a36;height:100%;">
                <h4 style="margin:0 0 6px;">📁 Загрузка файла</h4>
                <p style="color:#9CA3AF;font-size:0.9rem;margin:0;">
                Вкладка <b>«Загрузка»</b> → закинь wav/mp3/m4a → «Обработать».
                Подходит для записей, сделанных вне приложения.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    total_dur = sum(float(s.get("duration") or 0) for s in sessions)
    kinds = [s.get("kind", "note") for s in sessions]
    n_calls = sum(1 for k in kinds if k == "call")
    n_notes = len(sessions) - n_calls

    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns([1, 1, 1, 1, 1.4])
    mcol1.metric("Всего сессий", len(sessions))
    mcol2.metric("Звонков", n_calls)
    mcol3.metric("Заметок", n_notes)
    mcol4.metric("Общее время", f"{total_dur / 60:.0f} мин")
    with mcol5:
        st.write("")
        st.download_button(
            "📦 Скачать всё (ZIP)",
            data=_bulk_zip_bytes(sessions),
            file_name=f"voice-notes_history_{datetime.now().strftime('%Y%m%d')}.zip",
            mime="application/zip",
            use_container_width=True,
            help="Markdown и JSON всех сессий в одном архиве",
        )
        if st.button(
            "📂 Открыть папку экспортов",
            use_container_width=True,
            help="Папка data/exports с .md файлами всех сессий",
        ):
            _open_folder(EXPORTS_DIR)

    with st.expander("📈 Аналитика истории", expanded=len(sessions) >= 3):
        df_h = pd.DataFrame(
            [
                {
                    "date": (s.get("created_at") or "")[:10],
                    "hour": int((s.get("created_at") or "T00")[11:13] or 0)
                    if len(s.get("created_at") or "") >= 13
                    else 0,
                    "duration_min": float(s.get("duration") or 0) / 60,
                    "kind": "Звонок" if s.get("kind") == "call" else "Заметка",
                }
                for s in sessions
            ]
        )
        if not df_h.empty:
            acols = st.columns(2)
            with acols[0]:
                df_daily = df_h.groupby(["date", "kind"], as_index=False)["duration_min"].sum()
                bar_daily = (
                    alt.Chart(df_daily)
                    .mark_bar()
                    .encode(
                        x=alt.X("date:O", title="Дата"),
                        y=alt.Y("duration_min:Q", title="Минут"),
                        color=alt.Color(
                            "kind:N",
                            scale=alt.Scale(
                                domain=["Звонок", "Заметка"],
                                range=["#A78BFA", "#4C9AFF"],
                            ),
                            legend=alt.Legend(orient="top", title=None),
                        ),
                        tooltip=["date", "kind", "duration_min"],
                    )
                    .properties(height=220, title="Минут по дням")
                )
                st.altair_chart(bar_daily, use_container_width=True)
            with acols[1]:
                df_kind = df_h.groupby("kind", as_index=False).size()
                pie_kind = (
                    alt.Chart(df_kind)
                    .mark_arc(innerRadius=45)
                    .encode(
                        theta="size:Q",
                        color=alt.Color(
                            "kind:N",
                            scale=alt.Scale(
                                domain=["Звонок", "Заметка"],
                                range=["#A78BFA", "#4C9AFF"],
                            ),
                            legend=alt.Legend(orient="bottom", title=None),
                        ),
                        tooltip=["kind", "size"],
                    )
                    .properties(height=220, title="Тип сессии")
                )
                st.altair_chart(pie_kind, use_container_width=True)

            df_hour = df_h.groupby("hour", as_index=False).size().rename(columns={"size": "count"})
            hour_chart = (
                alt.Chart(df_hour)
                .mark_area(
                    line={"color": "#4C9AFF"},
                    color=alt.Gradient(
                        gradient="linear",
                        stops=[
                            alt.GradientStop(color="#4C9AFF", offset=0),
                            alt.GradientStop(color="#A78BFA", offset=1),
                        ],
                        x1=1, x2=1, y1=1, y2=0,
                    ),
                    opacity=0.5,
                )
                .encode(
                    x=alt.X("hour:Q", title="Час суток", scale=alt.Scale(domain=[0, 23])),
                    y=alt.Y("count:Q", title="Сессий"),
                    tooltip=["hour", "count"],
                )
                .properties(height=160, title="Когда обычно записываешь")
            )
            st.altair_chart(hour_chart, use_container_width=True)

    fcol1, fcol2, fcol3, fcol4 = st.columns([3, 1, 1, 1])
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
    with fcol4:
        semantic = st.toggle(
            "🧠 Семантически",
            value=False,
            help="Использовать NVIDIA embeddings для поиска по смыслу, а не подстроке.",
        )

    filtered = sessions
    if kind_filter == "Звонки":
        filtered = [s for s in filtered if s.get("kind") == "call"]
    elif kind_filter == "Заметки":
        filtered = [s for s in filtered if s.get("kind", "note") == "note"]

    scored: list[tuple[float, dict]] | None = None
    if query and semantic:
        scored = _semantic_search(filtered, query, top_k=30)
        filtered = [s for _, s in scored]
    elif query:
        filtered = [s for s in filtered if _session_matches(s, query)]

    if not scored:
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

    score_map: dict[str, float] = {}
    if scored:
        for sc, s in scored:
            sid = s.get("id")
            if sid:
                score_map[sid] = sc

    for s in filtered:
        summary = s.get("summary", {})
        kind_badge = "📞" if s.get("kind") == "call" else "📝"
        score_str = ""
        if s.get("id") in score_map:
            score_str = f" · 🧠 {score_map[s['id']]:.2f}"
        with st.expander(
            f"{kind_badge} {s['created_at']}{score_str} — {summary.get('tldr', '(без саммари)')[:80]}"
        ):
            st.caption(f"ID: `{s['id']}` | Длительность: {format_duration(s.get('duration') or 0)}")

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
            btn_cols = st.columns([1, 1, 1])
            with btn_cols[0]:
                if st.button("🔄 Открыть", key=f"open_{s['id']}", use_container_width=True):
                    if _load_session_into_state(s["id"]):
                        st.toast(f"Сессия {s['id']} загружена", icon="✅")
                        st.rerun()
            with btn_cols[1]:
                st.download_button(
                    "⬇️ Markdown",
                    data=session_to_markdown(s),
                    file_name=filename_for_session(s),
                    mime="text/markdown",
                    key=f"dl_{s['id']}",
                    use_container_width=True,
                )
            with btn_cols[2]:
                if st.button("🗑 Удалить", key=f"del_{s['id']}", use_container_width=True):
                    delete_session(s["id"])
                    st.rerun()


def main():
    st.set_page_config(
        page_title="Voice Notes AI",
        page_icon="🎙",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/AndruLostVeber/voice-catcher",
            "Report a bug": "https://github.com/AndruLostVeber/voice-catcher/issues",
            "About": (
                "**Voice Notes AI** — голосовые заметки и анализ звонков.\n\n"
                "Whisper + NVIDIA NIM (Llama / Nemotron / Mixtral / GPT-OSS).\n\n"
                "GitHub: AndruLostVeber/voice-catcher"
            ),
        },
    )
    inject_css(st)
    init_state()

    title_col, badge_col = st.columns([5, 1])
    with title_col:
        st.title("🎙 Voice Notes AI")
        st.caption("Заметки и звонки → транскрипт (Whisper) → саммари и глубокий анализ (NVIDIA LLM)")
    with badge_col:
        st.markdown("")  # пустота — позже сюда счётчик/статус

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
