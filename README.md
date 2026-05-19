# 🎙 Voice Notes AI

Голосовой ассистент: заметки + запись звонков → транскрипт → структурированное саммари.

Работает с **любым** приложением для звонков (MAX, Telegram, Zoom, Discord) через системный аудио-захват.

## Возможности

- 🎙 **Заметки** — запись с микрофона в браузере
- 📞 **Звонки** — параллельный захват системного звука и микрофона с диаризацией **Я / Собеседник**
- 📁 **Загрузка** готовых файлов (wav/mp3/m4a/ogg/flac)
- 🧠 **Саммари** через NVIDIA Llama 3.3 70B:
  - TL;DR
  - Ключевые мысли обеих сторон
  - Action items с указанием **кто отвечает**
  - Договорённости и открытые вопросы
  - Темы и тональность
- ⬇️ **Экспорт в Markdown** одной кнопкой
- 🗂 История всех сессий с поиском и удалением

## Стек

- **ASR:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — локально, CPU/GPU, отличное качество русского
- **LLM:** [NVIDIA NIM API](https://build.nvidia.com) — Llama 3.3 70B / Nemotron 70B (бесплатный тариф)
- **Системный звук:** [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) — WASAPI loopback (только Windows)
- **Микрофон:** sounddevice + scipy
- **UI:** Streamlit

## Быстрый старт

```powershell
# 1. Зависимости (Python 3.10+, протестировано на 3.13)
pip install -r requirements.txt

# 2. Ключ NVIDIA в .env (получить на https://build.nvidia.com)
copy .env.example .env
# отредактируй .env: вставь свой NVIDIA_API_KEY

# 3. Запуск
streamlit run app.py
```

Откроется на http://localhost:8501.

## Структура

```
voice-assistant/
├── app.py                    # Streamlit UI (4 вкладки)
├── src/
│   ├── recorder.py           # запись с микрофона (sounddevice)
│   ├── call_recorder.py      # WASAPI loopback + микрофон параллельно
│   ├── asr.py                # faster-whisper + merge диалогов
│   ├── summarizer.py         # NVIDIA LLM, два промпта: заметка / звонок
│   ├── exporter.py           # экспорт сессии в Markdown
│   └── storage.py            # JSON-хранилище сессий
├── data/
│   ├── recordings/           # сохранённые wav-файлы
│   └── sessions/             # JSON история
├── .env                      # ключи (в git не коммитим!)
└── requirements.txt
```

## Как работает запись звонка

```
┌─────────────────┐         ┌──────────────────┐
│  MAX / Zoom /   │  → ─→ →  │ WASAPI Loopback  │ → system.wav
│  Telegram / ... │         │ (default output) │
└─────────────────┘         └──────────────────┘
                                                 
┌─────────────────┐         ┌──────────────────┐
│   Микрофон      │  → ─→ →  │   PyAudio Input  │ → mic.wav
└─────────────────┘         └──────────────────┘
         ↓
   Whisper × 2
         ↓
   merge_dialog по таймлайну → "[Я] ... [Собеседник] ..."
         ↓
   NVIDIA Llama 3.3 70B → структурированное саммари
```

⚠️ В Windows должен быть выбран правильный **Default Output device** — именно с него идёт захват звука собеседника.

## Модели

Меняются в сайдбаре:

**Whisper:**
- `tiny` / `base` — мгновенно, для коротких заметок
- `small` — баланс (по умолчанию)
- `medium` / `large-v3` — максимальное качество

**LLM (NVIDIA):**
- `meta/llama-3.3-70b-instruct` — лучшее качество для русского
- `nvidia/llama-3.1-nemotron-70b-instruct` — NVIDIA-тюн под инструкции
- `qwen/qwen2.5-7b-instruct` — быстрее, мультиязычная

## Переменные окружения (`.env`)

```ini
NVIDIA_API_KEY=nvapi-...
LLM_MODEL=meta/llama-3.3-70b-instruct
WHISPER_MODEL=small
# cpu (по умолчанию) или cuda — для cuda нужны CUDA 12 + cuBLAS DLL
WHISPER_DEVICE=cpu
```

## Заметки

- **Первый запуск Whisper медленный** — модель качается (~500 МБ для `small`).
- NVIDIA free tier: ~5000 кредитов, ~40 запросов/мин на модель.
- WASAPI loopback работает только на Windows.
- ⚠️ Запись разговоров без уведомления собеседника может нарушать ст. 137 УК РФ. Для личных заметок — норм, для распространения — нет.

## TODO / идеи

- Стриминговая транскрипция в реальном времени
- Замена ASR на NVIDIA Parakeet (gRPC) опционально
- Семантический поиск по истории (NVIDIA embeddings)
- MAX-бот: отправил голосовое → получил саммари в чате
- Linux/macOS — захват через PulseAudio / ScreenCaptureKit
