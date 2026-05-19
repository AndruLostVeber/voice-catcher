# 🎙 Voice Notes AI

Голосовой ассистент-транскрайбер: запись с микрофона → расшифровка → структурированное саммари.

## Стек

- **ASR:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (локально, CPU/GPU, отличное качество русского)
- **LLM:** NVIDIA NIM API — Llama 3.3 70B / Nemotron 70B (бесплатный тариф build.nvidia.com)
- **UI:** Streamlit
- **Audio:** sounddevice + scipy

## Быстрый старт

```powershell
# 1. Зависимости (Python 3.10+)
pip install -r requirements.txt

# 2. Ключ NVIDIA в .env (получить на https://build.nvidia.com)
copy .env.example .env
# отредактируй .env

# 3. Запуск
streamlit run app.py
```

Откроется на http://localhost:8501.

## Что умеет

- 🎙 Запись с микрофона прямо в браузере (старт/стоп)
- 📁 Загрузка готовых файлов (wav/mp3/m4a/ogg/flac)
- 📝 Транскрипт по сегментам с таймкодами
- 🧠 Саммари:
  - TL;DR (1-2 предложения)
  - Ключевые мысли
  - Список задач (action items)
  - Темы и тональность
- 🗂 История всех сессий (сохраняется в `data/sessions/*.json`)

## Структура

```
voice-assistant/
├── app.py                  # Streamlit UI
├── src/
│   ├── recorder.py         # запись с микрофона (sounddevice)
│   ├── asr.py              # faster-whisper обёртка
│   ├── summarizer.py       # NVIDIA LLM client (OpenAI-compat)
│   └── storage.py          # JSON-хранилище сессий
├── data/
│   ├── recordings/         # сохранённые wav-файлы
│   └── sessions/           # JSON истории
├── .env                    # ключи (в git не коммитим!)
└── requirements.txt
```

## Модели

В сайдбаре можно переключать:

**Whisper:**
- `tiny` / `base` — мгновенно, для коротких заметок
- `small` — баланс (по умолчанию)
- `medium` / `large-v3` — максимальное качество для длинных записей

**LLM (NVIDIA):**
- `meta/llama-3.3-70b-instruct` — лучшее качество для русского
- `nvidia/llama-3.1-nemotron-70b-instruct` — NVIDIA-тюн под инструкции
- `qwen/qwen2.5-7b-instruct` — быстрее, мультиязычная

## Заметки

- **Первый запуск Whisper медленный** — модель скачивается (~500 МБ для `small`).
- NVIDIA free tier: ~5000 кредитов, ~40 запросов/мин на модель.
- На Windows может потребоваться [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) для сборки `sounddevice`.

## TODO / идеи

- Стриминговое распознавание в реальном времени
- Замена ASR на NVIDIA Parakeet (gRPC) опционально
- Экспорт саммари в Markdown / Notion
- Диаризация (кто что сказал) через `pyannote.audio`
