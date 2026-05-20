# 🎙 Voice Notes AI
НА ДАННЫЙ МОМЕНТ ПРОЕКТ СЫРОВАТ
> Голосовые заметки + запись звонков → расшифровка → **глубокий AI-анализ**.

Работает с **любым** приложением для звонков (MAX, Telegram, Zoom, Discord) через системный аудио-захват WASAPI loopback.

[![GitHub](https://img.shields.io/badge/github-AndruLostVeber%2Fvoice--catcher-blue?logo=github)](https://github.com/AndruLostVeber/voice-catcher)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![NVIDIA](https://img.shields.io/badge/NVIDIA%20NIM-Llama%20%7C%20Nemotron%20%7C%20Mixtral%20%7C%20GPT--OSS-76b900)

## ✨ Что умеет

### 📝 Базовые сценарии
- 🎙 **Заметки** — запись с микрофона прямо в браузере
- 📞 **Звонки** — параллельный захват системного звука + микрофона с диаризацией **Я / Собеседник**
- 🎮 **Discord** — бот заходит в голосовой канал и пишет каждого участника отдельным треком (настоящая диаризация по людям, без дублирования)
- 📁 **Загрузка файлов** (wav/mp3/m4a/ogg/flac)

### 🧠 Анализ через NVIDIA NIM
- **Саммари:** TL;DR, ключевые мысли, темы, тональность
- **Action items с указанием «кто отвечает»**
- **Договорённости** и **открытые вопросы**
- 🔬 **Глубокий анализ звонка** (отдельный LLM-вызов):
  - Стиль общения каждого участника
  - Эмоциональный таймлайн с интенсивностью
  - Сигналы конфликта и согласия с цитатами
  - Качество коммуникации и баланс власти
  - Риски / недосказанности
  - Конкретные рекомендации следующих шагов
  - Интересные цитаты с обоснованием

### 📊 Статистика и визуализация
- 📊 **Локальная статистика разговора** (без LLM):
  - Слова, секунды и доля каждого говорящего
  - Темп речи (WPM)
  - Тишина, перекрытие, самая длинная пауза
  - Кто заговорил первым
- 📈 **Графики Altair:**
  - Pie chart баланса времени речи
  - Bar chart слов и WPM по сторонам
  - Line chart эмоций по фазам
  - Тренд сессий по дням и часам суток

### 💾 Сохранение и экспорт
- 💾 **Автосохранение** Markdown в `data/exports/` после каждой сессии
- ⬇️ **Кнопка скачать Markdown** для каждой сессии (включая глубокий анализ)
- 📦 **Bulk-экспорт всей истории в ZIP** (Markdown + оригинальный JSON)
- 📂 **Открыть папку экспортов** одной кнопкой
- 🔄 **Переоткрытие сессии** из истории — загружает всё в основной вид с графиками
- 🔄 **Переанализ другой моделью** — пересчитать саммари и анализ без перезаписи аудио

### 🔍 Поиск
- 🗂 **Текстовый поиск** по транскриптам, ключевым мыслям и темам
- 🧠 **Семантический поиск** через NVIDIA embeddings (`nv-embedqa-e5-v5`) — найти звонки по смыслу
- 🔽 Фильтры по типу (звонки / заметки) и сортировка

### 🔔 UX
- 🔔 **Windows-уведомления** + звук, когда обработка готова
- 🎚 **Выбор loopback-устройства и микрофона** прямо перед звонком
- 🌐 **Выбор языка распознавания** (auto, ru, en, uk, de, es)
- 🎨 Кастомная тёмная тема с градиентными акцентами и пульсирующим индикатором записи
- ⚡ **Параллельная обработка** двух дорожек и параллельные LLM-вызовы
- 👋 Welcome empty state с карточками сценариев

## 🧩 Стек

| Слой | Технология |
|------|-----------|
| ASR | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (локально, CPU/GPU, отличное качество русского) |
| LLM | [NVIDIA NIM API](https://build.nvidia.com): Llama 3.3 70B · Nemotron Super 49B · Llama 4 Maverick · Mixtral 8x22B · GPT-OSS 120B/20B |
| Loopback | [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) — WASAPI (только Windows) |
| Микрофон | sounddevice + scipy |
| UI | Streamlit + Altair + custom CSS |
| Уведомления | plyer + winsound |

## 🚀 Быстрый старт

```powershell
# 1. Зависимости (Python 3.10+, протестировано на 3.13)
pip install -r requirements.txt

# 2. Ключ NVIDIA в .env (получить на https://build.nvidia.com — 5000 кредитов бесплатно)
copy .env.example .env
# отредактируй .env: вставь свой NVIDIA_API_KEY

# 3. Запуск
streamlit run app.py
```

Откроется на http://localhost:8501.

## 🗂 Структура проекта

```
voice-assistant/
├── app.py                    # Streamlit UI (4 вкладки + сайдбар)
├── src/
│   ├── recorder.py           # запись с микрофона (sounddevice)
│   ├── call_recorder.py      # WASAPI loopback + микрофон параллельно (callback API)
│   ├── asr.py                # faster-whisper + merge диалогов по таймлайну
│   ├── summarizer.py         # NVIDIA LLM: summarize, summarize_dialog (два промпта)
│   ├── analyzer.py           # talk_stats (локально) + deep_analyze (LLM)
│   ├── exporter.py           # экспорт сессии в Markdown
│   ├── storage.py            # JSON-хранилище сессий
│   ├── notify.py             # Windows-уведомления (plyer + winsound)
│   └── theme.py              # кастомный CSS
├── data/
│   ├── recordings/           # сохранённые wav-файлы
│   ├── sessions/             # JSON история (источник правды)
│   └── exports/              # автоматически сохранённые .md
├── .streamlit/
│   ├── config.toml           # тёмная тема, primary color
│   └── credentials.toml      # пустой email (без онбординга)
├── .env                      # ключи (в git не коммитим!)
└── requirements.txt
```

## 🎮 Как подключить Discord-бота

1. Создай Application на [discord.com/developers/applications](https://discord.com/developers/applications)
2. Bot → Reset Token → скопируй
3. Включи **Privileged Intents**: `SERVER MEMBERS` и `VOICE STATE`
4. OAuth2 → URL Generator → scope `bot`, permissions `Connect`, `Speak`, `View Channels`
5. Открой сгенерированный URL и добавь бота на сервер
6. Вставь токен в `.env` (`DISCORD_BOT_TOKEN=...`) или прямо в UI на вкладке «🎮 Discord»
7. Нажми «🔌 Подключить бота» → выбери голосовой канал → «🔊 Слушать канал»
8. Когда разговор закончится → «■ Завершить и обработать»

Каждый говорящий получит отдельный `.wav`, далее они объединятся в multi-speaker диалог с реальными именами участников. Глубокий анализ адаптируется под произвольное число участников.

## 🔄 Как работает запись звонка

```
┌──────────────────────┐         ┌──────────────────────────┐
│  MAX / Zoom /        │  → ─→ →  │  WASAPI Loopback         │ → system.wav
│  Telegram / Discord  │         │  (выбираешь устройство)  │
└──────────────────────┘         └──────────────────────────┘
                                                                 ↘
┌──────────────────────┐         ┌──────────────────────────┐    │
│   Микрофон           │  → ─→ →  │   PyAudio Input          │ → mic.wav
│   (выбираешь)        │         │   (callback-API)         │    │
└──────────────────────┘         └──────────────────────────┘    │
                                                                 ↓
                                       Whisper × 2 параллельно
                                                                 ↓
                                  merge_dialog по таймлайну
                                                                 ↓
                              «[Я] ...        [Собеседник] ...»
                                                                 ↓
                             ┌───────────────────────┬─────────────────────┐
                             │  summarize_dialog     │   deep_analyze       │
                             │  (структура диалога)  │   (стиль, эмоции,    │
                             │                       │   риски, шаги)       │
                             └───────────────────────┴─────────────────────┘
                                       параллельные LLM вызовы
                                                                 ↓
                                  ✅ Markdown + 🔔 Уведомление
```

## 🧠 Модели

Меняются в сайдбаре. Все бесплатные на NVIDIA developer tier (~5000 кредитов).

**Whisper:**
- `tiny` / `base` — мгновенно, для коротких заметок
- `small` — баланс (по умолчанию)
- `medium` / `large-v3` — максимальное качество

**LLM (NVIDIA NIM):**
- `nvidia/llama-3.3-nemotron-super-49b-v1` — **по умолчанию**, баланс качества и скорости
- `meta/llama-3.3-70b-instruct` — отличный русский
- `meta/llama-4-maverick-17b-128e-instruct` — новейшая, MoE
- `mistralai/mixtral-8x22b-instruct-v0.1` — мультиязычный MoE
- `openai/gpt-oss-120b` — мощная для глубокого анализа
- `openai/gpt-oss-20b` — быстрая

## 🔧 Переменные окружения (`.env`)

```ini
NVIDIA_API_KEY=nvapi-...
LLM_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1
WHISPER_MODEL=small
# cpu (по умолчанию) или cuda — для cuda нужны CUDA 12 + cuBLAS DLL
WHISPER_DEVICE=cpu
```

## ⚠️ Важные замечания

- **Default Output в Windows** — звук собеседника захватывается с того устройства, которое стоит как Default Output. В UI можно выбрать конкретное устройство явно.
- **Первый запуск Whisper медленный** — модель качается (~500 МБ для `small`). Дальше кэш в `~/.cache/huggingface/`.
- **WASAPI loopback** работает только на Windows.
- **Лимиты NVIDIA:** ~5000 кредитов на free tier, ~40 запросов/мин на модель. Глубокий анализ тратит ещё 1 кредит за звонок.
- ⚖️ **Юридически:** запись разговоров без уведомления собеседника может нарушать ст. 137 УК РФ (тайна переговоров). Для **личных** заметок — норм. Для распространения / публикации — нет. **Всегда предупреждай собеседника.**

## 🛤 Roadmap

- [ ] MAX-бот: голосовое → саммари в чат
- [ ] Стриминговая транскрипция в реальном времени
- [ ] Семантический поиск по истории (NVIDIA embeddings)
- [ ] NVIDIA Parakeet (gRPC) как альтернатива Whisper
- [ ] Linux/macOS — захват через PulseAudio / ScreenCaptureKit
- [ ] Voice Activity Detection индикатор в realtime
- [ ] Шаринг конкретной сессии через короткую ссылку

## 📜 Лицензия

MIT.
