"""Discord-бот для записи голосового канала с per-user диаризацией.

Использует py-cord 2.x с WaveSink. Каждый говорящий → отдельный WAV.
Запускается в фоновом потоке со своим asyncio event loop, чтобы не
конфликтовать с синхронным Streamlit.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import discord
from discord.sinks import WaveSink


@dataclass
class VoiceChannelInfo:
    guild_id: int
    guild_name: str
    channel_id: int
    channel_name: str
    members: list[str]


@dataclass
class DiscordRecording:
    files: dict[str, Path]  # display_name -> wav path
    channel: str
    guild: str
    started_at: float
    duration: float


class DiscordBot:
    """Singleton-обёртка над py-cord ботом с управлением из Streamlit."""

    _instance: "DiscordBot | None" = None

    @classmethod
    def get(cls) -> "DiscordBot":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._bot: discord.Bot | None = None
        self._ready_event = threading.Event()
        self._ready_error: str | None = None
        self._current_vc: discord.VoiceClient | None = None
        self._current_channel: discord.VoiceChannel | None = None
        self._sink: WaveSink | None = None
        self._record_started_at: float | None = None
        self._last_recording: DiscordRecording | None = None
        self._record_done = threading.Event()

    # ---- lifecycle ----

    @property
    def is_connected(self) -> bool:
        return (
            self._bot is not None
            and not self._bot.is_closed()
            and self._ready_event.is_set()
            and self._ready_error is None
        )

    @property
    def is_recording(self) -> bool:
        return self._current_vc is not None and self._sink is not None

    @property
    def status(self) -> str:
        if not self._thread or not self._thread.is_alive():
            return "offline"
        if self._ready_error:
            return f"error: {self._ready_error}"
        if not self._ready_event.is_set():
            return "connecting"
        if self.is_recording:
            return "recording"
        return "ready"

    def start(self, token: str, timeout: float = 15.0) -> None:
        if self.is_connected:
            return
        self._ready_event.clear()
        self._ready_error = None

        intents = discord.Intents.default()
        intents.voice_states = True
        intents.guilds = True
        intents.members = True
        bot = discord.Bot(intents=intents)
        self._bot = bot

        @bot.event
        async def on_ready():
            self._loop = asyncio.get_running_loop()
            self._ready_event.set()

        @bot.event
        async def on_error(event, *args, **kwargs):
            self._ready_error = f"event error in {event}"

        def _run():
            try:
                asyncio.run(bot.start(token))
            except discord.LoginFailure as e:
                self._ready_error = f"login failure: {e}"
                self._ready_event.set()
            except Exception as e:
                self._ready_error = str(e)
                self._ready_event.set()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=timeout):
            self._ready_error = "timeout while connecting"

    def stop(self) -> None:
        if self._bot and self._loop and not self._bot.is_closed():
            try:
                fut = asyncio.run_coroutine_threadsafe(self._bot.close(), self._loop)
                fut.result(timeout=5)
            except Exception:
                pass
        self._bot = None
        self._loop = None
        self._ready_event.clear()
        self._current_vc = None
        self._sink = None

    # ---- discoverability ----

    def voice_channels(self) -> list[VoiceChannelInfo]:
        if not self.is_connected or self._bot is None:
            return []
        out: list[VoiceChannelInfo] = []
        for g in self._bot.guilds:
            for ch in g.voice_channels:
                out.append(
                    VoiceChannelInfo(
                        guild_id=g.id,
                        guild_name=g.name,
                        channel_id=ch.id,
                        channel_name=ch.name,
                        members=[m.display_name for m in ch.members],
                    )
                )
        return out

    # ---- recording ----

    async def _on_record_finished(self, sink: WaveSink, *, out_dir: Path, channel_name: str):
        files: dict[str, Path] = {}
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for user_id, audio in sink.audio_data.items():
            user = self._bot.get_user(user_id) if self._bot else None
            name = (user.display_name if user else f"user_{user_id}").replace("/", "_")
            path = out_dir / f"discord_{ts}_{name}.wav"
            try:
                buf = audio.file
                buf.seek(0)
                path.write_bytes(buf.read())
                files[name] = path
            except Exception as e:
                print(f"[discord] save {name}: {e}")
        self._last_recording = DiscordRecording(
            files=files,
            channel=channel_name,
            guild=self._current_channel.guild.name if self._current_channel else "?",
            started_at=self._record_started_at or 0.0,
            duration=time.time() - (self._record_started_at or time.time()),
        )
        self._record_done.set()

    def join_and_record(self, channel_id: int, out_dir: Path) -> None:
        if not self.is_connected or self._bot is None or self._loop is None:
            raise RuntimeError("Бот не подключён")
        if self.is_recording:
            raise RuntimeError("Уже идёт запись")

        out_dir.mkdir(parents=True, exist_ok=True)

        async def _do_join():
            channel = self._bot.get_channel(channel_id)
            if not isinstance(channel, discord.VoiceChannel):
                raise RuntimeError("Канал не найден или не голосовой")
            vc = await channel.connect()
            self._current_vc = vc
            self._current_channel = channel
            self._sink = WaveSink()
            self._record_started_at = time.time()
            self._record_done.clear()
            self._last_recording = None
            vc.start_recording(
                self._sink,
                self._on_record_finished,
                out_dir=out_dir,
                channel_name=channel.name,
            )

        fut = asyncio.run_coroutine_threadsafe(_do_join(), self._loop)
        fut.result(timeout=15)

    def stop_and_collect(self, timeout: float = 30.0) -> DiscordRecording | None:
        if not self.is_recording or self._loop is None:
            return self._last_recording

        async def _do_stop():
            if self._current_vc is not None:
                try:
                    self._current_vc.stop_recording()
                except Exception:
                    pass
                try:
                    await self._current_vc.disconnect(force=True)
                except Exception:
                    pass

        fut = asyncio.run_coroutine_threadsafe(_do_stop(), self._loop)
        try:
            fut.result(timeout=10)
        except Exception:
            pass

        # ждём пока callback запишет файлы
        self._record_done.wait(timeout=timeout)
        self._current_vc = None
        self._current_channel = None
        self._sink = None
        rec = self._last_recording
        return rec

    @property
    def current_record_duration(self) -> float:
        if self._record_started_at is None:
            return 0.0
        return time.time() - self._record_started_at
