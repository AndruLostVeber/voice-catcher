from __future__ import annotations

import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pyaudiowpatch as pyaudio

CHUNK = 1024
SAMPLE_FORMAT = pyaudio.paInt16


@dataclass
class CallRecording:
    system_path: Path
    mic_path: Path
    duration: float


class CallRecorder:
    """Параллельная запись системного звука (WASAPI loopback) и микрофона
    через callback-API PyAudio. Файлы пишутся напрямую из колбэка."""

    def __init__(self):
        self._pa = pyaudio.PyAudio()
        self._streams: list = []
        self._files: list = []
        self._errors: list[str] = []
        self.started_at: float | None = None

    @staticmethod
    def _make_callback(wf: wave.Wave_write):
        def cb(in_data, frame_count, time_info, status):
            wf.writeframes(in_data)
            return (in_data, pyaudio.paContinue)

        return cb

    def _open_stream(self, device_info: dict, wav_path: Path):
        channels = int(device_info.get("maxInputChannels", 1))
        rate = int(device_info["defaultSampleRate"])

        wf = wave.open(str(wav_path), "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(self._pa.get_sample_size(SAMPLE_FORMAT))
        wf.setframerate(rate)
        self._files.append(wf)

        stream = self._pa.open(
            format=SAMPLE_FORMAT,
            channels=channels,
            rate=rate,
            frames_per_buffer=CHUNK,
            input=True,
            input_device_index=device_info["index"],
            stream_callback=self._make_callback(wf),
            start=False,
        )
        self._streams.append(stream)

    def start(self, out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        system_path = out_dir / f"call_{ts}_system.wav"
        mic_path = out_dir / f"call_{ts}_mic.wav"

        try:
            loopback = self._pa.get_default_wasapi_loopback()
        except OSError as e:
            raise RuntimeError(
                "WASAPI loopback не найден. Проверь Default Output в Windows."
            ) from e

        try:
            mic = self._pa.get_default_input_device_info()
        except OSError as e:
            raise RuntimeError("Микрофон по умолчанию не найден.") from e

        try:
            self._open_stream(loopback, system_path)
        except Exception as e:
            self._errors.append(f"loopback open: {e}")
            raise RuntimeError(f"Не открылся системный поток: {e}") from e

        try:
            self._open_stream(mic, mic_path)
        except Exception as e:
            self._errors.append(f"mic open: {e}")
            self._cleanup()
            raise RuntimeError(f"Не открылся микрофон: {e}") from e

        for s in self._streams:
            s.start_stream()

        self.started_at = time.time()
        return system_path, mic_path

    def _cleanup(self):
        for s in self._streams:
            try:
                if s.is_active():
                    s.stop_stream()
                s.close()
            except Exception as e:
                self._errors.append(f"stream close: {e}")
        self._streams = []
        for f in self._files:
            try:
                f.close()
            except Exception as e:
                self._errors.append(f"file close: {e}")
        self._files = []

    def stop(self) -> float:
        duration = time.time() - (self.started_at or time.time())
        self._cleanup()
        try:
            self._pa.terminate()
        except Exception:
            pass
        return duration

    @property
    def duration(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.time() - self.started_at

    @property
    def errors(self) -> list[str]:
        return list(self._errors)


def get_loopback_info() -> dict | None:
    p = pyaudio.PyAudio()
    try:
        info = p.get_default_wasapi_loopback()
        return {
            "name": info["name"],
            "channels": int(info.get("maxInputChannels", 0)),
            "rate": int(info["defaultSampleRate"]),
            "index": info["index"],
        }
    except OSError:
        return None
    finally:
        p.terminate()
