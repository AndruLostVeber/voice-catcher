from __future__ import annotations

import threading
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
    """Параллельная запись системного звука (WASAPI loopback) и микрофона.

    Системный поток — то, что слышит пользователь (голос собеседника в звонке).
    Микрофон — то, что говорит сам пользователь.
    """

    def __init__(self):
        self._pa = pyaudio.PyAudio()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._errors: list[str] = []
        self.started_at: float | None = None

    def _record_to_wav(self, device_info: dict, wav_path: Path) -> None:
        channels = int(device_info.get("maxInputChannels", 1))
        rate = int(device_info["defaultSampleRate"])

        wf = wave.open(str(wav_path), "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(self._pa.get_sample_size(SAMPLE_FORMAT))
        wf.setframerate(rate)

        try:
            stream = self._pa.open(
                format=SAMPLE_FORMAT,
                channels=channels,
                rate=rate,
                frames_per_buffer=CHUNK,
                input=True,
                input_device_index=device_info["index"],
            )
        except Exception as e:
            self._errors.append(f"open stream ({device_info['name']}): {e}")
            wf.close()
            return

        try:
            while not self._stop.is_set():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    wf.writeframes(data)
                except OSError as e:
                    self._errors.append(f"read ({device_info['name']}): {e}")
                    break
        finally:
            stream.stop_stream()
            stream.close()
            wf.close()

    def start(self, out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        system_path = out_dir / f"call_{ts}_system.wav"
        mic_path = out_dir / f"call_{ts}_mic.wav"

        try:
            loopback = self._pa.get_default_wasapi_loopback()
        except OSError as e:
            raise RuntimeError(
                "WASAPI loopback не найден. Убедись что выбрано устройство вывода по умолчанию."
            ) from e

        try:
            mic = self._pa.get_default_input_device_info()
        except OSError as e:
            raise RuntimeError("Микрофон по умолчанию не найден.") from e

        self._stop.clear()
        self._errors.clear()
        self._threads = [
            threading.Thread(target=self._record_to_wav, args=(loopback, system_path), daemon=True),
            threading.Thread(target=self._record_to_wav, args=(mic, mic_path), daemon=True),
        ]
        for t in self._threads:
            t.start()

        self.started_at = time.time()
        return system_path, mic_path

    def stop(self) -> float:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3.0)
        duration = time.time() - (self.started_at or time.time())
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
    """Возвращает инфу о текущем default loopback (для UI), или None если недоступно."""
    p = pyaudio.PyAudio()
    try:
        info = p.get_default_wasapi_loopback()
        return {
            "name": info["name"],
            "channels": int(info.get("maxInputChannels", 0)),
            "rate": int(info["defaultSampleRate"]),
        }
    except OSError:
        return None
    finally:
        p.terminate()
