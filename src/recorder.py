from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"


class Recorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []
        self._consumer: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self.started_at: float | None = None

    def _callback(self, indata, frames, t, status):
        if status:
            print(f"[recorder] {status}")
        self._queue.put(indata.copy())

    def _consume(self):
        while not self._stop_flag.is_set():
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._frames.append(chunk)

    def start(self):
        if self._stream is not None:
            raise RuntimeError("recorder already running")
        self._frames.clear()
        self._stop_flag.clear()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._callback,
        )
        self._stream.start()
        self.started_at = time.time()
        self._consumer = threading.Thread(target=self._consume, daemon=True)
        self._consumer.start()

    def stop(self, out_path: str | Path) -> Path:
        if self._stream is None:
            raise RuntimeError("recorder not running")
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._stop_flag.set()
        if self._consumer is not None:
            self._consumer.join(timeout=2.0)

        while not self._queue.empty():
            self._frames.append(self._queue.get_nowait())

        if not self._frames:
            raise RuntimeError("no audio captured")

        audio = np.concatenate(self._frames, axis=0)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(out_path, self.sample_rate, audio)
        return out_path

    @property
    def duration(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.time() - self.started_at


def list_input_devices() -> list[dict]:
    devices = sd.query_devices()
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]
