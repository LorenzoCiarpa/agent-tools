"""Shared fixtures: a fake backend and a fake microphone (no models, no hardware)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from uvt_core.device import DeviceInfo
from uvt_transcribe.backends import register_backend
from uvt_transcribe.backends.base import SAMPLE_RATE, TranscriptionBackend
from uvt_transcribe.models import Segment, Transcript

DATA = Path(__file__).parent / "data"


class FakeBackend(TranscriptionBackend):
    """Returns one segment per call: 'chunk <n>' spanning the whole audio."""

    name: ClassVar[str] = "fake"
    instances: ClassVar[list[FakeBackend]] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls: list[dict[str, Any]] = []
        self.loaded = False
        self.closed = False
        FakeBackend.instances.append(self)

    def load(self) -> None:
        self.loaded = True

    @property
    def model_name(self) -> str:
        return self.requested_model or "fake-model"

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(kind="cpu", name="fake")

    def transcribe(
        self,
        audio,
        *,
        language=None,
        initial_prompt=None,
        vad=True,
        beam_size=None,
        on_segment=None,
    ):
        if isinstance(audio, str | Path):
            from uvt_transcribe.audio import load_audio

            audio = load_audio(Path(audio))
        duration = len(audio) / SAMPLE_RATE
        n = len(self.calls)
        self.calls.append(
            {
                "language": language,
                "initial_prompt": initial_prompt,
                "vad": vad,
                "duration": duration,
            }
        )
        seg = Segment(id=0, start=0.0, end=round(duration, 3), text=f"chunk {n}")
        if on_segment:
            on_segment(seg)
        return Transcript(
            segments=[seg], language=language or "en", language_probability=0.99, duration=duration
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Never read the developer's real config; register the fake backend."""
    monkeypatch.setenv("UVT_CONFIG_FILE", str(tmp_path / "no-config.toml"))
    for key in list(__import__("os").environ):
        if key.startswith("UVT_TRANSCRIBE__"):
            monkeypatch.delenv(key)
    register_backend("fake", FakeBackend)
    FakeBackend.instances.clear()


@pytest.fixture
def fake_backends() -> list[FakeBackend]:
    """Instances created by the code under test, newest last."""
    return FakeBackend.instances


@pytest.fixture
def jfk() -> Path:
    return DATA / "jfk.ogg"


@pytest.fixture
def jfk_samples(jfk) -> np.ndarray:
    from uvt_transcribe.audio import load_audio

    return load_audio(jfk)


class FakeInputStream:
    """Feeds pre-recorded audio to the callback, faster than real time."""

    def __init__(
        self,
        source: np.ndarray,
        *,
        device,
        channels,
        samplerate,
        dtype,
        blocksize,
        callback,
        speed: float = 8.0,
    ):
        self.source = source
        self.blocksize = blocksize
        self.callback = callback
        self.speed = speed
        self.rate = samplerate
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        pos = 0
        while not self._stop.is_set():
            block = self.source[pos : pos + self.blocksize]
            if block.size < self.blocksize:  # after the recording: silence
                block = np.concatenate([block, np.zeros(self.blocksize - block.size, np.float32)])
            pos += self.blocksize
            self.callback(block.reshape(-1, 1).astype(np.float32), self.blocksize, None, None)
            time.sleep(self.blocksize / self.rate / self.speed)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


@pytest.fixture
def fake_sd(monkeypatch, jfk_samples):
    """A fake sounddevice module with two microphones; the default one 'plays' JFK.

    The device runs at 48 kHz so the resampling path is exercised too.
    """
    from uvt_transcribe import audio, live

    source = audio.resample(
        np.concatenate([np.zeros(8000, np.float32), jfk_samples]), SAMPLE_RATE, 48_000
    )
    devices = [
        {"name": name, "hostapi": 0, "max_input_channels": ch, "default_samplerate": 48000.0}
        for name, ch in (("Speakers", 0), ("Built-in Mic", 1), ("USB Mic", 2))
    ]

    def check_input_settings(*, device, channels, samplerate, dtype):
        if samplerate != 48000:
            raise ValueError("Invalid sample rate")

    module = SimpleNamespace(
        query_devices=lambda: devices,
        query_hostapis=lambda: [{"name": "FakeAPI"}],
        default=SimpleNamespace(device=(1, 0)),
        check_input_settings=check_input_settings,
        InputStream=lambda **kw: FakeInputStream(source, **kw),
    )
    monkeypatch.setattr(live, "import_sounddevice", lambda: module)
    return module
