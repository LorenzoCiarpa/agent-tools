"""Audio helpers: decoding (PyAV, no ffmpeg binary needed), WAV writing, resampling."""

from __future__ import annotations

import gc
import io
import struct
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from uvt_core.errors import InvalidInputError, MissingDependencyError
from uvt_transcribe.backends.base import SAMPLE_RATE

if TYPE_CHECKING:
    import numpy as np


def load_audio(path: Path, sampling_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio/video file to mono float32 at ``sampling_rate``.

    Uses PyAV, which bundles FFmpeg's libraries: no ffmpeg executable needed.
    (We do not use ``faster_whisper.decode_audio``: faster-whisper 1.2.1 is
    incompatible with PyAV >= 19.)
    """
    try:
        import av
        import numpy as np
    except ImportError as exc:  # pragma: no cover - base dependencies
        raise MissingDependencyError(
            f"Cannot decode audio: {exc}", hint="Reinstall uvt-transcribe."
        ) from exc

    chunks: list[np.ndarray] = []
    try:
        with av.open(str(path), mode="r") as container:
            if not container.streams.audio:
                raise InvalidInputError(
                    f"{path} has no audio track.", hint="Pass a file that contains audio."
                )
            resampler = av.audio.resampler.AudioResampler(
                format="s16", layout="mono", rate=sampling_rate
            )
            for frame in _valid_frames(container.decode(audio=0), av):
                frame.pts = None  # let the resampler ignore broken timestamps
                chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
            chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
            del resampler
    except InvalidInputError:
        raise
    except Exception as exc:
        raise InvalidInputError(
            f"Cannot decode audio from {path}: {exc}",
            hint="Check that the file is a valid audio/video file with an audio track.",
        ) from exc
    finally:
        gc.collect()  # PyAV resampler objects are only freed by the GC
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _valid_frames(frames: Iterator, av: object) -> Iterator:
    """Skip frames FFmpeg cannot decode instead of aborting the whole file."""
    invalid = av.error.InvalidDataError  # type: ignore[attr-defined]
    iterator = iter(frames)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            return
        except invalid:
            continue


def to_int16(samples: np.ndarray) -> bytes:
    import numpy as np

    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def wav_bytes(samples: np.ndarray, sampling_rate: int = SAMPLE_RATE) -> bytes:
    """Encode mono float32 samples as a 16-bit PCM WAV file in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sampling_rate)
        wf.writeframes(to_int16(samples))
    return buf.getvalue()


def resample(samples: np.ndarray, src_rate: int, dst_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Linear-interpolation resampling: good enough for speech recognition."""
    import numpy as np

    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    n_out = round(samples.size * dst_rate / src_rate)
    x_old = np.arange(samples.size, dtype=np.float64) / src_rate
    x_new = np.arange(n_out, dtype=np.float64) / dst_rate
    return np.interp(x_new, x_old, samples).astype(np.float32)


class StreamingWavWriter:
    """Mono 16-bit WAV writer that keeps the header valid while recording.

    The RIFF/data sizes are rewritten on every ``flush()``, so if the process
    dies the file is still playable up to the last flush.
    """

    def __init__(self, path: Path, sampling_rate: int = SAMPLE_RATE) -> None:
        self.path = Path(path)
        self.sampling_rate = sampling_rate
        self.frames = 0
        self._fh: BinaryIO = self.path.open("wb")
        self._write_header()

    def _write_header(self) -> None:
        data_size = self.frames * 2
        header = b"".join(
            [
                b"RIFF",
                struct.pack("<I", 36 + data_size),
                b"WAVE",
                b"fmt ",
                struct.pack(
                    "<IHHIIHH", 16, 1, 1, self.sampling_rate, self.sampling_rate * 2, 2, 16
                ),
                b"data",
                struct.pack("<I", data_size),
            ]
        )
        self._fh.seek(0)
        self._fh.write(header)
        self._fh.seek(0, io.SEEK_END)

    def write(self, samples: np.ndarray) -> None:
        data = to_int16(samples)
        self._fh.write(data)
        self.frames += len(data) // 2

    def flush(self) -> None:
        self._write_header()
        self._fh.flush()

    @property
    def duration(self) -> float:
        return self.frames / self.sampling_rate

    def close(self) -> None:
        if not self._fh.closed:
            self.flush()
            self._fh.close()

    def __enter__(self) -> StreamingWavWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
