"""``live-transcribe``: microphone -> VAD segmentation -> transcription, as you speak.

Pipeline (three stages, so a slow model never loses audio):

1. **capture**: a sounddevice callback puts raw blocks on a queue;
2. **main thread**: writes every block to the WAV file immediately (header
   kept valid, so the recording survives a crash), forwards it to the worker
   and watches the stop conditions;
3. **worker thread**: runs the Silero VAD bundled with faster-whisper (ONNX,
   no torch) on a rolling buffer, cuts a segment at each pause, transcribes
   it with the previous text as ``initial_prompt``, appends it to the
   ``.live.txt`` log (flushed + fsynced) and reports it.

Stop: Enter or Ctrl+C (interactive), ``max_duration``, ``stop_file`` (works on
every OS), SIGINT/SIGTERM. A second Ctrl+C aborts without waiting.
"""

from __future__ import annotations

import contextlib
import os
import queue
import signal
import sys
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from uvt_core.errors import InvalidInputError, MissingDependencyError, ToolError
from uvt_core.jobs import Artifact, ToolOutput
from uvt_core.logging import get_logger
from uvt_transcribe.audio import StreamingWavWriter, resample
from uvt_transcribe.backends import TranscriptionBackend
from uvt_transcribe.backends.base import SAMPLE_RATE, SegmentCallback
from uvt_transcribe.formats import format_timestamp, safe_filename, write_outputs
from uvt_transcribe.models import Segment, SourceInfo, Transcript, join_text
from uvt_transcribe.options import (
    LiveTranscribeParams,
    formats,
    make_backend,
    output_dir,
    resolve_options,
)
from uvt_transcribe.pipeline import build_result

if TYPE_CHECKING:
    import numpy as np

    from uvt_transcribe.models import TranscriptionResult

log = get_logger("transcribe.live")

PORTAUDIO_HINTS = {
    "Linux": "Install PortAudio: sudo apt install libportaudio2  "
    "(Fedora: sudo dnf install portaudio · Arch: sudo pacman -S portaudio)",
    "Darwin": "PortAudio is bundled with the sounddevice wheel: reinstall the tool "
    "(uv tool install --reinstall ...), or brew install portaudio.",
    "Windows": "PortAudio is bundled with the sounddevice wheel: reinstall the tool "
    "(uv tool install --reinstall ...).",
}
_PROMPT_CHARS = 200
_VAD_EVERY_S = 0.3


class MicInfo(BaseModel):
    index: int
    name: str
    hostapi: str
    channels: int
    default_samplerate: float
    is_default: bool


class MicList(ToolOutput):
    mics: list[MicInfo]


def import_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except OSError as exc:  # raised when the PortAudio shared library is missing
        raise MissingDependencyError(
            f"PortAudio is not available ({exc}); it is needed for microphone capture.",
            os_hints=PORTAUDIO_HINTS,
        ) from exc
    except ImportError as exc:  # pragma: no cover - base dependency
        raise MissingDependencyError(f"sounddevice is not importable: {exc}") from exc
    return sd


def list_microphones() -> list[MicInfo]:
    """Input devices as seen by PortAudio."""
    sd = import_sounddevice()
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        default_in = sd.default.device[0]
    except Exception as exc:
        raise ToolError(f"Cannot query audio devices: {exc}") from exc
    return [
        MicInfo(
            index=i,
            name=d["name"],
            hostapi=hostapis[d["hostapi"]]["name"],
            channels=d["max_input_channels"],
            default_samplerate=d["default_samplerate"],
            is_default=i == default_in,
        )
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


def resolve_mic(mic: str | None) -> MicInfo:
    mics = list_microphones()
    if not mics:
        raise ToolError(
            "No microphone found.", hint="Connect a microphone, or check OS audio permissions."
        )
    if mic is None or mic == "":
        default = next((m for m in mics if m.is_default), None)
        return default or mics[0]
    if mic.strip().isdigit():
        index = int(mic)
        for m in mics:
            if m.index == index:
                return m
        raise InvalidInputError(
            f"No input device with index {index}.", hint="List them with --list-mics."
        )
    matches = [m for m in mics if mic.lower() in m.name.lower()]
    if not matches:
        raise InvalidInputError(
            f"No microphone matches {mic!r}.", hint="List them with --list-mics."
        )
    if len(matches) > 1:
        log.warning("%d devices match %r; using %s", len(matches), mic, matches[0].name)
    return matches[0]


class _StopController:
    """Collects stop requests from signals, Enter, the stop file and timers."""

    def __init__(self, stop_file: Path | None, external: threading.Event | None) -> None:
        self.event = external or threading.Event()
        self.reason: str | None = None
        self.stop_file = stop_file
        self._interrupts = 0

    def request(self, reason: str) -> None:
        if not self.event.is_set():
            self.reason = reason
            log.info("Stopping (%s)…", reason)
        self.event.set()

    def _on_signal(self, signum: int, frame: Any) -> None:
        self._interrupts += 1
        if self._interrupts > 1 and signum == signal.SIGINT:
            raise KeyboardInterrupt
        self.request(signal.Signals(signum).name)

    @contextlib.contextmanager
    def installed(self) -> Iterator[None]:
        previous: dict[int, Any] = {}
        if threading.current_thread() is threading.main_thread():
            for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
                sig = getattr(signal, name, None)
                if sig is not None:
                    previous[sig] = signal.signal(sig, self._on_signal)
        if sys.stdin is not None and sys.stdin.isatty():
            threading.Thread(target=self._wait_enter, daemon=True, name="stdin-stop").start()
        try:
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def _wait_enter(self) -> None:
        with contextlib.suppress(Exception):
            sys.stdin.readline()
            self.request("Enter")

    def poll(self, recorded_s: float, max_duration: float | None) -> bool:
        if self.stop_file is not None and self.stop_file.exists():
            self.request(f"stop file {self.stop_file}")
        if max_duration is not None and recorded_s >= max_duration:
            self.request(f"max duration {max_duration:g} s")
        return self.event.is_set()


class _Segmenter:
    """Worker: VAD on a rolling buffer, transcription of each closed segment."""

    def __init__(
        self,
        backend: TranscriptionBackend,
        opts: LiveTranscribeParams,
        log_file: Any,
        on_segment: SegmentCallback | None,
    ) -> None:
        import numpy as np
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        self.np = np
        self._vad = get_speech_timestamps
        self._vad_opts = VadOptions(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=int(opts.min_silence * 1000),
            speech_pad_ms=200,
            max_speech_duration_s=opts.max_segment,
        )
        self.backend = backend
        self.opts = opts
        self.language = opts.language
        self.log_file = log_file
        self.on_segment = on_segment
        self.queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self.segments: list[Segment] = []
        self.language_probability: float | None = None
        self.error: BaseException | None = None
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0.0  # seconds from the start of the recording
        self.thread = threading.Thread(target=self._run, name="live-worker", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        np = self.np
        since_vad = 0
        try:
            while True:
                chunk = self.queue.get()
                if chunk is None:
                    self._process(final=True)
                    return
                self._buf = np.concatenate([self._buf, chunk])
                since_vad += len(chunk)
                if since_vad >= _VAD_EVERY_S * SAMPLE_RATE:
                    since_vad = 0
                    self._process(final=False)
        except BaseException as exc:  # noqa: BLE001 - reported by the main thread
            self.error = exc

    def _process(self, *, final: bool) -> None:
        buf = self._buf
        if buf.size == 0:
            return
        stamps = self._vad(buf, self._vad_opts)
        buffered = buf.size / SAMPLE_RATE
        if not stamps:
            keep = int(0.5 * SAMPLE_RATE)  # keep a little audio so word onsets are not cut
            if buf.size > 2 * keep:
                self._advance(buf.size - keep)
            return
        silence_after = buffered - stamps[-1]["end"] / SAMPLE_RATE
        if final or silence_after > 0.05:
            cut = stamps[-1]["end"] if not final else buf.size
            self._emit(stamps[0]["start"], cut)
        elif buffered >= self.opts.max_segment:
            cut = stamps[-1]["start"] if len(stamps) > 1 else buf.size
            self._emit(stamps[0]["start"], cut)

    def _advance(self, n: int) -> None:
        self._buf = self._buf[n:]
        self._buf_start += n / SAMPLE_RATE

    def _prompt(self) -> str | None:
        previous = join_text(self.segments)[-_PROMPT_CHARS:]
        parts = [p for p in (self.opts.initial_prompt, previous) if p]
        return " ".join(parts) or None

    def _emit(self, start: int, end: int) -> None:
        audio = self._buf[start:end]
        offset = self._buf_start + start / SAMPLE_RATE
        self._advance(end)
        if audio.size < int(0.3 * SAMPLE_RATE):
            return
        transcript = self.backend.transcribe(
            audio,
            language=self.language,
            initial_prompt=self._prompt(),
            vad=False,
            beam_size=self.opts.beam_size,
        )
        if self.language is None and transcript.language:
            # Lock the language after the first detection: short segments flip easily.
            self.language = transcript.language
            self.language_probability = transcript.language_probability
            log.info("Detected language: %s", self.language)
        for seg in transcript.segments:
            if not seg.text.strip():
                continue
            seg = seg.model_copy(
                update={
                    "id": len(self.segments),
                    "start": round(offset + seg.start, 3),
                    "end": round(offset + min(seg.end, audio.size / SAMPLE_RATE), 3),
                    "words": None,
                }
            )
            self.segments.append(seg)
            stamp = format_timestamp(seg.start, decimal=".", always_hours=False)
            self.log_file.write(f"[{stamp}] {seg.text}\n")
            self.log_file.flush()
            with contextlib.suppress(OSError):
                os.fsync(self.log_file.fileno())
            if self.on_segment is not None:
                self.on_segment(seg)


def _open_stream(sd: Any, mic: MicInfo, q: queue.Queue) -> tuple[Any, int]:
    """Open a mono input stream, at 16 kHz if the device allows it."""
    status_warned = {"done": False}

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status and not status_warned["done"]:
            status_warned["done"] = True
            log.warning("Audio input status: %s", status)
        q.put(indata[:, 0].copy())

    rates = [SAMPLE_RATE, int(mic.default_samplerate)]
    last_error: Exception | None = None
    for rate in dict.fromkeys(rates):
        try:
            sd.check_input_settings(device=mic.index, channels=1, samplerate=rate, dtype="float32")
            stream = sd.InputStream(
                device=mic.index,
                channels=1,
                samplerate=rate,
                dtype="float32",
                blocksize=int(rate * 0.1),
                callback=callback,
            )
            return stream, rate
        except Exception as exc:  # noqa: BLE001 - PortAudioError
            last_error = exc
    raise ToolError(
        f"Cannot open microphone {mic.name!r}: {last_error}",
        hint="Pick another device with --mic (see --list-mics) and check OS permissions.",
    )


def live_transcribe(
    params: LiveTranscribeParams,
    *,
    on_segment: SegmentCallback | None = None,
    backend: TranscriptionBackend | None = None,
    stop_event: threading.Event | None = None,
) -> TranscriptionResult:
    """Record from the microphone and transcribe while recording.

    ``stop_event`` lets an embedding program stop the session.
    """
    opts, cfg = resolve_options(params)
    stop_file = opts.stop_file.expanduser() if opts.stop_file else None
    if stop_file is not None and stop_file.exists():
        raise InvalidInputError(
            f"Stop file {stop_file} already exists.",
            hint="Delete it before starting, or choose another path.",
        )
    sd = import_sounddevice()
    mic = resolve_mic(opts.mic)
    out_dir = output_dir(opts)
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = safe_filename(opts.name or datetime.now().strftime("live-%Y%m%d-%H%M%S"))
    wav_path = out_dir / f"{basename}.wav"
    log_path = out_dir / f"{basename}.live.txt"

    owned = backend is None
    engine = backend or make_backend(opts, cfg)
    try:
        engine.load()
        recorded, segmenter, stopper = _record(
            sd, mic, engine, opts, wav_path, log_path, on_segment, stop_file, stop_event
        )
        segments = segmenter.segments
        language = segmenter.language
        language_probability = segmenter.language_probability
        if opts.final_pass and recorded > 0.5:
            log.info("Final pass over the full recording…")
            final = engine.transcribe(
                wav_path,
                language=language,
                initial_prompt=opts.initial_prompt,
                vad=True,
                beam_size=opts.beam_size,
            )
            segments = final.segments
            language = final.language or language
    finally:
        if owned:
            engine.close()

    transcript = Transcript(
        segments=segments,
        language=language,
        language_probability=language_probability,
        duration=round(recorded, 3),
    )
    source = SourceInfo(
        kind="microphone", mic=mic.name, path=str(wav_path.resolve()), duration=round(recorded, 3)
    )
    result = build_result(transcript, engine, source)
    artifacts = [Artifact.from_path("audio", wav_path), Artifact.from_path("live_log", log_path)]
    result.artifacts = artifacts + write_outputs(result, formats(opts), out_dir, basename)
    log.info(
        "Recorded %.1f s, %d segments (stopped by %s)",
        recorded,
        len(segments),
        stopper.reason or "end of input",
    )
    return result


def _record(
    sd: Any,
    mic: MicInfo,
    engine: TranscriptionBackend,
    opts: LiveTranscribeParams,
    wav_path: Path,
    log_path: Path,
    on_segment: SegmentCallback | None,
    stop_file: Path | None,
    stop_event: threading.Event | None,
) -> tuple[float, _Segmenter, _StopController]:
    capture_q: queue.Queue = queue.Queue()
    stopper = _StopController(stop_file, stop_event)
    with (
        log_path.open("a", encoding="utf-8", newline="\n") as log_file,
        StreamingWavWriter(wav_path) as wav,
        stopper.installed(),
    ):
        segmenter = _Segmenter(engine, opts, log_file, on_segment)
        stream, rate = _open_stream(sd, mic, capture_q)
        segmenter.start()
        how = "press Enter or Ctrl+C to stop" if sys.stdin and sys.stdin.isatty() else ""
        extra = [
            x
            for x in (
                how,
                opts.max_duration and f"max {opts.max_duration:g} s",
                stop_file and f"stop file {stop_file}",
            )
            if x
        ]
        log.info(
            "Listening on %s (%d Hz)%s", mic.name, rate, f" — {', '.join(extra)}" if extra else ""
        )
        last_flush = time.monotonic()
        try:
            with stream:
                while not stopper.poll(wav.duration, opts.max_duration):
                    _pump(capture_q, rate, wav, segmenter, timeout=0.1)
                    if segmenter.error is not None:
                        raise segmenter.error
                    if time.monotonic() - last_flush >= 1.0:
                        wav.flush()
                        last_flush = time.monotonic()
            # Stream closed: drain what was captured before stopping.
            while not capture_q.empty():
                _pump(capture_q, rate, wav, segmenter, timeout=0)
            wav.flush()
            log.info("Finishing the last segment…")
            segmenter.queue.put(None)
            while segmenter.thread.is_alive():
                segmenter.thread.join(timeout=0.2)
            if segmenter.error is not None:
                raise segmenter.error
        except ToolError:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise ToolError(
                f"Live transcription failed: {exc}", hint=f"The recording so far is in {wav_path}."
            ) from exc
        return wav.duration, segmenter, stopper


def _pump(
    capture_q: queue.Queue,
    rate: int,
    wav: StreamingWavWriter,
    segmenter: _Segmenter,
    *,
    timeout: float,
) -> None:
    try:
        block = capture_q.get(timeout=timeout) if timeout else capture_q.get_nowait()
    except queue.Empty:
        return
    samples = resample(block, rate, SAMPLE_RATE)
    wav.write(samples)
    segmenter.queue.put(samples)
