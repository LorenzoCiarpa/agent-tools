"""faster-whisper (CTranslate2) backend: the default, no torch required.

* ``--device auto``: CUDA in float16 when a GPU is usable, else CPU int8.
  A short warm-up runs at load time: on GPUs newer than the CTranslate2 build
  (e.g. Blackwell) the first run JIT-compiles kernels, and a broken CUDA setup
  surfaces here so ``auto`` can fall back to CPU instead of failing mid-file.
* Default model: ``model_gpu`` / ``model_cpu`` from the ``[transcribe]``
  config, else large-v3-turbo on CUDA and small on CPU.
* ``--backend-option`` keys are forwarded to ``WhisperModel(...)`` or
  ``WhisperModel.transcribe(...)`` (e.g. ``word_timestamps=true``,
  ``cpu_threads=8``, ``hotwords=...``); ``batch_size=N`` switches to the
  batched pipeline (faster on GPU for long files).
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from uvt_core import device as dev
from uvt_core.device import DeviceInfo
from uvt_core.errors import InvalidInputError, MissingDependencyError, ToolError
from uvt_core.logging import get_logger
from uvt_transcribe.audio import load_audio
from uvt_transcribe.backends._cuda import CUDA_HINT, preload_cuda_libraries
from uvt_transcribe.backends.base import (
    SAMPLE_RATE,
    SegmentCallback,
    TranscriptionBackend,
    quiet_hf_hub,
)
from uvt_transcribe.models import Segment, Transcript, Word

if TYPE_CHECKING:
    import numpy as np

log = get_logger("transcribe.faster_whisper")

DEFAULT_GPU_MODEL = "large-v3-turbo"
DEFAULT_CPU_MODEL = "small"
_DEFAULT_COMPUTE = {"cuda": "float16", "cpu": "int8"}
_CONSTRUCTOR_KEYS = {
    "device_index",
    "cpu_threads",
    "num_workers",
    "download_root",
    "local_files_only",
    "revision",
}


def _coerce(value: Any) -> Any:
    """``--backend-option`` values arrive as strings; parse obvious literals."""
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


class FasterWhisperBackend(TranscriptionBackend):
    name: ClassVar[str] = "faster-whisper"
    description: ClassVar[str] = "Whisper via CTranslate2 (CUDA or CPU, no torch)."

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model: Any = None
        self._pipeline: Any = None
        self._kind: str = "cpu"
        self._model_name: str = self.requested_model or "?"
        self._compute_type: str | None = None
        self._ctor_opts: dict[str, Any] = {}
        self._transcribe_opts: dict[str, Any] = {}
        self._batch_size: int | None = None

    # -- loading -----------------------------------------------------------------

    def _split_options(self) -> None:
        from faster_whisper import WhisperModel

        allowed = set(inspect.signature(WhisperModel.transcribe).parameters) - {"self", "audio"}
        for key, raw in self.options.items():
            value = _coerce(raw)
            if key == "batch_size":
                self._batch_size = int(value)
            elif key in _CONSTRUCTOR_KEYS:
                self._ctor_opts[key] = value
            elif key in allowed:
                self._transcribe_opts[key] = value
            else:
                raise InvalidInputError(
                    f"Unknown faster-whisper option {key!r}.",
                    hint="Valid keys: batch_size, "
                    + ", ".join(sorted(_CONSTRUCTOR_KEYS | allowed)),
                )
        if "download_root" not in self._ctor_opts and self.config.get("download_root"):
            self._ctor_opts["download_root"] = str(Path(self.config["download_root"]).expanduser())

    def load(self) -> None:
        if self._model is not None:
            return
        preload_cuda_libraries()
        quiet_hf_hub()
        try:
            import faster_whisper  # noqa: F401
        except ImportError as exc:
            raise MissingDependencyError(
                f"faster-whisper is not importable: {exc}", hint="Reinstall uvt-transcribe."
            ) from exc
        self._split_options()

        requested = (self.requested_device or "auto").lower()
        if requested == "mps":
            log.info("faster-whisper has no Apple GPU support; using the CPU.")
            requested = "cpu"
        kind = dev.detect_device(requested)
        if kind == "mps":
            kind = "cpu"  # CTranslate2 runs on the CPU (Accelerate) on macOS
        if kind == "cuda" and dev.cuda_device_count() == 0:
            if requested == "cuda":
                raise ToolError(
                    "--device cuda was requested but no CUDA GPU is usable.",
                    hint="Check `nvidia-smi`, or use --device auto / cpu.",
                )
            kind = "cpu"

        try:
            self._load_on(kind)
        except ToolError:
            raise
        except Exception as exc:
            if kind == "cuda" and requested == "auto":
                log.warning("CUDA is not usable (%s); falling back to CPU.", exc)
                log.warning(
                    "  hint: install the 'cuda' extra (bundled cuBLAS) or a CUDA 12 toolkit "
                    "to use the GPU; --device cpu skips this check."
                )
                self._model = None
                self._load_on("cpu")
            elif kind == "cuda":
                raise ToolError(f"Could not run on CUDA: {exc}", hint=CUDA_HINT) from exc
            else:
                raise ToolError(f"Could not load the model: {exc}") from exc

    def _default_model(self, kind: str) -> str:
        key = "model_gpu" if kind == "cuda" else "model_cpu"
        default = DEFAULT_GPU_MODEL if kind == "cuda" else DEFAULT_CPU_MODEL
        return str(self.requested_model or self.config.get(key) or default)

    def _load_on(self, kind: str) -> None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        model_name = self._default_model(kind)
        compute = self.requested_compute_type or self.config.get(f"compute_type_{kind}")
        compute = str(compute or _DEFAULT_COMPUTE[kind])
        log.info("Loading faster-whisper model %s on %s (%s)…", model_name, kind, compute)
        t0 = time.perf_counter()
        try:
            model = WhisperModel(model_name, device=kind, compute_type=compute, **self._ctor_opts)
        except ValueError as exc:
            if "Invalid model size" in str(exc):
                raise InvalidInputError(
                    f"Unknown faster-whisper model {model_name!r}.",
                    hint="Use a size such as tiny, base, small, medium, large-v3, "
                    "large-v3-turbo, distil-large-v3.5, a Hugging Face repo id with a "
                    "CTranslate2 model, or a local model directory.",
                ) from exc
            raise ValueError(f"Invalid compute type or option: {exc}") from exc
        self._model, self._kind, self._model_name, self._compute_type = (
            model,
            kind,
            model_name,
            compute,
        )
        if kind == "cuda":
            self._warmup()
        if self._batch_size:
            self._pipeline = BatchedInferencePipeline(model=model)
        log.info("Model ready in %.1f s", time.perf_counter() - t0)

    def _warmup(self) -> None:
        import numpy as np

        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        segments, _ = self._model.transcribe(silence, language="en", beam_size=1)
        list(segments)

    # -- properties ----------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device_info(self) -> DeviceInfo:
        index = int(self._ctor_opts.get("device_index", 0) or 0)
        info = dev.describe(self._kind, index=index, compute_type=self._compute_type)  # type: ignore[arg-type]
        info.details["engine"] = "ctranslate2"
        return info

    # -- transcription ---------------------------------------------------------------

    def transcribe(
        self,
        audio: Path | np.ndarray,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        vad: bool = True,
        beam_size: int | None = None,
        on_segment: SegmentCallback | None = None,
    ) -> Transcript:
        self.load()
        samples = load_audio(audio) if isinstance(audio, Path | str) else audio
        duration = len(samples) / SAMPLE_RATE
        kwargs: dict[str, Any] = {
            "language": language,
            "beam_size": int(beam_size or self.config.get("beam_size") or 5),
            "vad_filter": vad,
            "initial_prompt": initial_prompt,
        }
        kwargs.update(self._transcribe_opts)
        engine = self._model
        if self._pipeline is not None:
            engine = self._pipeline
            kwargs["batch_size"] = self._batch_size
            if not vad and duration > 30:
                log.warning("Batched mode needs VAD for audio > 30 s; enabling it.")
                kwargs["vad_filter"] = True
        try:
            segments_iter, info = engine.transcribe(samples, **kwargs)
        except ValueError as exc:
            raise InvalidInputError(f"Invalid transcription settings: {exc}") from exc

        segments: list[Segment] = []
        next_report = 0.1
        for raw in segments_iter:
            seg = Segment(
                id=len(segments),
                start=round(float(raw.start), 3),
                end=round(float(raw.end), 3),
                text=raw.text.strip(),
                avg_logprob=float(raw.avg_logprob),
                no_speech_prob=float(raw.no_speech_prob),
                words=[
                    Word(
                        start=round(float(w.start), 3),
                        end=round(float(w.end), 3),
                        word=w.word,
                        probability=float(w.probability),
                    )
                    for w in raw.words
                ]
                if raw.words
                else None,
            )
            segments.append(seg)
            log.debug("[%.1f → %.1f] %s", seg.start, seg.end, seg.text)
            if on_segment is not None:
                on_segment(seg)
            if duration > 60 and seg.end / duration >= next_report:
                log.info(
                    "Progress %3.0f%% (%.0f / %.0f s)", 100 * seg.end / duration, seg.end, duration
                )
                next_report = (int(10 * seg.end / duration) + 1) / 10
        return Transcript(
            segments=segments,
            language=info.language,
            language_probability=round(float(info.language_probability), 4),
            duration=round(float(info.duration), 3),
        )

    def close(self) -> None:
        self._model = None
        self._pipeline = None
