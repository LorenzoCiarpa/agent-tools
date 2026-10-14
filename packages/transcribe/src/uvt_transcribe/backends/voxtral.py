"""Local Voxtral (Mistral) backend via Hugging Face transformers + torch.

Install the extra (torch build chosen for the local GPU by uv):

    uv tool install "uvt-transcribe[voxtral] @ git+...#subdirectory=packages/transcribe" \
        --torch-backend auto

RTX 50xx (Blackwell, sm_120) needs a CUDA >= 12.8 torch build (cu128 or
newer); ``--torch-backend auto`` picks one from the driver version.

Voxtral returns text only (no timestamps), so the audio is cut into chunks of
at most ``chunk_seconds`` (default 30) on pauses found by the Silero VAD, and
each chunk becomes one segment with its start/end time.

Options (``--backend-option``): ``chunk_seconds``, ``max_new_tokens``,
``dtype`` (bfloat16/float16/float32).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from uvt_core.device import DeviceInfo
from uvt_core.errors import InvalidInputError, MissingDependencyError, ToolError
from uvt_core.logging import get_logger
from uvt_transcribe.audio import load_audio
from uvt_transcribe.backends.base import (
    SAMPLE_RATE,
    SegmentCallback,
    TranscriptionBackend,
    install_hint,
    quiet_hf_hub,
)
from uvt_transcribe.models import Segment, Transcript

if TYPE_CHECKING:
    import numpy as np

log = get_logger("transcribe.voxtral")

DEFAULT_MODEL = "mistralai/Voxtral-Mini-3B-2507"
_OPTIONS = {"chunk_seconds", "max_new_tokens", "dtype"}


def speech_chunks(
    samples: np.ndarray, max_seconds: float, *, vad: bool = True
) -> list[tuple[int, int]]:
    """Split audio into ``(start, end)`` sample ranges of at most ``max_seconds``,
    cutting on pauses when ``vad`` is on (silence-only audio gives no chunks)."""
    limit = int(max_seconds * SAMPLE_RATE)
    if not vad:
        return [(i, min(i + limit, len(samples))) for i in range(0, len(samples), limit)]
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    stamps = get_speech_timestamps(
        samples,
        VadOptions(
            min_silence_duration_ms=500, speech_pad_ms=200, max_speech_duration_s=max_seconds
        ),
    )
    chunks: list[tuple[int, int]] = []
    for stamp in stamps:
        start, end = stamp["start"], stamp["end"]
        if chunks and end - chunks[-1][0] <= limit:
            chunks[-1] = (chunks[-1][0], end)
        else:
            chunks.append((start, end))
    return chunks


class VoxtralBackend(TranscriptionBackend):
    name: ClassVar[str] = "voxtral"
    description: ClassVar[str] = "Mistral Voxtral, local, with transformers + torch."

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        unknown = set(self.options) - _OPTIONS
        if unknown:
            raise InvalidInputError(
                f"Unknown voxtral option(s): {', '.join(sorted(unknown))}.",
                hint=f"Valid keys: {', '.join(sorted(_OPTIONS))}.",
            )
        self._model: Any = None
        self._processor: Any = None
        self._device = "cpu"
        self._dtype_name = "float32"
        self._model_id = str(
            self.requested_model or self.config.get("voxtral_model") or DEFAULT_MODEL
        )
        self.chunk_seconds = float(self.options.get("chunk_seconds", 30))
        self.max_new_tokens = self.options.get("max_new_tokens")

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import mistral_common  # noqa: F401 - required by the Voxtral processor
            import torch
            from transformers import AutoProcessor, VoxtralForConditionalGeneration
        except ImportError as exc:
            raise MissingDependencyError(
                f"The voxtral backend needs torch, transformers and mistral-common ({exc}).",
                hint=install_hint("voxtral", torch=True),
            ) from exc
        quiet_hf_hub()
        self._device = self._pick_device(torch)
        default_dtype = {"cuda": "bfloat16", "mps": "float16"}.get(self._device, "float32")
        self._dtype_name = str(
            self.options.get("dtype") or self.requested_compute_type or default_dtype
        )
        dtype = getattr(torch, self._dtype_name, None)
        if dtype is None:
            raise InvalidInputError(f"Unknown torch dtype {self._dtype_name!r}.")

        log.info("Loading %s on %s (%s)…", self._model_id, self._device, self._dtype_name)
        t0 = time.perf_counter()
        try:
            self._processor = AutoProcessor.from_pretrained(self._model_id)
            model = VoxtralForConditionalGeneration.from_pretrained(self._model_id, dtype=dtype)
            self._model = model.to(self._device).eval()
        except OSError as exc:
            raise ToolError(
                f"Could not load {self._model_id}: {exc}",
                hint="Check the model id and your network; the first run downloads ~9 GB.",
            ) from exc
        log.info("Model ready in %.1f s", time.perf_counter() - t0)

    def _pick_device(self, torch: Any) -> str:
        requested = (self.requested_device or "auto").lower()
        has_cuda = torch.cuda.is_available()
        has_mps = (
            getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        )
        if requested == "auto":
            requested = "cuda" if has_cuda else "mps" if has_mps else "cpu"
        if requested == "cuda":
            if not has_cuda:
                raise ToolError(
                    "CUDA requested but this torch build cannot use it "
                    f"(torch {torch.__version__}, CUDA {torch.version.cuda}).",
                    hint="Reinstall with a CUDA torch: add --torch-backend auto (RTX 50xx need "
                    "cu128 or newer) to the uv tool install command, or use --device cpu.",
                )
            major, minor = torch.cuda.get_device_capability()
            arch = f"sm_{major}{minor}"
            supported = torch.cuda.get_arch_list()
            if supported and arch not in supported and f"compute_{major}{minor}" not in supported:
                newest = max((a for a in supported if a.startswith("sm_")), default="?")
                if newest == "?" or int(newest[3:]) < major * 10 + minor:
                    raise ToolError(
                        f"This torch build ({torch.__version__}) has no kernels for {arch} "
                        f"(supports up to {newest}).",
                        hint="Reinstall with --torch-backend cu128 (or newer) / auto.",
                    )
        if requested == "mps" and not has_mps:
            raise ToolError(
                "Apple GPU (mps) requested but not available.", hint="Use --device cpu."
            )
        return requested

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def device_info(self) -> DeviceInfo:
        from uvt_core import device as dev

        kind = self._device if self._device in {"cuda", "mps", "cpu"} else "cpu"
        info = dev.describe(kind, compute_type=self._dtype_name)  # type: ignore[arg-type]
        info.details["engine"] = "transformers"
        return info

    def _generate(self, audio: np.ndarray, language: str | None) -> str:
        import torch

        inputs = self._processor.apply_transcription_request(
            audio=audio,
            model_id=self._model_id,
            language=language,
            sampling_rate=SAMPLE_RATE,
            format="WAV",
        )
        inputs = inputs.to(self._device, dtype=self._model.dtype)
        seconds = len(audio) / SAMPLE_RATE
        max_new = int(self.max_new_tokens or 64 + seconds * 8)  # ~3 tokens/s of speech, plus margin
        with torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        return self._processor.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)[0]

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
        samples = load_audio(Path(audio)) if isinstance(audio, Path | str) else audio
        duration = len(samples) / SAMPLE_RATE
        if initial_prompt:
            log.debug("voxtral ignores initial_prompt (no prompt support in transcription mode)")
        segments: list[Segment] = []
        chunks = speech_chunks(samples, self.chunk_seconds, vad=vad)
        for n, (start, end) in enumerate(chunks, start=1):
            text = self._generate(samples[start:end], language).strip()
            if text:
                seg = Segment(
                    id=len(segments),
                    start=round(start / SAMPLE_RATE, 3),
                    end=round(end / SAMPLE_RATE, 3),
                    text=text,
                )
                segments.append(seg)
                if on_segment is not None:
                    on_segment(seg)
            if len(chunks) > 3:
                log.info("Chunk %d/%d done", n, len(chunks))
        return Transcript(segments=segments, language=language, duration=round(duration, 3))

    def close(self) -> None:
        if self._model is not None and self._device == "cuda":
            import torch

            self._model = None
            torch.cuda.empty_cache()
        self._model = None
