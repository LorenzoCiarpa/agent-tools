"""Mistral cloud transcription (Voxtral Mini Transcribe) for machines without a GPU.

Needs ``MISTRAL_API_KEY`` in the environment (or the env var named by the
``api_key_env`` config key). Audio is decoded locally and uploaded as 16 kHz
mono WAV in chunks of at most ``chunk_seconds`` (default 1800), so any input
format works and long files stay under the API limits.

The API cannot return timestamps when a language is given; in that case each
chunk becomes one segment. Options (``--backend-option``): ``timestamps``
(force segment timestamps, ignoring --language), ``diarize``, ``context_bias``
(comma-separated words), ``temperature``, ``chunk_seconds``, ``endpoint``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from uvt_core.device import DeviceInfo
from uvt_core.errors import InvalidInputError, MissingDependencyError, ToolError
from uvt_core.logging import get_logger
from uvt_transcribe.audio import load_audio, wav_bytes
from uvt_transcribe.backends.base import SAMPLE_RATE, SegmentCallback, TranscriptionBackend
from uvt_transcribe.models import Segment, Transcript

if TYPE_CHECKING:
    import numpy as np

log = get_logger("transcribe.api")

DEFAULT_MODEL = "voxtral-mini-latest"
DEFAULT_ENDPOINT = "https://api.mistral.ai/v1/audio/transcriptions"
_OPTIONS = {"timestamps", "diarize", "context_bias", "temperature", "chunk_seconds", "endpoint"}
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class MistralApiBackend(TranscriptionBackend):
    name: ClassVar[str] = "api"
    description: ClassVar[str] = "Mistral cloud API (voxtral-mini-latest), needs MISTRAL_API_KEY."

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        unknown = set(self.options) - _OPTIONS
        if unknown:
            raise InvalidInputError(
                f"Unknown api option(s): {', '.join(sorted(unknown))}.",
                hint=f"Valid keys: {', '.join(sorted(_OPTIONS))}.",
            )
        self._model = str(self.requested_model or self.config.get("api_model") or DEFAULT_MODEL)
        self._endpoint = str(
            self.options.get("endpoint") or self.config.get("api_endpoint") or DEFAULT_ENDPOINT
        )
        self.chunk_seconds = float(self.options.get("chunk_seconds", 1800))
        self._client: Any = None
        self.transport: Any = None  # tests inject an httpx.MockTransport here

    def load(self) -> None:
        if self._client is not None:
            return
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - base dependency
            raise MissingDependencyError(f"httpx is not importable: {exc}") from exc
        env_name = str(self.config.get("api_key_env") or "MISTRAL_API_KEY")
        key = os.environ.get(env_name, "").strip()
        if not key:
            raise ToolError(
                f"No API key: set the {env_name} environment variable.",
                hint="Create a key at https://console.mistral.ai/ and export "
                f"{env_name}=... (or use --backend faster-whisper to run locally).",
            )
        if self.requested_device not in (None, "auto"):
            log.info("--device is ignored by the api backend (it runs remotely).")
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(900.0, connect=30.0),
            transport=self.transport,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(kind="remote", name="Mistral API", details={"endpoint": self._endpoint})

    def _form(self, language: str | None) -> dict[str, Any]:
        form: dict[str, Any] = {"model": self._model}
        want_timestamps = _truthy(self.options.get("timestamps", language is None))
        if want_timestamps:
            if language:
                log.warning(
                    "The Mistral API cannot combine timestamps and language; "
                    "ignoring --language %s.",
                    language,
                )
            form["timestamp_granularities"] = ["segment"]
        elif language:
            form["language"] = language
        if _truthy(self.options.get("diarize", False)):
            form["diarize"] = "true"
        if self.options.get("context_bias"):
            words = [w.strip() for w in str(self.options["context_bias"]).split(",") if w.strip()]
            form["context_bias"] = words
        if "temperature" in self.options:
            form["temperature"] = str(self.options["temperature"])
        return form

    def _post(self, payload: bytes, form: dict[str, Any]) -> dict[str, Any]:
        import httpx

        for attempt in range(4):
            try:
                resp = self._client.post(
                    self._endpoint,
                    data=form,
                    files={"file": ("audio.wav", payload, "audio/wav")},
                )
            except httpx.HTTPError as exc:
                if attempt == 3:
                    raise ToolError(
                        f"Mistral API request failed: {exc}", hint="Check your network connection."
                    ) from exc
                time.sleep(2**attempt)
                continue
            if resp.status_code in _RETRY_STATUS and attempt < 3:
                delay = float(resp.headers.get("retry-after", 2**attempt) or 2**attempt)
                log.warning("Mistral API returned %s; retrying in %.0f s", resp.status_code, delay)
                time.sleep(min(delay, 30))
                continue
            if resp.status_code in {401, 403}:
                raise ToolError(
                    "The Mistral API rejected the API key.",
                    hint="Check MISTRAL_API_KEY and the key's permissions.",
                )
            if resp.status_code >= 400:
                raise ToolError(f"Mistral API error {resp.status_code}: {resp.text[:500]}")
            return resp.json()
        raise ToolError("Mistral API kept failing after retries.")  # pragma: no cover

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
        form = self._form(language)
        step = int(self.chunk_seconds * SAMPLE_RATE)
        segments: list[Segment] = []
        detected: str | None = language
        for offset in range(0, max(len(samples), 1), step):
            chunk = samples[offset : offset + step]
            if chunk.size == 0:
                break
            start_s = offset / SAMPLE_RATE
            body = self._post(wav_bytes(chunk), form)
            detected = detected or body.get("language")
            raw_segments = body.get("segments") or []
            if not raw_segments and (body.get("text") or "").strip():
                raw_segments = [
                    {"start": 0.0, "end": len(chunk) / SAMPLE_RATE, "text": body["text"]}
                ]
            for raw in raw_segments:
                text = str(raw.get("text", "")).strip()
                if not text:
                    continue
                seg = Segment(
                    id=len(segments),
                    start=round(start_s + float(raw.get("start") or 0.0), 3),
                    end=round(start_s + float(raw.get("end") or 0.0), 3),
                    text=text,
                    speaker=raw.get("speaker_id"),
                )
                segments.append(seg)
                if on_segment is not None:
                    on_segment(seg)
        return Transcript(segments=segments, language=detected, duration=round(duration, 3))

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
