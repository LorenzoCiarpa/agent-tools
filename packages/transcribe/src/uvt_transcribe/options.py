"""Typed parameters of the transcription tools and their resolution.

Every field defaults to ``None`` = "not given", so the precedence is:
explicit params (CLI / JobSpec) > ``UVT_TRANSCRIBE__*`` env > ``[transcribe]``
config table > built-in default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from uvt_core.config import config_path, tool_config
from uvt_core.errors import InvalidInputError
from uvt_core.jobs import format_validation_error, input_field
from uvt_transcribe.backends import DEFAULT_BACKEND, TranscriptionBackend, create_backend
from uvt_transcribe.formats import parse_formats
from uvt_transcribe.models import Format

CONFIG_SECTION = "transcribe"
DeviceChoice = Literal["auto", "cuda", "cpu", "mps"]


class TranscribeOptions(BaseModel):
    """Options shared by all transcription commands."""

    model_config = ConfigDict(extra="forbid")

    backend: str | None = Field(
        default=None, description="faster-whisper (default), voxtral or api."
    )
    model: str | None = Field(
        default=None,
        description="Model name/path. Default depends on backend and device "
        "(faster-whisper: large-v3-turbo on CUDA, small on CPU).",
    )
    device: DeviceChoice | None = Field(default=None, description="Compute device (default auto).")
    compute_type: str | None = Field(
        default=None, description="Precision, e.g. float16, int8 (default per device)."
    )
    language: str | None = Field(
        default=None, description="ISO 639-1 code such as 'it' or 'en'; None/'auto' = detect."
    )
    formats: list[Format] | None = Field(
        default=None, description="Output files to write: txt, srt, vtt, json (default txt)."
    )
    output_dir: Path | None = Field(default=None, description="Default: current directory.")
    name: str | None = Field(default=None, description="Base name of the output files.")
    vad: bool | None = Field(default=None, description="Skip silence with Silero VAD (default on).")
    beam_size: int | None = Field(default=None, ge=1, le=20)
    initial_prompt: str | None = Field(
        default=None, description="Text that primes the model (names, jargon, style)."
    )
    backend_options: dict[str, Any] = Field(
        default_factory=dict, description="Backend-specific settings (KEY=VALUE)."
    )

    @field_validator("language", mode="before")
    @classmethod
    def _language(cls, value: Any) -> Any:
        # "auto" is kept as an explicit choice (it must beat a configured language)
        # and turned into None by resolve_options().
        if isinstance(value, str):
            value = value.strip().lower()
            return value or None
        return value

    @field_validator("formats", mode="before")
    @classmethod
    def _formats(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list | tuple):
            return parse_formats(value)
        return value


class TranscribeFileParams(TranscribeOptions):
    media: Path = input_field(description="Local audio or video file.")


class YtTranscribeParams(TranscribeOptions):
    url: str = Field(description="Video URL (YouTube or any site yt-dlp supports).")
    prefer_subs: bool = Field(
        default=False,
        description="Use manually created subtitles when available instead of transcribing.",
    )
    sub_langs: list[str] | None = Field(
        default=None,
        description="Subtitle languages to look for, in order (default: --language, "
        "then the video language, then en).",
    )
    keep_audio: bool = Field(default=False, description="Keep the downloaded audio file.")
    cookies_from_browser: str | None = Field(
        default=None, description="Pass browser cookies to yt-dlp, e.g. 'firefox' or 'chrome'."
    )


class LiveTranscribeParams(TranscribeOptions):
    mic: str | None = Field(
        default=None, description="Microphone index or (part of) its name; default device if None."
    )
    max_duration: float | None = Field(
        default=None, gt=0, description="Stop after this many seconds of recording."
    )
    stop_file: Path | None = Field(
        default=None, description="Stop as soon as this file exists (works on every OS)."
    )
    final_pass: bool = Field(
        default=False, description="Re-transcribe the full recording at the end."
    )
    min_silence: float = Field(
        default=0.6, gt=0, le=10, description="Pause (s) that closes a segment."
    )
    max_segment: float = Field(
        default=20.0, ge=2, le=60, description="Longest segment (s) before a forced cut."
    )


P = TypeVar("P", bound=TranscribeOptions)


def resolve_options(params: P) -> tuple[P, dict[str, Any]]:
    """Fill unset options from config/env/defaults. Returns (params, config table)."""
    cfg = tool_config(CONFIG_SECTION)
    shared = set(TranscribeOptions.model_fields) - {"backend_options"}
    from_config = {k: v for k, v in cfg.items() if k in shared}
    explicit = params.model_dump(exclude_none=True)
    merged: dict[str, Any] = {**from_config, **explicit}
    merged.setdefault("backend", DEFAULT_BACKEND)
    merged.setdefault("device", "auto")
    merged.setdefault("formats", ["txt"])
    merged.setdefault("vad", True)
    try:
        resolved = type(params).model_validate(merged)
    except ValidationError as exc:
        raise InvalidInputError(
            f"Invalid transcription settings: {format_validation_error(exc)}",
            hint=f"Check the [{CONFIG_SECTION}] table in {config_path()} and UVT_TRANSCRIBE__* "
            "environment variables.",
        ) from exc
    if resolved.language == "auto":
        resolved.language = None
    return resolved, cfg


def make_backend(opts: TranscribeOptions, cfg: dict[str, Any]) -> TranscriptionBackend:
    return create_backend(
        opts.backend or DEFAULT_BACKEND,
        model=opts.model,
        device=opts.device or "auto",
        compute_type=opts.compute_type,
        config=cfg,
        options=opts.backend_options,
    )


def output_dir(opts: TranscribeOptions) -> Path:
    return (opts.output_dir or Path.cwd()).expanduser()


def formats(opts: TranscribeOptions) -> list[Format]:
    return list(opts.formats if opts.formats is not None else ["txt"])
