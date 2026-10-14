"""Shared steps of the transcription tools (backend run, result assembly)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from uvt_core.errors import InvalidInputError
from uvt_core.logging import get_logger
from uvt_transcribe.backends import TranscriptionBackend
from uvt_transcribe.backends.base import SegmentCallback
from uvt_transcribe.formats import safe_filename, write_outputs
from uvt_transcribe.models import SourceInfo, Transcript, TranscriptionResult
from uvt_transcribe.options import (
    TranscribeFileParams,
    TranscribeOptions,
    formats,
    make_backend,
    output_dir,
    resolve_options,
)

if TYPE_CHECKING:
    import numpy as np

log = get_logger("transcribe")


def build_result(
    transcript: Transcript,
    backend: TranscriptionBackend | None,
    source: SourceInfo,
    *,
    from_subtitles: bool = False,
) -> TranscriptionResult:
    return TranscriptionResult(
        text=transcript.text,
        device=backend.device_info if backend else None,
        source=source,
        backend=backend.name if backend else None,
        model=backend.model_name if backend else None,
        language=transcript.language,
        language_probability=transcript.language_probability,
        duration=transcript.duration,
        segments=transcript.segments,
        from_subtitles=from_subtitles,
    )


def run_backend(
    opts: TranscribeOptions,
    cfg: dict[str, Any],
    audio: Path | np.ndarray,
    *,
    backend: TranscriptionBackend | None,
    on_segment: SegmentCallback | None,
) -> tuple[Transcript, TranscriptionBackend]:
    """Load (if needed) a backend and transcribe ``audio`` with the shared options."""
    owned = backend is None
    engine = backend or make_backend(opts, cfg)
    try:
        engine.load()
        log.info(
            "Transcribing with %s (%s on %s)",
            engine.name,
            engine.model_name,
            engine.device_info.kind,
        )
        transcript = engine.transcribe(
            audio,
            language=opts.language,
            initial_prompt=opts.initial_prompt,
            vad=opts.vad if opts.vad is not None else True,
            beam_size=opts.beam_size,
            on_segment=on_segment,
        )
    finally:
        if owned:
            engine.close()
    return transcript, engine


def transcribe_file(
    params: TranscribeFileParams,
    *,
    on_segment: SegmentCallback | None = None,
    backend: TranscriptionBackend | None = None,
) -> TranscriptionResult:
    """Transcribe a local audio/video file and write the requested formats."""
    opts, cfg = resolve_options(params)
    media = opts.media.expanduser()
    if not media.is_file():
        raise InvalidInputError(
            f"Input file not found: {media}", hint="Pass the path of an existing audio/video file."
        )
    transcript, engine = run_backend(opts, cfg, media, backend=backend, on_segment=on_segment)
    result = build_result(transcript, engine, SourceInfo(kind="file", path=str(media.resolve())))
    result.source.duration = transcript.duration
    basename = safe_filename(opts.name or media.stem)
    result.artifacts = write_outputs(result, formats(opts), output_dir(opts), basename)
    return result
