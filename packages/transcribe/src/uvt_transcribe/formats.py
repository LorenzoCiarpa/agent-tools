"""Writers for txt / srt / vtt / json and a WebVTT reader (for YouTube subtitles)."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from uvt_core.jobs import Artifact
from uvt_core.output import dumps
from uvt_transcribe.models import FORMATS, Format, Segment, TranscriptionResult

_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{dev}{i}" for dev in ("COM", "LPT") for i in range(1, 10)
}


def safe_filename(name: str, max_length: int = 120) -> str:
    """A file name that is valid on Windows, macOS and Linux."""
    cleaned = _INVALID_FILENAME.sub("_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(". ")
    if cleaned.split(".")[0].upper() in _WINDOWS_RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned[:max_length].rstrip(". ") or "transcript"


def parse_formats(values: Iterable[str] | None) -> list[Format] | None:
    """Accept repeated and comma-separated values: ``--format txt,srt --format vtt``.

    ``none`` means "write no files" (the result is still available with --json).
    """
    if values is None:
        return None
    out: list[Format] = []
    for value in values:
        for part in str(value).split(","):
            fmt = part.strip().lower()
            if not fmt:
                continue
            if fmt == "none":
                return []
            if fmt not in FORMATS:
                raise ValueError(f"unknown format {fmt!r} (choose from {', '.join(FORMATS)}, none)")
            if fmt not in out:
                out.append(fmt)  # type: ignore[arg-type]
    return out


def format_timestamp(seconds: float, *, decimal: str = ",", always_hours: bool = True) -> str:
    millis = max(0, round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    if hours or always_hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{millis:03d}"
    return f"{minutes:02d}:{secs:02d}{decimal}{millis:03d}"


def to_txt(segments: Sequence[Segment]) -> str:
    return "".join(f"{s.text.strip()}\n" for s in segments if s.text.strip())


def to_srt(segments: Sequence[Segment]) -> str:
    blocks = []
    for i, seg in enumerate((s for s in segments if s.text.strip()), start=1):
        start, end = format_timestamp(seg.start), format_timestamp(seg.end)
        blocks.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n")
    return "\n".join(blocks)


def to_vtt(segments: Sequence[Segment]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        if not seg.text.strip():
            continue
        start = format_timestamp(seg.start, decimal=".")
        end = format_timestamp(seg.end, decimal=".")
        lines += [f"{start} --> {end}", seg.text.strip(), ""]
    return "\n".join(lines)


def to_json(result: TranscriptionResult) -> str:
    """The ``.json`` output file: transcript, metadata and timestamped segments.

    Unset fields are omitted to keep files small (the ``JobResult`` printed by
    ``--json`` keeps the full, stable schema).
    """
    payload = {
        "text": result.text,
        "language": result.language,
        "duration": result.duration,
        "backend": result.backend,
        "model": result.model,
        "from_subtitles": result.from_subtitles,
        "source": result.source.model_dump(mode="json", exclude_none=True),
        "segments": [s.model_dump(mode="json", exclude_none=True) for s in result.segments],
    }
    return dumps(payload, indent=2) + "\n"


def write_outputs(
    result: TranscriptionResult, formats: Sequence[Format], output_dir: Path, basename: str
) -> list[Artifact]:
    """Write the requested formats as ``<output_dir>/<basename>.<fmt>``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for fmt in formats:
        if fmt == "txt":
            content = to_txt(result.segments)
        elif fmt == "srt":
            content = to_srt(result.segments)
        elif fmt == "vtt":
            content = to_vtt(result.segments)
        else:
            content = to_json(result)
        path = output_dir / f"{basename}.{fmt}"
        path.write_text(content, encoding="utf-8", newline="\n")
        artifacts.append(Artifact.from_path(fmt, path))
    return artifacts


_VTT_TIME = re.compile(
    r"(?P<start>(?:\d+:)?\d{1,2}:\d{2}[.,]\d{3})\s+-->\s+(?P<end>(?:\d+:)?\d{1,2}:\d{2}[.,]\d{3})"
)
_TAG = re.compile(r"<[^>]+>")


def _parse_time(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2]) if len(parts) >= 2 else 0
    hours = int(parts[-3]) if len(parts) >= 3 else 0
    return hours * 3600 + minutes * 60 + seconds


def parse_vtt(text: str) -> list[Segment]:
    """Parse WebVTT (also SRT-ish) cues into segments.

    Strips inline tags and drops lines repeated from the previous cue, which is
    how "rolling" captions repeat text.
    """
    segments: list[Segment] = []
    previous_lines: list[str] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    for block in blocks:
        lines = block.strip("\n").split("\n")
        for i, line in enumerate(lines):
            match = _VTT_TIME.search(line)
            if not match:
                continue
            body = [html.unescape(_TAG.sub("", t)).strip() for t in lines[i + 1 :]]
            body = [t for t in body if t]
            fresh = [t for t in body if t not in previous_lines]
            previous_lines = body
            if fresh:
                segments.append(
                    Segment(
                        id=len(segments),
                        start=_parse_time(match["start"]),
                        end=_parse_time(match["end"]),
                        text=" ".join(fresh),
                    )
                )
            break
    return segments
