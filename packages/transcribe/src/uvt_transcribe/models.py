"""Serializable data types shared by backends, formats and the public API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from uvt_core.jobs import ToolOutput

Format = Literal["txt", "srt", "vtt", "json"]
FORMATS: tuple[Format, ...] = ("txt", "srt", "vtt", "json")


class Word(BaseModel):
    start: float
    end: float
    word: str
    probability: float | None = None


class Segment(BaseModel):
    """A piece of transcript with timestamps in seconds from the start of the audio."""

    id: int
    start: float
    end: float
    text: str
    words: list[Word] | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    speaker: str | None = None


class Transcript(BaseModel):
    """What a backend returns for one piece of audio."""

    segments: list[Segment] = Field(default_factory=list)
    language: str | None = None
    language_probability: float | None = None
    duration: float | None = Field(default=None, description="Audio duration in seconds.")

    @property
    def text(self) -> str:
        return join_text(self.segments)


def join_text(segments: list[Segment]) -> str:
    return " ".join(s.text.strip() for s in segments if s.text.strip())


class SourceInfo(BaseModel):
    """Where the audio came from (file, YouTube video or microphone)."""

    kind: Literal["file", "youtube", "url", "microphone"]
    path: str | None = None
    url: str | None = None
    id: str | None = None
    title: str | None = None
    channel: str | None = None
    uploader: str | None = None
    upload_date: str | None = None
    duration: float | None = None
    mic: str | None = None


class TranscriptionResult(ToolOutput):
    """Result of ``transcribe_file`` / ``yt_transcribe`` / ``live_transcribe``.

    ``text`` holds the full transcript and ``artifacts`` the files written
    (``name`` is the format, or ``audio`` / ``live_log`` / ``subtitles``).
    """

    source: SourceInfo
    backend: str | None = None
    model: str | None = None
    language: str | None = None
    language_probability: float | None = None
    duration: float | None = None
    segments: list[Segment] = Field(default_factory=list)
    from_subtitles: bool = Field(
        default=False, description="True if YouTube subtitles were used instead of a model."
    )
