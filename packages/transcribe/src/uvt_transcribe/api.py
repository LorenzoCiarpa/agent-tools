"""Public, typed API: one function per tool, ``params -> TranscriptionResult``.

The CLIs are thin wrappers around these functions; an MCP server or a remote
worker can call them directly. Pass an already loaded ``backend=`` to reuse a
model across calls (a long-lived worker should do that).
"""

from __future__ import annotations

from uvt_transcribe.backends import available_backends, create_backend, register_backend
from uvt_transcribe.live import list_microphones, live_transcribe
from uvt_transcribe.models import Segment, SourceInfo, Transcript, TranscriptionResult
from uvt_transcribe.options import LiveTranscribeParams, TranscribeFileParams, YtTranscribeParams
from uvt_transcribe.pipeline import transcribe_file
from uvt_transcribe.youtube import yt_transcribe

__all__ = [
    "LiveTranscribeParams",
    "Segment",
    "SourceInfo",
    "TranscribeFileParams",
    "Transcript",
    "TranscriptionResult",
    "YtTranscribeParams",
    "available_backends",
    "create_backend",
    "list_microphones",
    "live_transcribe",
    "register_backend",
    "transcribe_file",
    "yt_transcribe",
]
