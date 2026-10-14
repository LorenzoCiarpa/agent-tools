"""The ``TranscriptionBackend`` interface.

A backend turns audio into a ``Transcript``. Audio is either a path to a media
file or a mono float32 numpy array at 16 kHz (used by live transcription).

Backends must not import their heavy dependencies at module import time: do
it in ``load()`` and raise ``MissingDependencyError`` with the install command,
so ``--help`` and the other backends keep working.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from uvt_core.device import DeviceInfo
from uvt_core.logging import get_logger
from uvt_transcribe.models import Segment, Transcript

if TYPE_CHECKING:
    import numpy as np

SAMPLE_RATE = 16_000
SegmentCallback = Callable[[Segment], None]

INSTALL_SPEC = (
    "uvt-transcribe[{extra}] @ "
    "git+https://github.com/LorenzoCiarpa/agent-tools#subdirectory=packages/transcribe"
)


def quiet_hf_hub() -> None:
    """Silence huggingface_hub's own stderr handler (e.g. the HF_TOKEN rate-limit
    notice) unless debugging; download errors still surface as exceptions."""
    if not get_logger().isEnabledFor(logging.DEBUG):
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def install_hint(extra: str, *, torch: bool = False) -> str:
    cmd = f'uv tool install --reinstall "{INSTALL_SPEC.format(extra=extra)}"'
    if torch:
        cmd += " --torch-backend auto"
    return f"Install the '{extra}' extra: {cmd}"


class TranscriptionBackend(ABC):
    """Base class for transcription engines, selected with ``--backend NAME``."""

    name: ClassVar[str]
    #: Short description shown in ``--help`` / docs.
    description: ClassVar[str] = ""

    def __init__(
        self,
        *,
        model: str | None = None,
        device: str = "auto",
        compute_type: str | None = None,
        config: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """``config`` is the merged ``[transcribe]`` config table; ``options`` are
        backend-specific overrides from ``--backend-option KEY=VALUE``."""
        self.requested_model = model
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.config = dict(config or {})
        self.options = dict(options or {})

    @abstractmethod
    def load(self) -> None:
        """Import dependencies and load the model. Idempotent."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def device_info(self) -> DeviceInfo: ...

    @abstractmethod
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
        """Transcribe ``audio``. ``language=None`` means auto-detect.

        ``on_segment`` is called for every segment as soon as it is available.
        """

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources (GPU memory, HTTP clients)."""

    def __enter__(self) -> TranscriptionBackend:
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
