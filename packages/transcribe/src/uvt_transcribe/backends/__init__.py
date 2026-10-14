"""Backend registry: name -> implementation, imported lazily on first use."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from uvt_core.errors import InvalidInputError
from uvt_transcribe.backends.base import TranscriptionBackend

DEFAULT_BACKEND = "faster-whisper"

BackendFactory = Callable[..., TranscriptionBackend]

# "module:attribute" strings keep heavy imports out of `--help`.
_REGISTRY: dict[str, str | BackendFactory] = {
    "faster-whisper": "uvt_transcribe.backends.faster_whisper:FasterWhisperBackend",
    "voxtral": "uvt_transcribe.backends.voxtral:VoxtralBackend",
    "api": "uvt_transcribe.backends.mistral_api:MistralApiBackend",
}


def register_backend(name: str, factory: str | BackendFactory) -> None:
    """Register (or replace) a backend. ``factory`` is a class/callable or a
    ``"module:attribute"`` string."""
    _REGISTRY[name] = factory


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def _resolve(name: str) -> BackendFactory:
    try:
        target = _REGISTRY[name]
    except KeyError:
        raise InvalidInputError(
            f"Unknown backend {name!r}.",
            hint=f"Choose one of: {', '.join(available_backends())}.",
        ) from None
    if isinstance(target, str):
        module_name, _, attr = target.partition(":")
        target = getattr(importlib.import_module(module_name), attr)
    return target


def create_backend(name: str, **kwargs: Any) -> TranscriptionBackend:
    """Instantiate a backend by name (the model is loaded later by ``load()``)."""
    return _resolve(name)(**kwargs)


__all__ = [
    "DEFAULT_BACKEND",
    "TranscriptionBackend",
    "available_backends",
    "create_backend",
    "register_backend",
]
