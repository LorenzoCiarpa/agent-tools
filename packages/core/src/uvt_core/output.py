"""JSON output helpers and the stdout guard used by ``--json``.

With ``--json`` stdout must contain exactly one JSON document. Libraries
(yt-dlp, tqdm, C extensions) sometimes print to stdout, so while a job runs we
point file descriptor 1 at stderr and keep a private handle to the real
stdout for the final result.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from collections.abc import Iterator
from typing import Any, TextIO

from pydantic import BaseModel


def to_jsonable(obj: Any) -> Any:
    """Convert pydantic models (and nested containers of them) to JSON types."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    return obj


def dumps(obj: Any, *, indent: int | None = None) -> str:
    return json.dumps(to_jsonable(obj), ensure_ascii=False, indent=indent, default=str)


def emit_json(obj: Any, stream: TextIO | None = None) -> None:
    """Write one JSON document (+ newline) to ``stream`` (default: stdout)."""
    stream = stream or sys.stdout
    indent = 2 if _isatty(stream) else None
    stream.write(dumps(obj, indent=indent) + "\n")
    stream.flush()


def _isatty(stream: TextIO) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


def _is_fd1(stream: TextIO) -> bool:
    try:
        return stream.fileno() == 1
    except (AttributeError, ValueError, io.UnsupportedOperation):
        return False


@contextlib.contextmanager
def stdout_to_stderr(enabled: bool = True) -> Iterator[TextIO]:
    """Redirect everything written to stdout (Python *and* C level) to stderr.

    Yields a text stream connected to the real stdout. When disabled, yields
    ``sys.stdout`` and changes nothing.
    """
    if not enabled:
        yield sys.stdout
        return

    real_stdout = sys.stdout
    real_stdout.flush()
    sys.stderr.flush()
    try:
        # Always the process-level descriptors: C extensions write to fd 1
        # regardless of what sys.stdout currently is.
        saved_fd = os.dup(1)
        os.dup2(2, 1)
    except OSError:
        # No usable descriptors (e.g. pythonw on Windows): Python-level only.
        with contextlib.redirect_stdout(sys.stderr):
            yield real_stdout
        return

    try:
        if _is_fd1(real_stdout):
            private = os.fdopen(os.dup(saved_fd), "w", encoding="utf-8", newline="\n")
        else:
            # sys.stdout was already swapped (test runner, embedding app): keep it.
            private = None
        try:
            with contextlib.redirect_stdout(sys.stderr):
                yield private or real_stdout
        finally:
            if private is not None:
                private.flush()
                private.close()
            sys.stderr.flush()
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
