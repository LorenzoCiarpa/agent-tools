"""Exit codes and the exception hierarchy shared by every tool.

Tools raise these exceptions from their *logic* functions; the CLI runner turns
them into a documented exit code, a clear message on stderr and (with
``--json``) a ``JobResult`` with ``ok: false``.
"""

from __future__ import annotations

import platform
from collections.abc import Mapping
from enum import IntEnum


class ExitCode(IntEnum):
    """Process exit codes. Documented in every tool's ``--help``."""

    OK = 0
    ERROR = 1
    INVALID_INPUT = 2
    INTERRUPTED = 130


class ToolError(Exception):
    """An expected failure with an actionable message.

    ``hint`` should tell the user (or agent) what to do next.
    """

    exit_code: ExitCode = ExitCode.ERROR
    error_type: str = "error"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class InvalidInputError(ToolError):
    """Bad parameters, missing/unreadable input files, malformed job specs."""

    exit_code = ExitCode.INVALID_INPUT
    error_type = "invalid_input"


class MissingDependencyError(ToolError):
    """An optional Python extra or a system library/program is missing."""

    error_type = "missing_dependency"

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        os_hints: Mapping[str, str] | None = None,
    ) -> None:
        """``os_hints`` maps ``platform.system()`` values (Linux/Darwin/Windows) to
        install instructions; the one for the current OS is appended to ``hint``."""
        parts = [hint] if hint else []
        if os_hints:
            system = platform.system()
            if system in os_hints:
                parts.append(os_hints[system])
            else:
                parts.extend(f"{name}: {text}" for name, text in os_hints.items())
        super().__init__(message, hint="\n".join(parts) or None)


class Interrupted(ToolError):
    """The user or a signal stopped the job before it completed."""

    exit_code = ExitCode.INTERRUPTED
    error_type = "interrupted"
