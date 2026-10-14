"""Logging to stderr. stdout is reserved for results (text or ``--json``)."""

from __future__ import annotations

import logging
import os
import sys

LOGGER_NAME = "uvt"
LOG_LEVEL_ENV = "UVT_LOG_LEVEL"


class _StderrHandler(logging.StreamHandler):
    """Always writes to the *current* ``sys.stderr`` (tests and redirections swap it)."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property  # type: ignore[override]
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, _value) -> None:
        pass


_handler: logging.Handler | None = None


def setup_logging(verbose: int = 0, quiet: bool = False) -> logging.Logger:
    """Configure logging once per process.

    * default: our messages at INFO, third-party at WARNING
    * ``-v``: ours at DEBUG, third-party at INFO;  ``-vv``: everything at DEBUG
    * ``-q``: warnings and errors only
    * ``UVT_LOG_LEVEL`` (e.g. ``DEBUG``) applies when no flag is given.
    """
    global _handler
    if quiet:
        ours, theirs = logging.WARNING, logging.WARNING
    elif verbose >= 2:
        ours, theirs = logging.DEBUG, logging.DEBUG
    elif verbose == 1:
        ours, theirs = logging.DEBUG, logging.INFO
    else:
        env_level = os.environ.get(LOG_LEVEL_ENV, "").upper()
        ours = logging.getLevelNamesMapping().get(env_level, logging.INFO)
        theirs = logging.WARNING

    root = logging.getLogger()
    if _handler is not None:
        root.removeHandler(_handler)
    _handler = _StderrHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(_handler)
    root.setLevel(theirs)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(ours)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Logger in the ``uvt`` namespace, e.g. ``get_logger("transcribe")``."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)
