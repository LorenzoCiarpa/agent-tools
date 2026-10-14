"""User configuration.

A single TOML file in the user config dir (``uvt/config.toml``, located with
platformdirs), one table per tool::

    log_level = "INFO"

    [transcribe]
    model_gpu = "large-v3-turbo"
    model_cpu = "small"

Every key can be overridden with an environment variable:
``UVT_<KEY>`` for top-level keys and ``UVT_<SECTION>__<KEY>`` (double
underscore) for keys inside a table, e.g. ``UVT_TRANSCRIBE__MODEL_CPU=base``.
Values are parsed as TOML literals when possible (``true``, ``3``, ``["a"]``),
otherwise kept as strings. ``UVT_CONFIG_FILE`` points to a different file.

Precedence (highest first): CLI flags / job params > env vars > config file >
built-in defaults.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import platformdirs

from uvt_core.errors import InvalidInputError

APP_NAME = "uvt"
ENV_PREFIX = "UVT_"
CONFIG_FILE_ENV = "UVT_CONFIG_FILE"
SECTION_SEP = "__"
_RESERVED_ENV = {CONFIG_FILE_ENV}


def config_path(env: Mapping[str, str] | None = None) -> Path:
    """Path of the config file (it may not exist)."""
    env = os.environ if env is None else env
    if override := env.get(CONFIG_FILE_ENV):
        return Path(override).expanduser()
    return platformdirs.user_config_path(APP_NAME, appauthor=False) / "config.toml"


def parse_env_value(raw: str) -> Any:
    """Parse an env var value as a TOML literal, falling back to the raw string."""
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def env_overrides(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Collect ``UVT_*`` variables into a nested dict mirroring the TOML layout."""
    env = os.environ if env is None else env
    out: dict[str, Any] = {}
    for key, raw in env.items():
        upper = key.upper()
        if not upper.startswith(ENV_PREFIX) or upper in _RESERVED_ENV:
            continue
        name = upper[len(ENV_PREFIX) :].lower()
        section, sep, sub = name.partition(SECTION_SEP)
        if not section:
            continue
        value = parse_env_value(raw)
        if sep and sub:
            table = out.get(section)
            if not isinstance(table, dict):
                table = out[section] = {}
            table[sub] = value
        else:
            out[name] = value
    return out


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = dict(value) if isinstance(value, Mapping) else value
    return merged


def load_config(path: Path | None = None, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Load the config file (if present) merged with ``UVT_*`` env overrides."""
    path = path or config_path(env)
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
            raise InvalidInputError(
                f"Invalid config file {path}: {exc}",
                hint=f"Fix the TOML syntax, or set {CONFIG_FILE_ENV} to another file.",
            ) from exc
    return _deep_merge(data, env_overrides(env))


def tool_config(
    section: str, path: Path | None = None, env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Return the ``[section]`` table of the merged config (empty if missing)."""
    table = load_config(path, env).get(section, {})
    if not isinstance(table, dict):
        raise InvalidInputError(
            f"Config key '{section}' must be a table, got {type(table).__name__}.",
            hint=f"Use a [{section}] table in {config_path(env)}.",
        )
    return table
