"""``JobSpec`` / ``JobResult``: the contract between callers and tools.

* A ``JobSpec`` fully describes one invocation: which tool, which version,
  scalar ``params`` and file/URL ``inputs``. ``inputs`` are kept apart from
  ``params`` because they are what a remote dispatcher must transfer.
* A ``JobResult`` is what ``--json`` prints: success flag, exit code, error,
  timings, host/device and produced artifacts.

Both are plain JSON. Bump ``SCHEMA_VERSION`` on breaking changes.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import platform
import socket
import sys
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from uvt_core.device import DeviceInfo
from uvt_core.errors import ExitCode, InvalidInputError

SCHEMA_VERSION = 1
INPUT_MARKER = "x-uvt-input"


def utcnow() -> datetime:
    return datetime.now(UTC)


def package_version(dist_name: str) -> str:
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return "0.0.0+unknown"


class InputRef(BaseModel):
    """A job input: a local ``path`` or a ``url`` (exactly one)."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    url: str | None = None
    sha256: str | None = Field(default=None, description="Optional integrity check.")
    size: int | None = None
    media_type: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> InputRef:
        if (self.path is None) == (self.url is None):
            raise ValueError("an input needs exactly one of 'path' or 'url'")
        return self

    @property
    def value(self) -> str:
        return self.path if self.path is not None else str(self.url)


class JobSpec(BaseModel):
    """One tool invocation, serializable to JSON (``--job job.json``)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tool: str = Field(description="Command name, e.g. 'transcribe-file'.")
    tool_version: str | None = Field(
        default=None,
        description="PEP 440 specifier the tool must satisfy, e.g. '>=0.1,<0.2'. "
        "A bare version means '=='.",
    )
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, InputRef] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Free-form caller metadata, ignored by tools."
    )


class Artifact(BaseModel):
    """A file produced by a job."""

    name: str = Field(description="Role of the file, e.g. 'srt', 'audio'.")
    path: str
    media_type: str | None = None
    size: int | None = None
    sha256: str | None = None

    @classmethod
    def from_path(cls, name: str, path: Path, *, checksum: bool = True) -> Artifact:
        path = Path(path).resolve()
        media_type, _ = mimetypes.guess_type(path.name)
        return cls(
            name=name,
            path=str(path),
            media_type=media_type,
            size=path.stat().st_size,
            sha256=sha256_file(path) if checksum else None,
        )


class HostInfo(BaseModel):
    hostname: str
    os: str
    os_release: str
    machine: str
    python: str

    @classmethod
    def current(cls) -> HostInfo:
        return cls(
            hostname=socket.gethostname(),
            os=platform.system(),
            os_release=platform.release(),
            machine=platform.machine(),
            python=platform.python_version(),
        )


class ErrorInfo(BaseModel):
    type: str
    message: str
    hint: str | None = None


class ToolOutput(BaseModel):
    """Base class for what a tool's logic function returns.

    ``text``, ``artifacts`` and ``device`` map onto the ``JobResult`` fields of
    the same name; every other field of a subclass ends up in ``JobResult.data``.
    """

    text: str | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    device: DeviceInfo | None = None

    def data(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"text", "artifacts", "device"})


class JobResult(BaseModel):
    """What ``--json`` prints on stdout: exactly one of these."""

    schema_version: int = SCHEMA_VERSION
    job_id: str
    tool: str
    tool_version: str
    ok: bool
    exit_code: int = ExitCode.OK
    error: ErrorInfo | None = None
    started_at: datetime
    finished_at: datetime
    duration_s: float
    host: HostInfo
    device: DeviceInfo | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    text: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def input_field(**kwargs: Any) -> Any:
    """``pydantic.Field`` for a params attribute that is a job *input* (file or URL).

    Such fields are emitted under ``JobSpec.inputs`` instead of ``params``.
    """
    extra = dict(kwargs.pop("json_schema_extra", None) or {})
    extra[INPUT_MARKER] = True
    return Field(json_schema_extra=extra, **kwargs)


def input_fields(model: type[BaseModel]) -> list[str]:
    names = []
    for name, info in model.model_fields.items():
        extra = info.json_schema_extra
        if isinstance(extra, dict) and extra.get(INPUT_MARKER):
            names.append(name)
    return names


def format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {err.get('msg')}")
    return "; ".join(parts)


def load_job(source: str | Path) -> tuple[JobSpec, Path]:
    """Read a ``JobSpec`` from a file or ``-`` (stdin).

    Returns the spec and the directory relative input paths are resolved
    against (the job file's directory, or the cwd for stdin).
    """
    if str(source) == "-":
        raw, base = sys.stdin.read(), Path.cwd()
        label = "<stdin>"
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise InvalidInputError(
                f"Job file not found: {path}",
                hint="Pass an existing JSON file to --job, or '-' to read it from stdin.",
            )
        raw, base, label = path.read_text(encoding="utf-8"), path.resolve().parent, str(path)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidInputError(f"Job {label} is not valid JSON: {exc}") from exc
    try:
        job = JobSpec.model_validate(data)
    except ValidationError as exc:
        raise InvalidInputError(
            f"Job {label} is not a valid JobSpec: {format_validation_error(exc)}",
            hint="See docs/ARCHITECTURE.md for the JobSpec format, or generate one "
            "with --print-job.",
        ) from exc
    if job.schema_version > SCHEMA_VERSION:
        raise InvalidInputError(
            f"Job schema_version {job.schema_version} is newer than supported ({SCHEMA_VERSION}).",
            hint="Upgrade the tool: uv tool upgrade --all",
        )
    return job, base


def job_from_params(tool: str, params: BaseModel, *, tool_version: str | None = None) -> JobSpec:
    """Build the ``JobSpec`` equivalent of an invocation (``--print-job``)."""
    data = params.model_dump(mode="json", exclude_none=True)
    inputs: dict[str, InputRef] = {}
    for name in input_fields(type(params)):
        value = data.pop(name, None)
        if value is None:
            continue
        is_url = isinstance(value, str) and "://" in value
        if is_url:
            inputs[name] = InputRef(url=value)
        else:
            path = Path(value).resolve()
            inputs[name] = InputRef(
                path=str(path),
                size=path.stat().st_size if path.is_file() else None,
            )
    return JobSpec(tool=tool, tool_version=tool_version, params=data, inputs=inputs)
