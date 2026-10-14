"""The thin layer between a Typer CLI and a tool's logic function.

A tool is a typed function ``func(params) -> ToolOutput`` plus a pydantic
params model. ``run()`` takes care of everything the conventions require, so
every command behaves the same way:

* ``--job FILE|-``  read params/inputs from a ``JobSpec`` (CLI flags override it)
* ``--print-job``   print the equivalent ``JobSpec`` and exit without running
* ``--json``        print exactly one ``JobResult`` on stdout (logs on stderr)
* exit codes 0 ok, 1 error, 2 invalid input, 130 interrupted
* SIGTERM is handled like Ctrl+C (clean shutdown, result still printed)

A future MCP server or remote worker calls the same ``func(params)`` directly,
or ``execute_job(tool, func, job)`` to get a ``JobResult``.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Generic, TypeVar

import typer
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ValidationError

from uvt_core.errors import ExitCode, InvalidInputError, ToolError
from uvt_core.jobs import (
    ErrorInfo,
    HostInfo,
    JobResult,
    JobSpec,
    ToolOutput,
    format_validation_error,
    input_fields,
    job_from_params,
    load_job,
    package_version,
    utcnow,
)
from uvt_core.logging import get_logger, setup_logging
from uvt_core.output import emit_json, stdout_to_stderr

P = TypeVar("P", bound=BaseModel)

EXIT_CODES_HELP = "[bold]Exit codes:[/bold] 0 ok · 1 error · 2 invalid input · 130 interrupted."
AUTOMATION_PANEL = "Automation (agents, scripts)"

JsonOption = Annotated[
    bool,
    typer.Option(
        "--json",
        help="Print only a JobResult JSON object on stdout (logs and progress go to stderr).",
        rich_help_panel=AUTOMATION_PANEL,
    ),
]
JobOption = Annotated[
    str | None,
    typer.Option(
        "--job",
        metavar="FILE",
        help="Read parameters from a JobSpec JSON file ('-' = stdin). "
        "Options given on the command line override the job's params.",
        rich_help_panel=AUTOMATION_PANEL,
    ),
]
PrintJobOption = Annotated[
    bool,
    typer.Option(
        "--print-job",
        help="Print the JobSpec equivalent of this command line and exit without running.",
        rich_help_panel=AUTOMATION_PANEL,
    ),
]
VerboseOption = Annotated[
    int,
    typer.Option(
        "--verbose",
        "-v",
        count=True,
        show_default=False,
        help="More logs on stderr (-v debug, -vv also third-party debug).",
        rich_help_panel=AUTOMATION_PANEL,
    ),
]
QuietOption = Annotated[
    bool,
    typer.Option(
        "--quiet",
        "-q",
        help="Only warnings and errors on stderr.",
        rich_help_panel=AUTOMATION_PANEL,
    ),
]


@dataclass(frozen=True)
class CommonOptions:
    json: bool = False
    job: str | None = None
    print_job: bool = False
    verbose: int = 0
    quiet: bool = False


@dataclass(frozen=True)
class Tool(Generic[P]):
    """Static description of a command: its name, distribution and params model."""

    name: str
    dist: str
    params_model: type[P]

    @property
    def version(self) -> str:
        return package_version(self.dist)


def check_tool_version(spec: str, installed: str) -> None:
    text = spec.strip()
    if text and text[0].isdigit():
        text = f"=={text}"
    try:
        ok = SpecifierSet(text).contains(Version(installed), prereleases=True)
    except (InvalidSpecifier, InvalidVersion) as exc:
        raise InvalidInputError(f"Invalid tool_version {spec!r}: {exc}") from exc
    if not ok:
        raise InvalidInputError(
            f"Job requires version {spec!r} but {installed} is installed.",
            hint="Upgrade with `uv tool upgrade --all`, or regenerate the job with --print-job.",
        )


def build_params(
    tool: Tool[P],
    cli_values: Mapping[str, Any],
    job: JobSpec | None = None,
    base_dir: Path | None = None,
) -> P:
    """Merge job params/inputs with explicit CLI values and validate them."""
    data: dict[str, Any] = {}
    if job is not None:
        if job.tool != tool.name:
            raise InvalidInputError(
                f"This job is for '{job.tool}', not '{tool.name}'.",
                hint=f"Run it with: {job.tool} --job <file>",
            )
        if job.tool_version:
            check_tool_version(job.tool_version, tool.version)
        data.update(job.params)
        allowed = set(input_fields(tool.params_model))
        for name, ref in job.inputs.items():
            if name not in allowed:
                raise InvalidInputError(
                    f"Unknown input '{name}' for {tool.name}.",
                    hint=f"Valid inputs: {', '.join(sorted(allowed)) or 'none'}.",
                )
            value = ref.value
            if ref.path is not None and base_dir is not None:
                path = Path(ref.path).expanduser()
                value = str(path if path.is_absolute() else base_dir / path)
            data[name] = value
    data.update({k: v for k, v in cli_values.items() if v is not None})
    try:
        return tool.params_model.model_validate(data)
    except ValidationError as exc:
        raise InvalidInputError(
            f"Invalid parameters: {format_validation_error(exc)}",
            hint=f"See `{tool.name} --help`.",
        ) from exc


@contextlib.contextmanager
def sigterm_as_interrupt(enabled: bool = True) -> Iterator[None]:
    """Turn SIGTERM into KeyboardInterrupt so cleanup and the final result still happen."""
    if not enabled or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    previous = signal.signal(signal.SIGTERM, _handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _error_info(exc: BaseException) -> tuple[ErrorInfo, int]:
    if isinstance(exc, ToolError):
        return ErrorInfo(type=exc.error_type, message=exc.message, hint=exc.hint), int(
            exc.exit_code
        )
    if isinstance(exc, KeyboardInterrupt):
        return ErrorInfo(type="interrupted", message="Interrupted."), int(ExitCode.INTERRUPTED)
    return (
        ErrorInfo(
            type="internal",
            message=f"{type(exc).__name__}: {exc}",
            hint="Re-run with -vv to see the traceback; this is probably a bug.",
        ),
        int(ExitCode.ERROR),
    )


def _execute(
    tool: Tool[P], func: Callable[[P], ToolOutput], params: P, job_id: str | None
) -> tuple[JobResult, ToolOutput | None]:
    log = get_logger(tool.name)
    started, t0 = utcnow(), time.perf_counter()
    output: ToolOutput | None = None
    error: ErrorInfo | None = None
    code = int(ExitCode.OK)
    try:
        output = func(params)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - reported in the result
        if not isinstance(exc, ToolError | KeyboardInterrupt):
            log.debug("Unhandled exception:\n%s", traceback.format_exc())
        error, code = _error_info(exc)
    result = JobResult(
        job_id=job_id or uuid.uuid4().hex,
        tool=tool.name,
        tool_version=tool.version,
        ok=error is None,
        exit_code=code,
        error=error,
        started_at=started,
        finished_at=utcnow(),
        duration_s=round(time.perf_counter() - t0, 3),
        host=HostInfo.current(),
        device=output.device if output else None,
        artifacts=output.artifacts if output else [],
        text=output.text if output else None,
        data=output.data() if output else {},
    )
    return result, output


def execute(
    tool: Tool[P],
    func: Callable[[P], ToolOutput],
    params: P,
    *,
    job_id: str | None = None,
) -> JobResult:
    """Run ``func(params)`` and wrap the outcome in a ``JobResult`` (never raises
    for tool errors). This is the entry point for remote workers / MCP."""
    return _execute(tool, func, params, job_id)[0]


def execute_job(tool: Tool[P], func: Callable[[P], ToolOutput], job: JobSpec) -> JobResult:
    """Validate a ``JobSpec`` and execute it (what a remote worker will call)."""
    try:
        params = build_params(tool, {}, job)
    except ToolError as exc:
        return _failed_result(tool, exc, job_id=job.id)
    return execute(tool, func, params, job_id=job.id)


def _failed_result(tool: Tool[Any], exc: BaseException, *, job_id: str | None = None) -> JobResult:
    error, code = _error_info(exc)
    now = utcnow()
    return JobResult(
        job_id=job_id or uuid.uuid4().hex,
        tool=tool.name,
        tool_version=tool.version,
        ok=False,
        exit_code=code,
        error=error,
        started_at=now,
        finished_at=now,
        duration_s=0.0,
        host=HostInfo.current(),
    )


def _log_error(tool: Tool[Any], error: ErrorInfo) -> None:
    log = get_logger(tool.name)
    log.error("%s", error.message)
    if error.hint:
        for line in error.hint.splitlines():
            log.error("  hint: %s", line)


def run(
    tool: Tool[P],
    func: Callable[[P], ToolOutput],
    cli_values: Mapping[str, Any],
    opts: CommonOptions,
    *,
    on_success: Callable[[ToolOutput], None] | None = None,
    handle_sigterm: bool = True,
) -> int:
    """CLI entry: parse job/flags, run, report. Returns the exit code."""
    setup_logging(opts.verbose, opts.quiet)
    log = get_logger(tool.name)
    with stdout_to_stderr(opts.json) as out, sigterm_as_interrupt(handle_sigterm):
        try:
            job, base = load_job(opts.job) if opts.job else (None, None)
            params = build_params(tool, cli_values, job, base)
        except ToolError as exc:
            result = _failed_result(tool, exc)
            _log_error(tool, result.error)  # type: ignore[arg-type]
            if opts.json:
                emit_json(result, out)
            return result.exit_code

        if opts.print_job:
            major_minor = ".".join(tool.version.split(".")[:2])
            spec = job_from_params(tool.name, params, tool_version=f"=={major_minor}.*")
            if job is not None:
                spec.id, spec.meta = job.id, job.meta
            emit_json(spec, out)
            return int(ExitCode.OK)

        result, output = _execute(tool, func, params, job.id if job else None)
        if result.error:
            _log_error(tool, result.error)
        else:
            for artifact in result.artifacts:
                log.info("Wrote %s", artifact.path)
            level = logging.INFO if result.duration_s >= 1 else logging.DEBUG
            log.log(level, "Done in %.1f s", result.duration_s)
        if opts.json:
            emit_json(result, out)
        elif output is not None and on_success is not None:
            on_success(output)
        return result.exit_code


def new_app(help_text: str) -> typer.Typer:
    """A single-command Typer app with the shared settings."""
    return typer.Typer(
        help=help_text,
        add_completion=False,
        no_args_is_help=False,
        rich_markup_mode="rich",
        pretty_exceptions_enable=False,
        context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
    )


def configure_streams() -> None:
    """UTF-8 on pipes/files (agents read UTF-8; Windows defaults to the ANSI code
    page) and never crash on characters a console cannot show."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty():
                stream.reconfigure(errors="replace")  # type: ignore[union-attr]
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


def _is_usage_error(exc: BaseException) -> bool:
    # Click's ClickException (vendored inside recent Typer versions, so matched by shape).
    return all(hasattr(exc, attr) for attr in ("exit_code", "format_message", "show"))


def main(app: typer.Typer, tool: Tool[Any], argv: Sequence[str] | None = None) -> None:
    """Console-script entry point: run the Typer app with our exit-code policy.

    Usage errors (unknown option, bad value) exit with 2 and, if ``--json`` was
    requested, still print a ``JobResult`` so callers can always parse stdout.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    configure_streams()
    try:
        rv = app(args=args, prog_name=tool.name, standalone_mode=False)
    except typer.Abort:
        sys.exit(int(ExitCode.INTERRUPTED))
    except Exception as exc:
        if not _is_usage_error(exc):
            raise
        exc.show()  # type: ignore[attr-defined]
        code = int(getattr(exc, "exit_code", 0) or ExitCode.INVALID_INPUT)
        if "--json" in args:
            message = exc.format_message()  # type: ignore[attr-defined]
            err = InvalidInputError(message, hint=f"See `{tool.name} --help`.")
            emit_json(_failed_result(tool, err))
        sys.exit(code)
    sys.exit(rv if isinstance(rv, int) else 0)
