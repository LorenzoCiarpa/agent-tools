# AGENTS.md

Read this before changing the repo. It applies to humans and AI agents alike.

## What this repo is

A uv workspace of independently installable CLI tool packages (`packages/*`). Each one is
installed with `uv tool install "git+https://github.com/LorenzoCiarpa/agent-tools#subdirectory=packages/<dir>"`
into its **own venv**. So:

- tools never import each other at runtime; they cooperate through executables on `PATH` and JSON;
- `uvt-core` is the only shared library. Every tool depends on it, and it must stay light
  (no ML frameworks, nothing compiled and heavy);
- `uv tool install` ignores `uv.lock`, so every dependency needs a sensible version range in `pyproject.toml`.

```
pyproject.toml          workspace root (virtual), dev group: ruff + pytest, ruff/pytest config
packages/core           uvt-core      -> uvt_core       config, logging, device, output, jobs, runner
packages/transcribe     uvt-transcribe -> uvt_transcribe transcribe-file, yt-transcribe, live-transcribe
scripts/new_package.py  scaffolds a new package
docs/ARCHITECTURE.md    design decisions + remote execution roadmap
```

## Conventions (required for every tool)

1. **Logic first, CLI second.** Each feature is a typed function in `<pkg>/api.py` (or a
   module re-exported from it): `func(params: PydanticModel) -> ToolOutput subclass`.
   The Typer command only collects flags and calls `uvt_core.runner.run(TOOL, func, values, opts)`.
   No business logic in `cli/`. The same function is what an MCP server or a remote worker will call.
2. **Params models.** pydantic, `extra="forbid"`. CLI options default to `None` ("not
   given"), so job params and config apply. File/URL inputs that a remote dispatcher must
   transfer are declared with `uvt_core.jobs.input_field(...)`.
3. **Output.** Return a `ToolOutput` subclass. `text`, `artifacts` (`Artifact.from_path`) and
   `device` map onto `JobResult`; every other field goes to `JobResult.data`.
4. **stdout is sacred.** Logic never prints. Use `uvt_core.logging.get_logger(...)`, which
   writes to stderr. With `--json` the runner redirects fd 1 to stderr while the job runs, but
   don't rely on that.
5. **Errors.** Raise `InvalidInputError` (exit 2) for bad params/files, `ToolError` (exit 1) for
   failures, `MissingDependencyError` with `os_hints={"Linux": ..., "Darwin": ..., "Windows": ...}`
   for missing extras or system libraries. Messages must say what to do next (`hint`).
6. **Non-interactive.** No `input()`, no prompts, no pagers. Interactive niceties (e.g. "press
   Enter to stop") only when `sys.stdin.isatty()`.
7. **Portable.** `pathlib` everywhere, no shell (`shell=True`, `os.system`), no hard-coded
   separators or `/tmp`. Use `tempfile` and `platformdirs`. A GPU is optional: always have a CPU path.
8. **Lazy heavy imports.** Import torch / faster_whisper / yt_dlp / sounddevice inside the
   functions that need them. `--help` must stay fast and must work without optional
   extras or system libraries; `tests/test_cli_help.py` checks this.
9. **Help.** Every command has a docstring with an `[bold]Examples[/bold]` section and the
   exit-code epilog. Escape literal brackets in help text as `\\[like this]` (rich markup).
10. **Config.** Read tool defaults with `uvt_core.config.tool_config("<section>")`. Precedence is
    params > `UVT_<SECTION>__<KEY>` env > config file > defaults.

## Adding a package

```bash
uv run scripts/new_package.py <name> --command <cli-name> --description "One line"
uv sync --all-packages
uv run <cli-name> --help && uv run pytest packages/<name>
```

Then:
- add real dependencies with version ranges to `packages/<name>/pyproject.toml`; optional
  heavy stacks go in an extra;
- add the package to the table in `README.md`, and its commands to the `--help` step in
  `.github/workflows/ci.yml`;
- keep `[tool.uv.sources] uvt-core = { workspace = true }`. uv resolves it when installing from
  a git subdirectory (verified, see docs/ARCHITECTURE.md).

## Adding a transcription backend

Subclass `uvt_transcribe.backends.base.TranscriptionBackend` (`load`, `model_name`,
`device_info`, `transcribe`, `close`). Register it in `uvt_transcribe/backends/__init__.py`
as a `"module:Class"` string so it is imported lazily. Raise `MissingDependencyError` with
`install_hint("<extra>")` when its extra is missing. Add unit tests with mocked I/O and a `slow`
test with the real thing.

## Tests

- `uv run pytest`: unit tests only (default `-m 'not slow and not gpu and not network and not audio_device'`).
  These use a fake backend, a fake microphone and fake yt-dlp. They must pass on Linux, macOS and Windows.
- `uv run pytest -m slow` (optionally `-m "slow and gpu"` / `"slow and network"`) runs the real tiny
  model, the real VAD + live pipeline and a real YouTube video.
- Tests must not read the developer's config. The transcribe `conftest.py` points
  `UVT_CONFIG_FILE` at a temp path.

## Checklist before committing

- [ ] `uv sync --all-packages` works and `uv lock --check` passes (commit `uv.lock` when deps change)
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] `uv run pytest` passes; run `uv run pytest -m slow` when touching models, audio or yt-dlp code
- [ ] every command's `--help` still works, and `--json` still prints one JSON object
- [ ] new options are documented in `--help` and in the package README
- [ ] dependency ranges are explicit; no new mandatory system dependency, or it is documented with per-OS instructions
- [ ] behaviour changes bump the package `version`; breaking JobSpec/JobResult changes bump `SCHEMA_VERSION`
- [ ] small, descriptive commits. Push to `main` only with green tests.
