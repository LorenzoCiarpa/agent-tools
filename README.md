# agent-tools

A collection of Python CLI tools installed with [`uv tool`](https://docs.astral.sh/uv/concepts/tools/).
They are meant both for humans at a terminal and for AI agents. Every tool:

- works on Linux, macOS and Windows; a GPU speeds things up but is never required;
- prints exactly one JSON `JobResult` on stdout with `--json`;
- accepts its parameters as a JSON `JobSpec` with `--job job.json`;
- never prompts when stdin is not a TTY;
- uses documented exit codes: `0` ok, `1` error, `2` invalid input, `130` interrupted.

Each package is installed on its own and lives in its own venv. Tools talk to
each other only through executables on `PATH` and JSON, never through imports.

| Package | Directory | Commands |
|---|---|---|
| `uvt-core` | [`packages/core`](packages/core) | `uvt-info` (device + config of this machine); shared library for the other tools |
| `uvt-transcribe` | [`packages/transcribe`](packages/transcribe) | `transcribe-file`, `yt-transcribe`, `live-transcribe` |

Design notes and the remote-execution roadmap are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Conventions for contributors and agents are in [AGENTS.md](AGENTS.md).

## Install

You only need [uv](https://docs.astral.sh/uv/getting-started/installation/), which manages Python too:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install the tools straight from git:

```bash
# Transcription tools (CPU everywhere, NVIDIA GPU when available)
uv tool install "git+https://github.com/LorenzoCiarpa/agent-tools#subdirectory=packages/transcribe"

# On an NVIDIA machine, also bundle cuBLAS (needed for int8 on GPU or without a CUDA toolkit)
uv tool install "uvt-transcribe[cuda] @ git+https://github.com/LorenzoCiarpa/agent-tools#subdirectory=packages/transcribe"

# Local Voxtral as well (pulls torch; uv picks the right CUDA build for the driver)
uv tool install "uvt-transcribe[cuda,voxtral] @ git+https://github.com/LorenzoCiarpa/agent-tools#subdirectory=packages/transcribe" --torch-backend auto

# Just the machine-info command
uv tool install "git+https://github.com/LorenzoCiarpa/agent-tools#subdirectory=packages/core"
```

If the commands are not found afterwards, run `uv tool update-shell` and open a new shell.
To try a tool without installing it:

```bash
uvx --from "git+https://github.com/LorenzoCiarpa/agent-tools#subdirectory=packages/transcribe" yt-transcribe --help
```

System requirements:

- **Linux + `live-transcribe`**: the PortAudio library (`sudo apt install libportaudio2`).
  macOS and Windows get it bundled with the `sounddevice` wheel.
- Nothing else is required. FFmpeg is not needed: audio is decoded with PyAV's bundled
  libraries. The JavaScript runtime that yt-dlp needs for YouTube (Deno) is installed as a
  wheel on Linux x86-64/arm64, macOS and Windows x64.
- If the newest Python has no wheels yet for a dependency, pin one: `uv tool install --python 3.12 ...`.

## Update and uninstall

```bash
uv tool upgrade --all              # pulls the latest commit of every git-installed tool
uv tool list                       # what is installed, with its commands
uv tool uninstall uvt-transcribe   # or: uv tool uninstall --all
```

`uv tool upgrade` does not remember `--torch-backend`. For a Voxtral install, upgrade with
`uv tool install --reinstall "uvt-transcribe[cuda,voxtral] @ git+..." --torch-backend auto`.

## Configuration

Optional TOML file. `uvt-info` prints its path, for example `~/.config/uvt/config.toml` on
Linux, `~/Library/Application Support/uvt/config.toml` on macOS and
`%LOCALAPPDATA%\uvt\config.toml` on Windows.

```toml
log_level = "INFO"

[transcribe]
backend = "faster-whisper"       # or voxtral / api
model_gpu = "large-v3-turbo"     # faster-whisper on CUDA
model_cpu = "small"              # faster-whisper on CPU
formats = ["txt", "srt"]
output_dir = "~/transcripts"
language = "it"                  # default: auto-detect
```

Any key can be overridden with an environment variable: `UVT_<KEY>` for top-level keys
and `UVT_<SECTION>__<KEY>` for keys in a table, e.g. `UVT_TRANSCRIBE__MODEL_CPU=base`.
`UVT_CONFIG_FILE` points to another file. Precedence is CLI flags / job params, then env,
then the file, then built-in defaults.

## Development

```bash
git clone https://github.com/LorenzoCiarpa/agent-tools && cd agent-tools
uv sync --all-packages                    # one venv with every package (editable) + dev tools
uv run pytest                             # unit tests (fast, no models, no hardware)
uv run pytest -m slow                     # + real tiny model, real microphone pipeline (add "gpu", "network")
uv run ruff check . && uv run ruff format --check .

# Install your working copy as real tools (edits apply immediately, uvt-core included)
uv tool install --editable ./packages/transcribe
uv tool install --editable ./packages/core

# New tool package with the conventions already wired in
uv run scripts/new_package.py pdf --command pdf-extract --description "Extract text from PDFs"
```

Licensed under the [MIT license](LICENSE).
