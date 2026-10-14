# Architecture

## Goals and constraints

- **Many small CLI tools.** Each one is installed separately with `uv tool install` on
  different machines: a desktop with an RTX 5080 (Blackwell, 16 GB), a small always-on server
  that coordinates agents, and laptops without a GPU.
- **Linux, macOS and Windows.** A GPU is an optional accelerator, never a requirement.
- **Usable by people and by agents.** Machine-readable output, a declarative job format,
  no prompts, stable exit codes.
- **Remote execution later.** It must be a thin layer on top of what exists. Example: the
  server delegates a transcription to the GPU desktop, which is not always on.

## Layout

```
packages/core        uvt-core: config, logging, device, output, jobs (JobSpec/JobResult), runner, uvt-info
packages/transcribe  uvt-transcribe: api, options, pipeline, backends/, youtube, live, formats, audio, cli/
```

Every tool package depends only on `uvt-core`. Tool venvs are isolated (`uv tool`), so
tools never import each other. A future dispatcher calls them as executables with
`--job`/`--json`.

### Installing from a git subdirectory

`packages/transcribe/pyproject.toml` declares `uvt-core` with
`[tool.uv.sources] uvt-core = { workspace = true }`. When uv installs
`git+https://…/agent-tools#subdirectory=packages/transcribe`, it checks out the repo, finds the
workspace root above the subdirectory, and installs `uvt-core` from the same commit as a git
dependency (`…#subdirectory=packages/core`).

- Introduced in uv 0.2.35 (#5944) and 0.4.29 (#8665).
- Verified locally with `git+file://`, and from GitHub after the first push.
- We therefore keep the workspace source. No duplicated direct-URL dependency is needed.

Caveats:

- `pip` ignores `tool.uv.sources`. These packages are meant to be installed with uv.
- `uv tool install` resolves from scratch and does not use `uv.lock`, so every dependency
  has an explicit range.
- With `uv tool install --editable ./packages/transcribe`, the workspace dependency `uvt-core`
  is editable too (uv ≥ 0.11.7).

## The contract: JobSpec → JobResult

```jsonc
// JobSpec  (--job job.json, --print-job emits one)
{
  "schema_version": 1,
  "id": "6f1c…",                       // idempotency key; echoed as JobResult.job_id
  "tool": "transcribe-file",           // executable name
  "tool_version": "==0.1.*",           // PEP 440 specifier the worker must satisfy
  "params": {"language": "it", "formats": ["srt"]},
  "inputs": {"media": {"path": "talk.mp3", "sha256": "…", "size": 123456}},
  "created_at": "2026-10-14T15:00:00Z",
  "meta": {}                           // free-form caller data (requester, priority…)
}
```

```jsonc
// JobResult  (--json prints exactly one, on success and on failure)
{
  "schema_version": 1, "job_id": "6f1c…", "tool": "transcribe-file", "tool_version": "0.1.0",
  "ok": true, "exit_code": 0, "error": null,       // error: {type, message, hint}
  "started_at": "…", "finished_at": "…", "duration_s": 12.3,
  "host": {"hostname": "gpu-pc", "os": "Linux", "machine": "x86_64", "python": "3.12.3", …},
  "device": {"kind": "cuda", "name": "NVIDIA GeForce RTX 5080", "compute_type": "float16"},
  "artifacts": [{"name": "srt", "path": "/abs/talk.srt", "size": 812, "sha256": "…"}],
  "text": "…",
  "data": {"segments": […], "language": "it", "source": {…}, …}
}
```

Design choices:

- **`inputs` are separate from `params`.** Inputs are exactly what a dispatcher must move
  between machines: files with `path`, or `url`. A params model marks such fields with
  `uvt_core.jobs.input_field`. A relative input path is resolved against the job file's directory.
- **`--print-job` emits only the explicitly given params.** Machine-specific defaults
  (GPU vs CPU model, output dir, language) come from the executing machine's config.
  The same job therefore runs well on a GPU box and on a laptop.
- **Errors are results.** With `--json`, invalid input (exit 2), failures (1) and interruptions
  (130) still print a `JobResult` with `ok: false`. Usage errors (unknown flag) do too.
- **The device and the host are always reported.** A dispatcher can log where a job actually ran.
  `device.kind` is `cuda`, `mps`, `cpu` or `remote` (a cloud API did the work).

### Layers

```
typed logic      uvt_transcribe.api.transcribe_file(params, on_segment=…, backend=…) -> TranscriptionResult
job execution    uvt_core.runner.execute(tool, func, params) / execute_job(tool, func, job) -> JobResult
CLI              uvt_core.runner.run(...) inside a Typer command: flags + --job merge, --json, exit codes
```

A remote worker or MCP server plugs in at the first two layers. The `backend=` argument
lets a long-lived process keep a model loaded across jobs.

## Transcription decisions

| Topic | Decision | Why |
|---|---|---|
| Default engine | faster-whisper (CTranslate2) | No torch; CUDA or CPU; good accuracy/speed. |
| Device policy | `auto`: CUDA float16 if usable, else CPU int8. A 1 s warm-up at load time. | The warm-up exposes broken CUDA setups early, so `auto` can fall back to CPU. It also absorbs the PTX JIT on Blackwell: the CTranslate2 4.8 wheels ship SASS up to sm_86 plus PTX, so the first run on an RTX 5080 takes ~12 s (then cached in `~/.nv/ComputeCache`). |
| CUDA libraries | CTranslate2 ≥ 4.6.3 needs **cuBLAS 12 only** (no cuDNN). Optional `cuda` extra = `nvidia-cublas-cu12`, preloaded with ctypes before use. | Ubuntu's system cuBLAS 12.0 only supports float16 on sm_120 (int8 fails with `CUBLAS_STATUS_NOT_SUPPORTED`); cuBLAS 12.9 from the wheel supports everything. |
| Default models | `large-v3-turbo` on GPU, `small` on CPU (`model_gpu` / `model_cpu` config keys) | Turbo is fast on GPU; small is real-time-ish on laptops. |
| Audio decoding | Our own PyAV decoder (`audio.load_audio`), numpy arrays passed to all backends | No ffmpeg binary. faster-whisper 1.2.1's `decode_audio` breaks with PyAV ≥ 19 (`metadata_errors` kwarg). |
| YouTube | yt-dlp Python API, `format=bestaudio/best`, no postprocessors, `fixup=never` | No ffmpeg. Opus/webm and DASH m4a decode fine with PyAV. |
| JS runtime | `deno` wheel as a dependency (marker-limited to platforms with wheels); lookup: venv scripts dir, PATH, `~/.deno/bin`, then node | Current yt-dlp needs Deno (default) or Node ≥ 22 plus `yt-dlp-ejs` for YouTube challenges. Without one, extraction degrades and is deprecated. For YouTube URLs we fail fast with per-OS install instructions. |
| `--prefer-subs` | Only `info["subtitles"]` (manual), never `automatic_captions`; vtt > srt; fetched with `ydl.urlopen` | Auto-captions are worse than Whisper. No second extraction is needed. |
| Live segmentation | faster-whisper's bundled Silero VAD (ONNX) on a rolling buffer every 0.3 s; a segment closes after `min_silence` (0.6 s) or is forced at `max_segment` (20 s) | No torch, no webrtcvad. ~1 ms per call. |
| Live robustness | Capture callback → queue → **main thread writes the WAV immediately** (header rewritten every second) → worker thread transcribes. The `.live.txt` log is flushed and fsynced per segment. | A slow model never delays or loses audio. A crash leaves a playable WAV and the text so far. |
| Live context | `initial_prompt` = user prompt + last 200 chars of text. The language is locked after the first detection. | Continuity across segments; short segments mis-detect languages. |
| Voxtral | `Voxtral-Mini-3B-2507` in bf16 via transformers ≥ 5.2 + mistral-common ≥ 1.11.5; chunked on VAD pauses (≤ 30 s) | Fits in 16 GB. The local API returns text only, so chunking provides timestamps. |
| Mistral API | Raw httpx multipart to `/v1/audio/transcriptions`, `voxtral-mini-latest`. Always 16 kHz mono WAV, ≤ 30 min per request. | No SDK dependency, so any input format works. Note: `timestamp_granularities` cannot be combined with `language`. |

## Roadmap: remote execution

The pieces above are the contract; remote execution adds a transport and a scheduler.

### 1. Worker (on the GPU machine)

A new package, `uvt-worker`, with a small HTTP service (`uvt-worker serve`). It listens on the
LAN or a Tailscale/WireGuard address and requires a bearer token.

| Endpoint | Purpose |
|---|---|
| `GET /health` | `uvt-info --json` output, plus installed tools and versions (`uv tool list`), queue length |
| `POST /jobs` | Multipart: the `JobSpec` plus one part per file input (or `sha256` references to blobs already uploaded) |
| `GET /jobs/{id}` | Status: queued / running / done / failed |
| `GET /jobs/{id}/result` | The `JobResult` |
| `GET /jobs/{id}/artifacts/{name}` | Download an artifact |
| `DELETE /jobs/{id}` | Cancel or clean up |

Execution, v1: write the inputs into a job directory, rewrite `inputs.*.path` to local paths,
set `params.output_dir` to the job directory, then run the **installed executable**
(`transcribe-file --job job.json --json`) and parse stdout. This keeps the tool venvs isolated,
exactly as on the command line, and needs no imports across venvs.

v2: resident per-tool workers that call `execute_job()` in-process and reuse a loaded backend
(`backend=`), so they don't pay model load and PTX JIT costs per job.

### 2. Transfer

- **Inputs.** `path` inputs are uploaded. They are content-addressed by `sha256` (`InputRef.sha256`),
  so re-sending the same file is a no-op. `url` inputs (e.g. YouTube) are passed through and the
  worker downloads them itself.
- **Results.** The dispatcher fetches the `JobResult` and then the artifacts it needs. It verifies
  `size`/`sha256` and rewrites `artifacts[].path` to local paths. `host`/`device` keep showing
  where the work happened.
- **Large files.** Use chunked uploads with resume, and optionally compress inputs to Opus first.

### 3. Dispatcher (on the always-on server)

`uvt-dispatch run [--target gpu-pc] -- transcribe-file talk.mp3 -l it --json` will work like this:

1. Build the JobSpec locally with `transcribe-file … --print-job`. No logic is duplicated.
2. Choose a target from a small inventory (`[dispatch.targets.gpu-pc] url = …, mac = …`).
   Probe its health with a short timeout.
3. **If the GPU machine is off**, there are two options:
   - Send a Wake-on-LAN magic packet (UDP broadcast to port 9 with the configured MAC), then
     poll `/health` for up to `wake_timeout` (e.g. 90 s).
   - If it does not come up, apply the job's fallback policy: run locally on the CPU (same
     executable; the server's config picks `model_cpu`), or rerun with `--backend api` when
     `MISTRAL_API_KEY` is available. The policy is chosen per job with `meta.fallback` or a
     dispatcher flag: `["local", "api"]` by default, `[]` to fail instead.
4. Submit, wait (or return the job id when `--async`), and download artifacts.
5. Print the worker's `JobResult`, annotated with dispatch info (target, wake attempts, fallback
   used), so agents see one consistent result whatever happened.

Persistence: a SQLite queue on the server. The job `id` makes retries idempotent; there are
timeouts per tool and cancellation via `DELETE`.

### 4. Agents and MCP

Each tool's params model gives a JSON Schema (`Model.model_json_schema()`) and its logic
function gives the behaviour. An MCP server can expose every tool, local or dispatched,
with the result as a `JobResult`. Agents already get the same contract through `--json`/`--job`.
