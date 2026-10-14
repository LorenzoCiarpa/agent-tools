# uvt-transcribe

Speech-to-text from the command line, for people and for agents:

| Command | What it does |
|---|---|
| `transcribe-file` | transcribes a local audio/video file (anything FFmpeg can read) |
| `yt-transcribe` | downloads only the audio of a video with yt-dlp and transcribes it; can use manual subtitles instead |
| `live-transcribe` | transcribes the microphone while you speak, segmenting on pauses |

## Install

```bash
uv tool install "git+https://github.com/LorenzoCiarpa/agent-tools#subdirectory=packages/transcribe"
```

| Extra | When | Command |
|---|---|---|
| `cuda` | NVIDIA GPU: bundles cuBLAS 12 (needed for int8 on GPU, or when no CUDA toolkit is installed) | `uv tool install "uvt-transcribe[cuda] @ git+https://github.com/LorenzoCiarpa/agent-tools#subdirectory=packages/transcribe"` |
| `voxtral` | local Voxtral via transformers + torch | `uv tool install "uvt-transcribe[cuda,voxtral] @ git+https://github.com/LorenzoCiarpa/agent-tools#subdirectory=packages/transcribe" --torch-backend auto` |

- **System dependencies.** Only `live-transcribe` on Linux needs one: PortAudio
  (`sudo apt install libportaudio2`, `sudo dnf install portaudio`, `sudo pacman -S portaudio`).
- **No ffmpeg.** Audio is decoded with PyAV.
- **Deno is bundled.** yt-dlp needs a JS runtime for YouTube; the `deno` wheel provides one. If your
  platform has no wheel, install Deno (`curl -fsSL https://deno.land/install.sh | sh`,
  `brew install deno`, `winget install DenoLand.Deno`) or Node ≥ 22.

## Examples

```bash
transcribe-file talk.mp3                       # prints the text, writes talk.txt
transcribe-file lecture.mp4 -l it -f srt,vtt -o subs/
transcribe-file memo.m4a --device cpu --model base --json

yt-transcribe "https://www.youtube.com/watch?v=jNQXAC9IVRw" -f txt,json
yt-transcribe URL --prefer-subs --sub-lang it  # manual subtitles if they exist, else transcribe
yt-transcribe URL --keep-audio -o ~/transcripts

live-transcribe --list-mics
live-transcribe -l it --final-pass -f txt,srt   # stop with Enter or Ctrl+C
live-transcribe --max-duration 600 --stop-file /tmp/stop-rec --json   # for agents
```

All commands share these options:

| Option | Meaning |
|---|---|
| `--backend/-b` | `faster-whisper` (default), `voxtral`, `api` |
| `--model/-m` | model name or path; faster-whisper default `large-v3-turbo` on GPU, `small` on CPU |
| `--device` | `auto` (default), `cuda`, `cpu`, `mps`. This is the compute device; `--mic` picks the microphone |
| `--compute-type` | `float16` on GPU, `int8` on CPU by default |
| `--language/-l` | ISO 639-1 code; default: auto-detect |
| `--format/-f` | `txt`, `srt`, `vtt`, `json`. Repeat or comma-separate them; `none` writes no files. The json file has segments with timestamps |
| `--output-dir/-o`, `--name` | where the files go and their base name |
| `--vad/--no-vad`, `--beam-size`, `--initial-prompt` | decoding settings |
| `--backend-option/-B KEY=VALUE` | backend-specific settings, e.g. `word_timestamps=true`, `batch_size=16` |
| `--json`, `--job FILE`, `--print-job`, `-v`, `-q` | automation (see below) |

## Backends

| Backend | Runs on | Notes |
|---|---|---|
| `faster-whisper` | CUDA (float16) or CPU (int8); macOS runs on the CPU via Accelerate | Default. No torch. If CUDA fails with `--device auto`, it falls back to the CPU with a warning. On GPUs newer than the CTranslate2 build (e.g. RTX 50xx), the first run compiles kernels (~10 s, cached afterwards). |
| `voxtral` | CUDA (bf16), Apple GPU, CPU | `mistralai/Voxtral-Mini-3B-2507` (~9.5 GB VRAM). Text only, so it is chunked on pauses into ≤30 s segments (`-B chunk_seconds=N`). RTX 50xx needs torch cu128+; `--torch-backend auto` picks it. |
| `api` | Mistral cloud | Needs `MISTRAL_API_KEY`. Model `voxtral-mini-latest`. The API cannot return timestamps when a language is given, so with `-l` you get one segment per chunk. Options: `diarize=true`, `context_bias=word1,word2`, `timestamps=true`. |

## For agents and scripts

- `--json` prints exactly one JSON object on stdout (a `JobResult`). Logs and progress go to stderr.
  ```json
  {"ok": true, "exit_code": 0, "tool": "transcribe-file", "tool_version": "0.1.0",
   "duration_s": 3.2, "host": {"hostname": "gpu-pc", "os": "Linux", ...},
   "device": {"kind": "cuda", "name": "NVIDIA GeForce RTX 5080", "compute_type": "float16"},
   "artifacts": [{"name": "srt", "path": "/abs/talk.srt", "size": 1234, "sha256": "..."}],
   "text": "full transcript...",
   "data": {"language": "en", "duration": 61.2, "backend": "faster-whisper", "model": "large-v3-turbo",
            "source": {"kind": "file", "path": "/abs/talk.mp3"}, "segments": [{"start": 0.0, "end": 4.2, "text": "..."}]}}
  ```
  On failure `ok` is false and `error` has `type`, `message` and an actionable `hint`.
- `--print-job` prints the equivalent `JobSpec` without running it.
  `--job job.json` (or `--job -` for stdin) runs one. Flags on the command line override the job.
- Exit codes: `0` ok, `1` error, `2` invalid input, `130` interrupted.
- `live-transcribe` stops on `--max-duration`, on `--stop-file PATH` appearing, or on SIGINT/SIGTERM.
  It always writes `<name>.wav` and appends to `<name>.live.txt` while recording.
- The typed Python API is `uvt_transcribe.api` (`transcribe_file`, `yt_transcribe`,
  `live_transcribe`, `list_microphones`). Each one is `params -> TranscriptionResult`.

## Configuration keys (`[transcribe]`)

`backend`, `model`, `device`, `compute_type`, `language`, `formats`, `output_dir`, `vad`, `beam_size`,
`initial_prompt`, plus backend settings: `model_gpu`, `model_cpu`, `compute_type_cuda`,
`compute_type_cpu`, `download_root` (faster-whisper), `voxtral_model`, `api_model`,
`api_endpoint`, `api_key_env`. Every key can be set with `UVT_TRANSCRIBE__<KEY>`.

## Troubleshooting

- **"Sign in to confirm you're not a bot"**: YouTube is blocking the IP. Retry later, change network,
  or add `--cookies-from-browser firefox`.
- **Other YouTube errors**: run `uv tool upgrade uvt-transcribe` to get a newer yt-dlp. If a fix is
  only in yt-dlp nightly, add `--with "yt-dlp[default]>=2026.10.10.dev0"` to `uv tool install`.
- **CUDA errors / CPU fallback**: CTranslate2 needs cuBLAS for CUDA 12. Install the `cuda` extra or
  a CUDA 12 toolkit. The system cuBLAS 12.0 shipped by Ubuntu works only for float16.
- **Windows + CUDA**: CTranslate2 can crash *at process exit* (after the result is printed);
  see [faster-whisper#71](https://github.com/SYSTRAN/faster-whisper/issues/71). Windows cannot
  deliver SIGTERM to a process, so agents should stop `live-transcribe` with `--stop-file`.
