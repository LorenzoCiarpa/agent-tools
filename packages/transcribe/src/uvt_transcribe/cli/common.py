"""Options and helpers shared by the transcription CLIs."""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from uvt_core.jobs import ToolOutput
from uvt_core.logging import get_logger
from uvt_core.runner import EXIT_CODES_HELP
from uvt_transcribe.models import Segment, TranscriptionResult

MODEL_PANEL = "Model"
OUTPUT_PANEL = "Output"

EPILOG = (
    f"{EXIT_CODES_HELP}\n\n"
    "Defaults can be set in the \\[transcribe] table of the uvt config file "
    "(see `uvt-info`) or with UVT_TRANSCRIBE__<KEY> env vars, e.g. "
    "UVT_TRANSCRIBE__MODEL_CPU=base."
)


class Device(StrEnum):
    auto = "auto"
    cuda = "cuda"
    cpu = "cpu"
    mps = "mps"


BackendOpt = Annotated[
    str | None,
    typer.Option(
        "--backend",
        "-b",
        help="Engine: faster-whisper (default), voxtral (local, needs the \\[voxtral] extra), "
        "api (Mistral cloud, needs MISTRAL_API_KEY).",
        rich_help_panel=MODEL_PANEL,
        show_default=False,
    ),
]
ModelOpt = Annotated[
    str | None,
    typer.Option(
        "--model",
        "-m",
        help="Model name or path. faster-whisper default: large-v3-turbo on GPU, small on CPU.",
        rich_help_panel=MODEL_PANEL,
        show_default=False,
    ),
]
DeviceOpt = Annotated[
    Device | None,
    typer.Option(
        "--device",
        help="Compute device: auto = CUDA if usable, else CPU (Apple GPU for voxtral).",
        rich_help_panel=MODEL_PANEL,
        show_default=False,
    ),
]
ComputeTypeOpt = Annotated[
    str | None,
    typer.Option(
        "--compute-type",
        help="Precision: float16 (GPU default), int8 (CPU default), int8_float16, float32…",
        rich_help_panel=MODEL_PANEL,
        show_default=False,
    ),
]
LanguageOpt = Annotated[
    str | None,
    typer.Option(
        "--language",
        "-l",
        help="Spoken language as ISO 639-1 code (it, en, fr…); default auto-detect.",
        rich_help_panel=MODEL_PANEL,
        show_default=False,
    ),
]
BeamSizeOpt = Annotated[
    int | None,
    typer.Option(
        "--beam-size",
        help="Beam size (default 5).",
        rich_help_panel=MODEL_PANEL,
        show_default=False,
    ),
]
VadOpt = Annotated[
    bool | None,
    typer.Option(
        "--vad/--no-vad",
        help="Skip silences with the Silero VAD (default on).",
        rich_help_panel=MODEL_PANEL,
        show_default=False,
    ),
]
PromptOpt = Annotated[
    str | None,
    typer.Option(
        "--initial-prompt",
        help="Text that primes the model: names, jargon, punctuation style.",
        rich_help_panel=MODEL_PANEL,
        show_default=False,
    ),
]
BackendOptionOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--backend-option",
        "-B",
        metavar="KEY=VALUE",
        help="Backend-specific setting, repeatable (e.g. word_timestamps=true, batch_size=16).",
        rich_help_panel=MODEL_PANEL,
        show_default=False,
    ),
]
FormatOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--format",
        "-f",
        help="Files to write: txt, srt, vtt, json (repeat or comma-separate; 'none' = no "
        "files). Default txt. The json file has segments with timestamps.",
        rich_help_panel=OUTPUT_PANEL,
        show_default=False,
    ),
]
OutputDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--output-dir",
        "-o",
        help="Where to write files (default: current directory).",
        rich_help_panel=OUTPUT_PANEL,
        show_default=False,
    ),
]
NameOpt = Annotated[
    str | None,
    typer.Option(
        "--name",
        help="Base name of the output files (default derived from the input).",
        rich_help_panel=OUTPUT_PANEL,
        show_default=False,
    ),
]


def parse_backend_options(values: list[str] | None) -> dict[str, Any] | None:
    if not values:
        return None
    out: dict[str, Any] = {}
    for item in values:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise typer.BadParameter(
                f"expected KEY=VALUE, got {item!r}", param_hint="--backend-option"
            )
        out[key.strip()] = value.strip()
    return out


def transcription_values(
    *,
    backend: str | None,
    model: str | None,
    device: Device | None,
    compute_type: str | None,
    language: str | None,
    beam_size: int | None,
    vad: bool | None,
    initial_prompt: str | None,
    backend_option: list[str] | None,
    formats: list[str] | None,
    output_dir: Path | None,
    name: str | None,
) -> dict[str, Any]:
    """CLI values -> params dict (``None`` = not given, so job/config apply)."""
    return {
        "backend": backend,
        "model": model,
        "device": device.value if device else None,
        "compute_type": compute_type,
        "language": language,
        "beam_size": beam_size,
        "vad": vad,
        "initial_prompt": initial_prompt,
        "backend_options": parse_backend_options(backend_option),
        "formats": formats,
        "output_dir": output_dir,
        "name": name,
    }


class SegmentPrinter:
    """Human mode: stream segment text to stdout as it is produced."""

    def __init__(self) -> None:
        self.printed = False

    def __call__(self, segment: Segment) -> None:
        if segment.text.strip():
            sys.stdout.write(segment.text.strip() + "\n")
            sys.stdout.flush()
            self.printed = True

    def finish(self, output: ToolOutput) -> None:
        """Print the text if nothing was streamed (e.g. subtitles were used)."""
        if not self.printed and isinstance(output, TranscriptionResult) and output.text:
            for seg in output.segments:
                self(seg)


def log_segment(segment: Segment) -> None:
    """JSON mode: show progress on stderr instead of stdout."""
    get_logger("transcribe").info("[%.1f s] %s", segment.start, segment.text)
