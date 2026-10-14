"""``transcribe-file``: transcribe a local audio or video file."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from uvt_core.runner import (
    CommonOptions,
    JobOption,
    JsonOption,
    PrintJobOption,
    QuietOption,
    Tool,
    VerboseOption,
    main as run_main,
    new_app,
    run,
)
from uvt_transcribe.cli.common import (
    EPILOG,
    BackendOpt,
    BackendOptionOpt,
    BeamSizeOpt,
    ComputeTypeOpt,
    DeviceOpt,
    FormatOpt,
    LanguageOpt,
    ModelOpt,
    NameOpt,
    OutputDirOpt,
    PromptOpt,
    SegmentPrinter,
    VadOpt,
    transcription_values,
)
from uvt_transcribe.options import TranscribeFileParams

TOOL = Tool(name="transcribe-file", dist="uvt-transcribe", params_model=TranscribeFileParams)

app = new_app("Transcribe a local audio or video file.")


@app.command(epilog=EPILOG)
def transcribe_file(
    media: Annotated[
        Path | None,
        typer.Argument(help="Audio or video file (anything FFmpeg can read).", show_default=False),
    ] = None,
    backend: BackendOpt = None,
    model: ModelOpt = None,
    device: DeviceOpt = None,
    compute_type: ComputeTypeOpt = None,
    language: LanguageOpt = None,
    beam_size: BeamSizeOpt = None,
    vad: VadOpt = None,
    initial_prompt: PromptOpt = None,
    backend_option: BackendOptionOpt = None,
    formats: FormatOpt = None,
    output_dir: OutputDirOpt = None,
    name: NameOpt = None,
    json_: JsonOption = False,
    job: JobOption = None,
    print_job: PrintJobOption = False,
    verbose: VerboseOption = 0,
    quiet: QuietOption = False,
) -> None:
    """Transcribe a local audio or video file and write txt/srt/vtt/json.

    The transcript is printed on stdout as it is produced; with --json stdout
    contains only the final JobResult (text, segments, files, device, timing).

    [bold]Examples[/bold]

      transcribe-file talk.mp3

      transcribe-file lecture.mp4 -l it -f srt,vtt -o subs/

      transcribe-file memo.m4a --device cpu --model base --json

      transcribe-file interview.wav --backend api --json

      transcribe-file --job job.json --json
    """
    values = transcription_values(
        backend=backend,
        model=model,
        device=device,
        compute_type=compute_type,
        language=language,
        beam_size=beam_size,
        vad=vad,
        initial_prompt=initial_prompt,
        backend_option=backend_option,
        formats=formats,
        output_dir=output_dir,
        name=name,
    )
    values["media"] = media
    opts = CommonOptions(json=json_, job=job, print_job=print_job, verbose=verbose, quiet=quiet)
    printer = SegmentPrinter()

    def func(params: TranscribeFileParams):
        from uvt_transcribe.pipeline import transcribe_file as do_transcribe

        return do_transcribe(params, on_segment=None if json_ else printer)

    raise typer.Exit(run(TOOL, func, values, opts, on_success=printer.finish))


def main() -> None:
    run_main(app, TOOL)


if __name__ == "__main__":
    main()
