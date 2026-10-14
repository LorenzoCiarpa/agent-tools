"""``live-transcribe``: transcribe the microphone while you speak."""

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
from uvt_transcribe.options import LiveTranscribeParams

TOOL = Tool(name="live-transcribe", dist="uvt-transcribe", params_model=LiveTranscribeParams)

app = new_app("Transcribe the microphone live, segmenting on pauses.")
MIC_PANEL = "Microphone and stopping"


def _render_mics(output) -> None:
    for mic in output.mics:
        star = "*" if mic.is_default else " "
        typer.echo(
            f"{star} {mic.index:>3}  {mic.name}  [{mic.hostapi}, {mic.channels} ch, "
            f"{mic.default_samplerate:.0f} Hz]"
        )


@app.command(epilog=EPILOG)
def live_transcribe(
    list_mics: Annotated[
        bool,
        typer.Option(
            "--list-mics",
            help="List input devices (* = default) and exit.",
            rich_help_panel=MIC_PANEL,
        ),
    ] = False,
    mic: Annotated[
        str | None,
        typer.Option(
            "--mic",
            help="Microphone index or part of its name (see --list-mics). Default: system default.",
            rich_help_panel=MIC_PANEL,
            show_default=False,
        ),
    ] = None,
    max_duration: Annotated[
        float | None,
        typer.Option(
            "--max-duration",
            metavar="SECONDS",
            help="Stop after this many seconds.",
            rich_help_panel=MIC_PANEL,
            show_default=False,
        ),
    ] = None,
    stop_file: Annotated[
        Path | None,
        typer.Option(
            "--stop-file",
            metavar="PATH",
            help="Stop as soon as this file appears (for agents; works on Windows too).",
            rich_help_panel=MIC_PANEL,
            show_default=False,
        ),
    ] = None,
    final_pass: Annotated[
        bool | None,
        typer.Option(
            "--final-pass/--no-final-pass",
            help="At the end, re-transcribe the whole recording for a cleaner result.",
            show_default=False,
        ),
    ] = None,
    min_silence: Annotated[
        float | None,
        typer.Option(
            "--min-silence",
            metavar="SECONDS",
            help="Pause that closes a segment (default 0.6).",
            show_default=False,
        ),
    ] = None,
    max_segment: Annotated[
        float | None,
        typer.Option(
            "--max-segment",
            metavar="SECONDS",
            help="Force a cut after this long without pauses (default 20).",
            show_default=False,
        ),
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
    """Record the microphone and print the transcript while you speak.

    Audio is segmented on pauses (Silero VAD) and each segment is transcribed
    with the previous text as context. The full recording is always saved as
    WAV and the transcript is appended to <name>.live.txt as it is produced, so
    nothing is lost on a crash. Stop with Enter or Ctrl+C; agents use
    --max-duration, --stop-file or SIGTERM. --device is the compute device,
    --mic the microphone.

    Linux needs the PortAudio library: sudo apt install libportaudio2

    [bold]Examples[/bold]

      live-transcribe --list-mics

      live-transcribe -l it --final-pass -f txt,srt

      live-transcribe --max-duration 60 --json

      live-transcribe --stop-file /tmp/stop-recording --mic USB --json
    """
    opts = CommonOptions(json=json_, job=job, print_job=print_job, verbose=verbose, quiet=quiet)
    if list_mics:
        from uvt_core.runner import Tool as _Tool
        from uvt_transcribe.live import MicList, list_microphones
        from uvt_transcribe.options import TranscribeOptions

        mic_tool = _Tool(
            name="live-transcribe", dist="uvt-transcribe", params_model=TranscribeOptions
        )
        code = run(
            mic_tool, lambda _p: MicList(mics=list_microphones()), {}, opts, on_success=_render_mics
        )
        raise typer.Exit(code)

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
    values.update(
        mic=mic,
        max_duration=max_duration,
        stop_file=stop_file,
        final_pass=final_pass,
        min_silence=min_silence,
        max_segment=max_segment,
    )
    printer = SegmentPrinter()

    def func(params: LiveTranscribeParams):
        from uvt_transcribe.cli.common import log_segment
        from uvt_transcribe.live import live_transcribe as do_live

        return do_live(params, on_segment=log_segment if json_ else printer)

    raise typer.Exit(run(TOOL, func, values, opts, handle_sigterm=False))


def main() -> None:
    run_main(app, TOOL)


if __name__ == "__main__":
    main()
