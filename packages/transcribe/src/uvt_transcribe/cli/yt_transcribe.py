"""``yt-transcribe``: transcribe an online video (YouTube or any yt-dlp site)."""

from __future__ import annotations

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
from uvt_transcribe.options import YtTranscribeParams

TOOL = Tool(name="yt-transcribe", dist="uvt-transcribe", params_model=YtTranscribeParams)

app = new_app("Download only the audio of a video with yt-dlp and transcribe it.")


@app.command(epilog=EPILOG)
def yt_transcribe(
    url: Annotated[str | None, typer.Argument(help="Video URL.", show_default=False)] = None,
    prefer_subs: Annotated[
        bool | None,
        typer.Option(
            "--prefer-subs/--no-prefer-subs",
            help="If the video has manually created subtitles, use them and skip transcription.",
            show_default=False,
        ),
    ] = None,
    sub_langs: Annotated[
        list[str] | None,
        typer.Option(
            "--sub-lang",
            help="Subtitle language to look for (repeatable, in order of preference). "
            "Default: --language, the video language, then en.",
            show_default=False,
        ),
    ] = None,
    keep_audio: Annotated[
        bool | None,
        typer.Option(
            "--keep-audio/--no-keep-audio",
            help="Keep the downloaded audio next to the transcript.",
            rich_help_panel="Output",
            show_default=False,
        ),
    ] = None,
    cookies_from_browser: Annotated[
        str | None,
        typer.Option(
            "--cookies-from-browser",
            metavar="BROWSER",
            help="Use cookies from a browser (firefox, chrome…) if YouTube asks to sign in.",
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
    """Transcribe an online video: downloads only the audio (no ffmpeg needed).

    Works with YouTube and the other sites supported by yt-dlp. Output files are
    named "<title> \\[<id>].<format>" unless --name is given; the JobResult
    includes title, channel, duration and URL.

    [bold]Examples[/bold]

      yt-transcribe "https://www.youtube.com/watch?v=jNQXAC9IVRw"

      yt-transcribe URL --prefer-subs -l it -f txt,srt

      yt-transcribe URL --keep-audio -o ~/transcripts --json

      yt-transcribe URL --model tiny --device cpu --json
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
    values.update(
        url=url,
        prefer_subs=prefer_subs,
        sub_langs=sub_langs,
        keep_audio=keep_audio,
        cookies_from_browser=cookies_from_browser,
    )
    opts = CommonOptions(json=json_, job=job, print_job=print_job, verbose=verbose, quiet=quiet)
    printer = SegmentPrinter()

    def func(params: YtTranscribeParams):
        from uvt_transcribe.youtube import yt_transcribe as do_transcribe

        return do_transcribe(params, on_segment=None if json_ else printer)

    raise typer.Exit(run(TOOL, func, values, opts, on_success=printer.finish))


def main() -> None:
    run_main(app, TOOL)


if __name__ == "__main__":
    main()
