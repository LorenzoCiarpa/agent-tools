"""``uvt-info``: show what this machine offers (device, config) as JSON or text."""

from __future__ import annotations

from typing import Any

import typer
from pydantic import BaseModel, ConfigDict

from uvt_core import device as dev
from uvt_core.config import config_path, load_config
from uvt_core.jobs import ToolOutput
from uvt_core.runner import (
    EXIT_CODES_HELP,
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


class InfoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InfoOutput(ToolOutput):
    config_path: str
    config_exists: bool
    config: dict[str, Any]
    cuda_devices: list[str]


TOOL = Tool(name="uvt-info", dist="uvt-core", params_model=InfoParams)


def machine_info(_params: InfoParams | None = None) -> InfoOutput:
    """Detected compute device and effective configuration of this machine."""
    kind = dev.detect_device("auto")
    path = config_path()
    return InfoOutput(
        device=dev.describe(kind),
        config_path=str(path),
        config_exists=path.is_file(),
        config=load_config(),
        cuda_devices=list(dev.cuda_device_names()),
    )


def _render(output: ToolOutput) -> None:
    assert isinstance(output, InfoOutput)
    assert output.device is not None
    typer.echo(f"device:      {output.device.kind} ({output.device.name})")
    typer.echo(f"cuda:        {', '.join(output.cuda_devices) or 'none'}")
    state = "" if output.config_exists else " (not created yet)"
    typer.echo(f"config file: {output.config_path}{state}")
    for section, values in output.config.items():
        typer.echo(f"  {section} = {values}")


app = new_app("Show the compute device and uvt configuration of this machine.")


@app.command(epilog=EXIT_CODES_HELP)
def info(
    json_: JsonOption = False,
    job: JobOption = None,
    print_job: PrintJobOption = False,
    verbose: VerboseOption = 0,
    quiet: QuietOption = False,
) -> None:
    """Show the compute device and uvt configuration of this machine.

    [bold]Examples[/bold]

      uvt-info

      uvt-info --json
    """
    opts = CommonOptions(json=json_, job=job, print_job=print_job, verbose=verbose, quiet=quiet)
    raise typer.Exit(run(TOOL, machine_info, {}, opts, on_success=_render))


def main() -> None:
    run_main(app, TOOL)


if __name__ == "__main__":
    main()
