# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Create a new tool package that follows the repo conventions.

    uv run scripts/new_package.py pdf --command pdf-extract --description "Extract text from PDFs"
    uv sync --all-packages
    uv run pdf-extract --help

Generates packages/<name>/ with: pyproject (uv_build, uvt-core from the
workspace, a console script), README with the install command, a typed
``api.py`` (params model + logic function), a thin Typer ``cli.py`` wired to
the shared runner (--json, --job, --print-job, exit codes) and tests.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/LorenzoCiarpa/agent-tools"
NAME_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


def render(name: str, command: str, description: str) -> dict[str, str]:
    dist = f"uvt-{name}"
    module = f"uvt_{name.replace('-', '_')}"
    func = command.replace("-", "_")
    return {
        "pyproject.toml": f'''[project]
name = "{dist}"
version = "0.1.0"
description = "{description}"
readme = "README.md"
license = "MIT"
authors = [{{ name = "Lorenzo Ciarpaglini" }}]
requires-python = ">=3.11"
# `uv tool install` does not use the workspace lockfile: keep ranges explicit.
dependencies = [
    "uvt-core>=0.1,<0.2",
]

[project.scripts]
{command} = "{module}.cli:main"

[project.urls]
Repository = "{REPO_URL}"

[tool.uv.sources]
uvt-core = {{ workspace = true }}

[build-system]
requires = ["uv_build>=0.12,<0.14"]
build-backend = "uv_build"
''',
        "README.md": f"""# {dist}

{description}

## Install

```bash
uv tool install "git+{REPO_URL}#subdirectory=packages/{name}"
```

## Usage

```bash
{command} --help
{command} "hello" --json
```

`--json` prints a single `JobResult` on stdout. `--job job.json` reads the
parameters from a `JobSpec`, and `--print-job` prints one. Exit codes are
0 ok, 1 error, 2 invalid input. See the repository's AGENTS.md.
""",
        f"src/{module}/__init__.py": f'''"""{description}"""

from uvt_core.jobs import package_version

__version__ = package_version("{dist}")
''',
        f"src/{module}/py.typed": "",
        f"src/{module}/api.py": f'''"""Typed logic of `{command}`: params in, result out.

Keep heavy imports inside functions so `--help` stays fast.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from uvt_core.errors import InvalidInputError
from uvt_core.jobs import ToolOutput


class {_camel(func)}Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Input text.")
    upper: bool = Field(default=False, description="Upper-case the result.")


class {_camel(func)}Result(ToolOutput):
    length: int


def {func}(params: {_camel(func)}Params) -> {_camel(func)}Result:
    """Replace this example with the real logic."""
    if not params.text.strip():
        raise InvalidInputError("Empty text.", hint="Pass some text.")
    text = params.text.upper() if params.upper else params.text
    return {_camel(func)}Result(text=text, length=len(text))
''',
        f"src/{module}/cli.py": f'''"""`{command}`: thin Typer wrapper around `{module}.api`."""

from __future__ import annotations

from typing import Annotated

import typer

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
from {module}.api import {_camel(func)}Params, {func} as do_{func}

TOOL = Tool(name="{command}", dist="{dist}", params_model={_camel(func)}Params)

app = new_app("{description}")


@app.command(epilog=EXIT_CODES_HELP)
def {func}(
    text: Annotated[str | None, typer.Argument(help="Input text.", show_default=False)] = None,
    upper: Annotated[
        bool | None, typer.Option("--upper/--no-upper", help="Upper-case.", show_default=False)
    ] = None,
    json_: JsonOption = False,
    job: JobOption = None,
    print_job: PrintJobOption = False,
    verbose: VerboseOption = 0,
    quiet: QuietOption = False,
) -> None:
    """{description}

    [bold]Examples[/bold]

      {command} "hello"

      {command} "hello" --upper --json
    """
    opts = CommonOptions(json=json_, job=job, print_job=print_job, verbose=verbose, quiet=quiet)
    values = {{"text": text, "upper": upper}}
    code = run(TOOL, do_{func}, values, opts, on_success=lambda out: typer.echo(out.text))
    raise typer.Exit(code)


def main() -> None:
    run_main(app, TOOL)


if __name__ == "__main__":
    main()
''',
        "tests/test_api.py": f'''import json
import subprocess
import sys

import pytest

from uvt_core.errors import InvalidInputError
from {module}.api import {_camel(func)}Params, {func}


def test_{func}():
    assert {func}({_camel(func)}Params(text="hi", upper=True)).text == "HI"


def test_{func}_rejects_empty_text():
    with pytest.raises(InvalidInputError):
        {func}({_camel(func)}Params(text=" "))


def test_cli_json():
    proc = subprocess.run(
        [sys.executable, "-m", "{module}.cli", "hi", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["data"] == {{"length": 2}}
''',
    }


def _camel(snake: str) -> str:
    return "".join(part.capitalize() for part in snake.split("_"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="Package name without prefix, e.g. 'pdf' -> uvt-pdf")
    parser.add_argument("--command", help="Console script name (default: the package name)")
    parser.add_argument("--description", default="A uvt tool.", help="One-line description")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if not NAME_RE.match(args.name):
        parser.error("name must be lowercase letters/digits separated by single dashes")
    command = args.command or args.name
    if not NAME_RE.match(command):
        parser.error("command must be lowercase letters/digits separated by single dashes")
    target = args.root / "packages" / args.name
    if target.exists():
        print(f"error: {target} already exists", file=sys.stderr)
        return 1

    for rel, content in render(args.name, command, args.description).items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"created {path.relative_to(args.root)}")
    print(
        f"\nNext:\n  uv sync --all-packages\n  uv run {command} --help\n"
        f"  uv run pytest packages/{args.name}\n"
        "Then add the package to README.md and to the CI matrix in .github/workflows/ci.yml."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
