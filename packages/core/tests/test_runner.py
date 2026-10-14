import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from uvt_core.errors import InvalidInputError, ToolError
from uvt_core.jobs import JobSpec, ToolOutput
from uvt_core.output import stdout_to_stderr
from uvt_core.runner import CommonOptions, Tool, execute_job, run


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")
    word: str = "hi"
    fail: str | None = None


class Out(ToolOutput):
    length: int


def func(p: Params) -> Out:
    if p.fail == "tool":
        raise ToolError("boom", hint="do X")
    if p.fail == "input":
        raise InvalidInputError("bad input")
    if p.fail == "crash":
        raise RuntimeError("unexpected")
    print("noise from a library")  # must never reach stdout with --json
    return Out(text=p.word.upper(), length=len(p.word))


TOOL = Tool(name="demo", dist="uvt-core", params_model=Params)


@pytest.mark.parametrize(
    ("fail", "code", "etype"),
    [
        (None, 0, None),
        ("tool", 1, "error"),
        ("input", 2, "invalid_input"),
        ("crash", 1, "internal"),
    ],
)
def test_run_json_exit_codes(capfd, fail, code, etype):
    rc = run(TOOL, func, {"word": "abc", "fail": fail}, CommonOptions(json=True))
    out, err = capfd.readouterr()
    assert rc == code
    result = json.loads(out)  # stdout is exactly one JSON document
    assert result["ok"] is (code == 0)
    assert result["exit_code"] == code
    if etype:
        assert result["error"]["type"] == etype
    else:
        assert result["text"] == "ABC"
        assert result["data"] == {"length": 3}
        assert "noise from a library" in err


def test_run_job_file_and_print_job(tmp_path: Path, capfd):
    job = tmp_path / "job.json"
    job.write_text(JobSpec(tool="demo", params={"word": "xyz"}).model_dump_json(), encoding="utf-8")
    assert run(TOOL, func, {}, CommonOptions(json=True, job=str(job))) == 0
    assert json.loads(capfd.readouterr().out)["text"] == "XYZ"

    assert run(TOOL, func, {"word": "override"}, CommonOptions(job=str(job), print_job=True)) == 0
    spec = JobSpec.model_validate_json(capfd.readouterr().out)
    assert spec.params["word"] == "override"
    assert spec.tool == "demo"


def test_run_invalid_job_reports_json(tmp_path: Path, capfd):
    rc = run(TOOL, func, {}, CommonOptions(json=True, job=str(tmp_path / "missing.json")))
    assert rc == 2
    assert json.loads(capfd.readouterr().out)["error"]["type"] == "invalid_input"


def test_execute_job_never_raises():
    result = execute_job(TOOL, func, JobSpec(tool="demo", params={"nope": 1}))
    assert not result.ok and result.exit_code == 2


def test_stdout_guard_catches_c_level_writes(capfd):
    import os

    with stdout_to_stderr() as real:
        os.write(1, b"c-level\n")
        real.write("result\n")
    out, err = capfd.readouterr()
    assert out == "result\n"
    assert "c-level" in err


def test_uvt_info_cli_json():
    proc = subprocess.run(
        [sys.executable, "-m", "uvt_core.info", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["tool"] == "uvt-info" and data["device"]["kind"] in {"cuda", "mps", "cpu"}


def test_cli_usage_error_exit_2_with_json():
    proc = subprocess.run(
        [sys.executable, "-m", "uvt_core.info", "--json", "--bogus"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["error"]["type"] == "invalid_input"
