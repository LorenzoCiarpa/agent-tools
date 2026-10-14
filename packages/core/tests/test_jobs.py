import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from uvt_core.errors import InvalidInputError
from uvt_core.jobs import (
    Artifact,
    InputRef,
    JobSpec,
    input_field,
    input_fields,
    job_from_params,
    load_job,
)
from uvt_core.runner import Tool, build_params, check_tool_version


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")
    media: Path = input_field(description="file")
    language: str | None = None
    beam_size: int = 5


TOOL = Tool(name="demo", dist="uvt-core", params_model=Params)


def test_input_ref_requires_exactly_one():
    with pytest.raises(ValidationError):
        InputRef()
    with pytest.raises(ValidationError):
        InputRef(path="a", url="http://b")
    assert InputRef(url="https://x").value == "https://x"


def test_input_fields_detected():
    assert input_fields(Params) == ["media"]


def test_job_from_params_splits_inputs(tmp_path: Path):
    media = tmp_path / "a.wav"
    media.write_bytes(b"1234")
    job = job_from_params("demo", Params(media=media, language="it"), tool_version="==0.1.*")
    assert job.params == {"language": "it", "beam_size": 5}
    assert job.inputs["media"].path == str(media.resolve())
    assert job.inputs["media"].size == 4
    # Round-trips through JSON.
    again = JobSpec.model_validate_json(job.model_dump_json())
    assert again == job


def test_build_params_job_then_cli_override(tmp_path: Path):
    job = JobSpec(
        tool="demo",
        params={"language": "en", "beam_size": 2},
        inputs={"media": InputRef(path="a.wav")},
    )
    params = build_params(TOOL, {"beam_size": 7, "language": None}, job, base_dir=tmp_path)
    assert params.media == tmp_path / "a.wav"  # relative inputs resolve against the job dir
    assert params.language == "en"  # None on the CLI means "not given"
    assert params.beam_size == 7


def test_build_params_rejects_wrong_tool_and_unknown_keys():
    with pytest.raises(InvalidInputError, match="for 'other'"):
        build_params(TOOL, {}, JobSpec(tool="other"))
    with pytest.raises(InvalidInputError, match="Invalid parameters"):
        build_params(TOOL, {"media": "a", "nope": 1})
    with pytest.raises(InvalidInputError, match="Unknown input"):
        build_params(TOOL, {}, JobSpec(tool="demo", inputs={"x": InputRef(path="a")}))


def test_check_tool_version():
    check_tool_version("==0.1.*", "0.1.3")
    check_tool_version("0.1.3", "0.1.3")
    check_tool_version(">=0.1,<0.2", "0.1.0")
    with pytest.raises(InvalidInputError, match="requires version"):
        check_tool_version(">=0.2", "0.1.0")
    with pytest.raises(InvalidInputError, match="Invalid tool_version"):
        check_tool_version("~~1", "0.1.0")


def test_load_job_file_and_errors(tmp_path: Path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"tool": "demo", "params": {"beam_size": 1}}), encoding="utf-8")
    job, base = load_job(path)
    assert job.tool == "demo" and base == tmp_path.resolve()
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="not valid JSON"):
        load_job(path)
    path.write_text(json.dumps({"tool": "demo", "bogus": 1}), encoding="utf-8")
    with pytest.raises(InvalidInputError, match="not a valid JobSpec"):
        load_job(path)
    with pytest.raises(InvalidInputError, match="not found"):
        load_job(tmp_path / "missing.json")


def test_artifact_from_path(tmp_path: Path):
    f = tmp_path / "out.srt"
    f.write_text("hello", encoding="utf-8")
    art = Artifact.from_path("srt", f)
    assert art.size == 5
    assert art.sha256 == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
