import json
from pathlib import Path

import pytest

from uvt_core.errors import InvalidInputError
from uvt_core.jobs import JobSpec
from uvt_core.runner import main
from uvt_transcribe.api import TranscribeFileParams, transcribe_file
from uvt_transcribe.cli import transcribe_file as cli
from uvt_transcribe.options import resolve_options


def test_resolve_options_precedence(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[transcribe]\nformats = ["srt"]\nlanguage = "it"\nbeam_size = 3\n')
    monkeypatch.setenv("UVT_CONFIG_FILE", str(cfg))
    monkeypatch.setenv("UVT_TRANSCRIBE__BEAM_SIZE", "4")
    opts, table = resolve_options(TranscribeFileParams(media=Path("x"), language="auto"))
    assert opts.formats == ["srt"]  # config file
    assert opts.beam_size == 4  # env beats the file
    assert opts.language is None  # explicit "auto" beats the config
    assert opts.backend == "faster-whisper" and opts.vad is True  # built-in defaults
    assert table["language"] == "it"


def test_invalid_config_value_is_invalid_input(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("UVT_TRANSCRIBE__FORMATS", '["docx"]')
    with pytest.raises(InvalidInputError, match="Invalid transcription settings"):
        resolve_options(TranscribeFileParams(media=Path("x")))


def test_transcribe_file_with_fake_backend(jfk, tmp_path):
    seen = []
    params = TranscribeFileParams(
        media=jfk, backend="fake", formats=["txt", "srt", "json"], output_dir=tmp_path, name="out"
    )
    result = transcribe_file(params, on_segment=seen.append)
    assert result.text == "chunk 0"
    assert result.backend == "fake" and result.model == "fake-model"
    assert result.duration == pytest.approx(11.0, abs=0.1)
    assert [a.name for a in result.artifacts] == ["txt", "srt", "json"]
    assert (tmp_path / "out.txt").read_text() == "chunk 0\n"
    assert json.loads((tmp_path / "out.json").read_text())["segments"][0]["text"] == "chunk 0"
    assert len(seen) == 1


def test_transcribe_file_missing_input(tmp_path):
    with pytest.raises(InvalidInputError, match="not found"):
        transcribe_file(TranscribeFileParams(media=tmp_path / "nope.wav", backend="fake"))


def test_transcribe_file_undecodable_input(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"definitely not audio")
    with pytest.raises(InvalidInputError, match="Cannot decode"):
        transcribe_file(TranscribeFileParams(media=bad, backend="fake", formats=[]))


def test_unknown_backend():
    with pytest.raises(InvalidInputError, match="Unknown backend"):
        transcribe_file(TranscribeFileParams(media=Path(__file__), backend="nope"))


def _cli(argv, capfd):
    with pytest.raises(SystemExit) as exc:
        main(cli.app, cli.TOOL, argv)
    out, err = capfd.readouterr()
    return exc.value.code, out, err


def test_cli_json_result(jfk, tmp_path, capfd):
    code, out, err = _cli(
        [str(jfk), "-b", "fake", "-f", "srt", "-o", str(tmp_path), "--json"], capfd
    )
    assert code == 0, err
    result = json.loads(out)
    assert result["ok"] and result["tool"] == "transcribe-file"
    assert result["text"] == "chunk 0"
    assert result["data"]["segments"][0]["end"] == pytest.approx(11.0, abs=0.1)
    assert result["artifacts"][0]["name"] == "srt"
    assert result["host"]["hostname"]


def test_cli_human_mode_streams_text(jfk, tmp_path, capfd):
    code, out, _ = _cli([str(jfk), "-b", "fake", "-f", "none", "-o", str(tmp_path)], capfd)
    assert code == 0
    assert out == "chunk 0\n"


def test_cli_invalid_inputs_exit_2(tmp_path, capfd):
    code, out, _ = _cli([str(tmp_path / "missing.wav"), "-b", "fake", "--json"], capfd)
    assert code == 2 and json.loads(out)["error"]["type"] == "invalid_input"
    code, out, _ = _cli(["x.wav", "-f", "docx", "--json"], capfd)
    assert code == 2 and "unknown format" in json.loads(out)["error"]["message"]
    code, out, _ = _cli(["--json"], capfd)  # no input at all
    assert code == 2 and "media" in json.loads(out)["error"]["message"]


def test_cli_job_roundtrip(jfk, tmp_path, capfd):
    code, out, _ = _cli([str(jfk), "-b", "fake", "-l", "it", "-f", "txt", "--print-job"], capfd)
    assert code == 0
    job = JobSpec.model_validate_json(out)
    assert job.tool == "transcribe-file"
    assert job.inputs["media"].path == str(jfk.resolve())
    assert job.params == {
        "backend": "fake",
        "language": "it",
        "formats": ["txt"],
        "backend_options": {},
    }

    job.params["output_dir"] = str(tmp_path)
    job_file = tmp_path / "job.json"
    job_file.write_text(job.model_dump_json())
    code, out, err = _cli(["--job", str(job_file), "--json"], capfd)
    assert code == 0, err
    result = json.loads(out)
    assert result["job_id"] == job.id
    assert Path(result["artifacts"][0]["path"]) == tmp_path / "jfk.txt"
