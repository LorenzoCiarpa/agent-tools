"""Real models and real network. Skipped by default; run with e.g.

uv run pytest -m slow                 # tiny model on CPU (downloads ~75 MB once)
uv run pytest -m "slow and gpu"       # also on CUDA
uv run pytest -m "slow and network"   # real YouTube download
"""

import json
import re
import subprocess
import sys

import pytest

from uvt_transcribe.api import (
    LiveTranscribeParams,
    TranscribeFileParams,
    YtTranscribeParams,
    live_transcribe,
    transcribe_file,
    yt_transcribe,
)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower())


@pytest.mark.slow
def test_tiny_model_on_cpu(jfk, tmp_path):
    params = TranscribeFileParams(
        media=jfk, model="tiny", device="cpu", formats=["srt"], output_dir=tmp_path
    )
    result = transcribe_file(params)
    assert "ask not what your country can do for you" in _normalize(result.text)
    assert result.language == "en"
    assert result.device.kind == "cpu" and result.device.compute_type == "int8"
    assert (tmp_path / "jfk.srt").read_text().startswith("1\n00:00:00,000 --> ")


@pytest.mark.slow
@pytest.mark.gpu
def test_tiny_model_on_cuda(jfk, tmp_path):
    from uvt_core.device import cuda_device_count

    if cuda_device_count() == 0:
        pytest.skip("no CUDA device")
    params = TranscribeFileParams(
        media=jfk, model="tiny", device="cuda", formats=[], output_dir=tmp_path
    )
    result = transcribe_file(params)
    assert result.device.kind == "cuda" and result.device.compute_type == "float16"
    assert "ask not what your country" in _normalize(result.text)


@pytest.mark.slow
def test_live_pipeline_with_real_model(fake_sd, tmp_path):
    """Fake microphone playing JFK -> real VAD -> real tiny model."""
    params = LiveTranscribeParams(
        model="tiny",
        device="cpu",
        max_duration=14,
        output_dir=tmp_path,
        name="rec",
        formats=["txt"],
        language="en",
    )
    result = live_transcribe(params)
    text = _normalize(result.text)
    assert "my fellow americans" in text and "your country" in text
    assert len(result.segments) >= 2
    assert (tmp_path / "rec.live.txt").read_text().count("\n") == len(result.segments)


@pytest.mark.slow
def test_cli_json_end_to_end(jfk, tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "uvt_transcribe.cli.transcribe_file",
            str(jfk),
            "-m",
            "tiny",
            "--device",
            "cpu",
            "-o",
            str(tmp_path),
            "-f",
            "json",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["ok"] and "country" in result["text"].lower()


@pytest.mark.slow
@pytest.mark.network
def test_youtube_real_video(tmp_path):
    params = YtTranscribeParams(
        url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
        model="tiny",
        device="cpu",
        formats=["txt"],
        output_dir=tmp_path,
        keep_audio=True,
    )
    result = yt_transcribe(params)
    assert result.source.title == "Me at the zoo"
    assert "elephant" in result.text.lower()
    assert any(a.name == "audio" for a in result.artifacts)
