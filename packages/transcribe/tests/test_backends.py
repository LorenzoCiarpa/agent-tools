import importlib.util
import itertools
import json

import httpx
import numpy as np
import pytest

from uvt_core.errors import InvalidInputError, MissingDependencyError, ToolError
from uvt_transcribe.backends import available_backends, create_backend
from uvt_transcribe.backends.mistral_api import MistralApiBackend
from uvt_transcribe.backends.voxtral import speech_chunks


def test_registry_lists_builtin_backends():
    assert {"faster-whisper", "voxtral", "api"} <= set(available_backends())


def test_faster_whisper_rejects_unknown_option(jfk):
    backend = create_backend("faster-whisper", model="tiny", device="cpu", options={"nope": 1})
    with pytest.raises(InvalidInputError, match="Unknown faster-whisper option"):
        backend.load()


def test_faster_whisper_unknown_model():
    backend = create_backend("faster-whisper", model="gigantic-v9", device="cpu")
    with pytest.raises(InvalidInputError, match="Unknown faster-whisper model"):
        backend.load()


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="torch is installed")
def test_voxtral_without_extra_explains_install():
    with pytest.raises(MissingDependencyError) as exc:
        create_backend("voxtral").load()
    assert "[voxtral]" in exc.value.hint and "--torch-backend auto" in exc.value.hint


def test_speech_chunks_cut_on_pauses(jfk_samples):
    chunks = speech_chunks(jfk_samples, 4.0)
    assert len(chunks) >= 3
    assert all(end - start <= 4.5 * 16000 for start, end in chunks)
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(chunks))
    assert speech_chunks(np.zeros(16000 * 5, np.float32), 30.0) == []
    assert speech_chunks(np.zeros(16000 * 5, np.float32), 2.0, vad=False) == [
        (0, 32000),
        (32000, 64000),
        (64000, 80000),
    ]


def _api_backend(monkeypatch, handler, **kwargs):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    backend = MistralApiBackend(**kwargs)
    backend.transport = httpx.MockTransport(handler)
    return backend


def _multipart_fields(request: httpx.Request) -> dict[str, list[str]]:
    body = request.read().decode("latin-1")
    fields: dict[str, list[str]] = {}
    for part in body.split("--" + request.headers["content-type"].split("boundary=")[1]):
        if 'name="' not in part:
            continue
        name = part.split('name="')[1].split('"')[0]
        value = part.split("\r\n\r\n", 1)[1].rsplit("\r\n", 1)[0]
        fields.setdefault(name, []).append(value if name != "file" else f"<{len(value)} bytes>")
    return fields


def test_api_backend_request_and_segments(monkeypatch):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["fields"] = _multipart_fields(request)
        return httpx.Response(
            200,
            json={
                "model": "voxtral-mini-2602",
                "text": "ciao mondo",
                "language": "it",
                "segments": [
                    {"text": "ciao", "start": 0.1, "end": 0.5, "speaker_id": "speaker_1"},
                    {"text": "mondo", "start": 0.6, "end": 1.0},
                ],
            },
        )

    backend = _api_backend(monkeypatch, handler, options={"diarize": "true"})
    transcript = backend.transcribe(np.zeros(16000, np.float32))
    assert seen["auth"] == "Bearer test-key"
    assert seen["fields"]["model"] == ["voxtral-mini-latest"]
    assert seen["fields"]["timestamp_granularities"] == ["segment"]
    assert seen["fields"]["diarize"] == ["true"]
    assert "language" not in seen["fields"]
    assert [s.text for s in transcript.segments] == ["ciao", "mondo"]
    assert transcript.segments[0].speaker == "speaker_1"
    assert transcript.language == "it"
    assert backend.device_info.kind == "remote"


def test_api_backend_language_means_no_timestamps_and_chunking(monkeypatch):
    calls = []

    def handler(request):
        calls.append(_multipart_fields(request))
        return httpx.Response(200, json={"text": f"part {len(calls)}", "segments": []})

    backend = _api_backend(monkeypatch, handler, options={"chunk_seconds": "1"})
    transcript = backend.transcribe(np.zeros(int(16000 * 2.5), np.float32), language="it")
    assert len(calls) == 3
    assert calls[0]["language"] == ["it"] and "timestamp_granularities" not in calls[0]
    assert [(s.start, s.end, s.text) for s in transcript.segments] == [
        (0.0, 1.0, "part 1"),
        (1.0, 2.0, "part 2"),
        (2.0, 2.5, "part 3"),
    ]


def test_api_backend_errors(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ToolError, match="MISTRAL_API_KEY"):
        MistralApiBackend().load()

    backend = _api_backend(monkeypatch, lambda r: httpx.Response(401, json={"detail": "no"}))
    with pytest.raises(ToolError, match="rejected the API key"):
        backend.transcribe(np.zeros(1600, np.float32))

    monkeypatch.setattr("time.sleep", lambda s: None)
    attempts = []

    def flaky(request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, content=json.dumps({"text": "ok"}).encode())

    backend = _api_backend(monkeypatch, flaky)
    assert backend.transcribe(np.zeros(1600, np.float32)).segments[0].text == "ok"
    assert len(attempts) == 3
