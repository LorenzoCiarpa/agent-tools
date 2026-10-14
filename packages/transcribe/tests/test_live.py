import json
import threading
import wave

import pytest

from uvt_core.errors import InvalidInputError
from uvt_transcribe import live
from uvt_transcribe.options import LiveTranscribeParams


def test_list_and_resolve_mics(fake_sd):
    mics = live.list_microphones()
    assert [m.name for m in mics] == ["Built-in Mic", "USB Mic"]  # output-only device skipped
    assert live.resolve_mic(None).name == "Built-in Mic"  # system default
    assert live.resolve_mic("usb").index == 2
    assert live.resolve_mic("2").name == "USB Mic"
    with pytest.raises(InvalidInputError):
        live.resolve_mic("nope")
    with pytest.raises(InvalidInputError):
        live.resolve_mic("0")  # not an input device


def test_live_session_segments_on_pauses(fake_sd, fake_backends, tmp_path):
    seen = []
    params = LiveTranscribeParams(
        backend="fake", max_duration=15, output_dir=tmp_path, name="rec", formats=["txt", "json"]
    )
    result = live.live_transcribe(params, on_segment=seen.append)

    backend = fake_backends[-1]
    assert backend.closed
    # JFK's sentence has pauses: more than one segment, in order, inside the recording.
    assert len(result.segments) >= 2
    assert all(
        a.end <= b.start + 1e-6 for a, b in zip(result.segments, result.segments[1:], strict=False)
    )
    assert result.segments[-1].end <= result.duration + 1e-6
    assert [s.text for s in seen] == [s.text for s in result.segments]
    # Context: every call after the first gets the previous text as initial_prompt.
    assert backend.calls[0]["initial_prompt"] is None
    assert backend.calls[1]["initial_prompt"] == "chunk 0"
    assert all(c["vad"] is False for c in backend.calls)

    with wave.open(str(tmp_path / "rec.wav")) as wf:
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1
        assert 15 <= wf.getnframes() / 16000 < 16
    log_lines = (tmp_path / "rec.live.txt").read_text().splitlines()
    assert len(log_lines) == len(result.segments) and log_lines[0].endswith("chunk 0")
    assert json.loads((tmp_path / "rec.json").read_text())["source"]["kind"] == "microphone"
    assert {a.name for a in result.artifacts} == {"audio", "live_log", "txt", "json"}


def test_live_stop_file_and_final_pass(fake_sd, fake_backends, tmp_path):
    stop = tmp_path / "stop"
    params = LiveTranscribeParams(
        backend="fake", stop_file=stop, output_dir=tmp_path, name="rec", formats=[], final_pass=True
    )
    timer = threading.Timer(1.0, stop.touch)
    timer.start()
    result = live.live_transcribe(params)
    timer.join()
    backend = fake_backends[-1]
    final_call = backend.calls[-1]
    assert final_call["vad"] is True  # the final pass runs on the whole file
    assert final_call["duration"] == pytest.approx(result.duration, abs=0.01)
    assert len(result.segments) == 1  # final pass result replaces the live segments


def test_live_external_stop_event(fake_sd, tmp_path):
    event = threading.Event()
    threading.Timer(0.5, event.set).start()
    params = LiveTranscribeParams(backend="fake", output_dir=tmp_path, formats=[])
    result = live.live_transcribe(params, stop_event=event)
    assert result.duration > 0


def test_existing_stop_file_is_rejected(fake_sd, tmp_path):
    stop = tmp_path / "stop"
    stop.touch()
    with pytest.raises(InvalidInputError, match="already exists"):
        live.live_transcribe(LiveTranscribeParams(backend="fake", stop_file=stop))


def test_missing_portaudio_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(live.MissingDependencyError, match="PortAudio") as exc:
        live.import_sounddevice()
    assert exc.value.hint
