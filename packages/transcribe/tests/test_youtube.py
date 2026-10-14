import shutil
from pathlib import Path
from typing import ClassVar

import pytest

from uvt_core.errors import InvalidInputError, MissingDependencyError, ToolError
from uvt_transcribe import youtube
from uvt_transcribe.options import YtTranscribeParams

INFO = {
    "id": "abc123",
    "title": "A talk: part 1/2",
    "channel": "Chan",
    "uploader": "Up",
    "upload_date": "20240101",
    "duration": 11.0,
    "webpage_url": "https://www.youtube.com/watch?v=abc123",
    "extractor_key": "Youtube",
    "language": "it",
    "subtitles": {
        "live_chat": [{"ext": "json", "url": "u0"}],
        "en-US": [{"ext": "json3", "url": "u1"}, {"ext": "vtt", "url": "u2"}],
        "it": [{"ext": "srv3", "url": "u3"}, {"ext": "srt", "url": "u4"}],
    },
}


def test_is_youtube():
    assert youtube.is_youtube("https://youtu.be/x")
    assert youtube.is_youtube("https://m.youtube.com/watch?v=x")
    assert not youtube.is_youtube("https://vimeo.com/1")
    assert not youtube.is_youtube("https://notyoutube.com/x")


def test_pick_manual_subtitle():
    assert youtube.pick_manual_subtitle(INFO, None, None)[0] == "it"  # video language
    assert youtube.pick_manual_subtitle(INFO, None, "en")[0] == "en-US"  # prefix match
    assert youtube.pick_manual_subtitle(INFO, ["en"], None)[1]["url"] == "u2"  # vtt preferred
    assert youtube.pick_manual_subtitle(INFO, ["de"], None) is None  # explicit, no match
    assert youtube.pick_manual_subtitle({**INFO, "subtitles": {}}, None, None) is None


def test_map_download_error():
    assert isinstance(
        youtube._map_download_error(Exception("ERROR: Unsupported URL: x"), "x"), InvalidInputError
    )
    bot = youtube._map_download_error(Exception("Sign in to confirm you're not a bot"), "x")
    assert type(bot) is ToolError and "cookies" in bot.hint


class FakeYDL:
    """Stands in for yt_dlp.YoutubeDL: 'downloads' the JFK sample."""

    last_params: ClassVar[dict] = {}

    def __init__(self, params):
        FakeYDL.last_params = params
        self.params = params

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        return dict(INFO)

    def process_ie_result(self, info, download=True):
        target = Path(self.params["paths"]["home"]) / "abc123.ogg"
        shutil.copy(self.jfk, target)
        return {**info, "requested_downloads": [{"filepath": str(target)}]}

    def urlopen(self, url):
        import io

        body = b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nCiao a tutti\n"
        return io.BytesIO(body)


@pytest.fixture
def fake_ydl(monkeypatch, jfk):
    import yt_dlp

    FakeYDL.jfk = jfk
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(youtube, "find_js_runtime", lambda: {"deno": {"path": "deno"}})
    return FakeYDL


def test_yt_transcribe_downloads_and_transcribes(fake_ydl, tmp_path):
    params = YtTranscribeParams(
        url=INFO["webpage_url"],
        backend="fake",
        formats=["txt"],
        output_dir=tmp_path,
        keep_audio=True,
    )
    result = youtube.yt_transcribe(params)
    assert result.text == "chunk 0" and not result.from_subtitles
    assert result.source.title == "A talk: part 1/2" and result.source.kind == "youtube"
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["A talk_ part 1_2 [abc123].ogg", "A talk_ part 1_2 [abc123].txt"]
    assert [a.name for a in result.artifacts] == ["txt", "audio"]
    assert fake_ydl.last_params["fixup"] == "never"
    assert fake_ydl.last_params["format"] == "bestaudio/best"
    assert "postprocessors" not in fake_ydl.last_params


def test_yt_transcribe_prefers_manual_subs(fake_ydl, tmp_path):
    params = YtTranscribeParams(
        url=INFO["webpage_url"],
        backend="fake",
        prefer_subs=True,
        formats=["srt"],
        output_dir=tmp_path,
    )
    result = youtube.yt_transcribe(params)
    assert result.from_subtitles and result.backend is None
    assert result.text == "Ciao a tutti" and result.language == "it"


def test_yt_transcribe_requires_js_runtime(monkeypatch, fake_ydl):
    monkeypatch.setattr(youtube, "find_js_runtime", lambda: None)
    with pytest.raises(MissingDependencyError, match="JavaScript runtime") as exc:
        youtube.yt_transcribe(YtTranscribeParams(url=INFO["webpage_url"], backend="fake"))
    assert "deno" in exc.value.hint.lower()


def test_yt_transcribe_rejects_non_urls():
    with pytest.raises(InvalidInputError, match="http"):
        youtube.yt_transcribe(YtTranscribeParams(url="not a url", backend="fake"))
