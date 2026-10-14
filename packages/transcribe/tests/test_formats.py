import pytest

from uvt_transcribe.formats import (
    format_timestamp,
    parse_formats,
    parse_vtt,
    safe_filename,
    to_srt,
    to_txt,
    to_vtt,
)
from uvt_transcribe.models import Segment

SEGS = [
    Segment(id=0, start=0.0, end=1.5, text=" Hello there. "),
    Segment(id=1, start=3661.25, end=3662.0, text="Bye."),
]


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3661.25) == "01:01:01,250"
    assert format_timestamp(61.5, decimal=".", always_hours=False) == "01:01.500"


def test_srt_vtt_txt():
    assert to_srt(SEGS) == (
        "1\n00:00:00,000 --> 00:00:01,500\nHello there.\n\n2\n01:01:01,250 --> 01:01:02,000\nBye.\n"
    )
    assert to_vtt(SEGS).startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello there.\n")
    assert to_txt(SEGS) == "Hello there.\nBye.\n"


def test_parse_vtt_strips_tags_and_rolling_duplicates():
    vtt = """WEBVTT
Kind: captions

00:00:01.000 --> 00:00:02.000 align:start
<c>Hello</c><00:00:01.500><c> world</c>

00:00:02.000 --> 00:00:03.000
Hello world
second line &amp; more

1:02:03.000 --> 1:02:04.500
end
"""
    segs = parse_vtt(vtt)
    assert [s.text for s in segs] == ["Hello world", "second line & more", "end"]
    assert segs[2].start == pytest.approx(3723.0)
    assert segs[2].end == pytest.approx(3724.5)


def test_parse_vtt_roundtrip():
    assert [s.text for s in parse_vtt(to_vtt(SEGS))] == ["Hello there.", "Bye."]


def test_parse_formats():
    assert parse_formats(["txt,srt", "vtt", "srt"]) == ["txt", "srt", "vtt"]
    assert parse_formats(["none"]) == []
    assert parse_formats(None) is None
    with pytest.raises(ValueError, match="unknown format"):
        parse_formats(["docx"])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('a/b\\c:d*e?"f"<g>|h', "a_b_c_d_e__f__g__h"),
        ("CON", "_CON"),
        ("  trailing dots... ", "trailing dots"),
        ("", "transcript"),
    ],
)
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected
