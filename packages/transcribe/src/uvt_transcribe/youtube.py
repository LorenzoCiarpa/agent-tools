"""``yt-transcribe``: download only the audio with yt-dlp, then transcribe it.

* Audio only (``bestaudio``), no post-processing and ``fixup=never``: no
  ffmpeg needed, the file is decoded with PyAV.
* ``prefer_subs``: if the video has *manually created* subtitles, use them
  and skip the model entirely.
* YouTube needs a JavaScript runtime for yt-dlp's challenge solver
  (yt-dlp-ejs). The ``deno`` wheel is a dependency of this package, so it is
  normally inside the tool's venv; otherwise a ``deno`` or ``node`` on PATH is used.
"""

from __future__ import annotations

import shutil
import sysconfig
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from uvt_core.errors import InvalidInputError, MissingDependencyError, ToolError
from uvt_core.jobs import Artifact
from uvt_core.logging import get_logger
from uvt_transcribe.backends import TranscriptionBackend
from uvt_transcribe.backends.base import SegmentCallback
from uvt_transcribe.formats import parse_vtt, safe_filename, write_outputs
from uvt_transcribe.models import SourceInfo, Transcript, TranscriptionResult
from uvt_transcribe.options import YtTranscribeParams, formats, output_dir, resolve_options
from uvt_transcribe.pipeline import build_result, run_backend

log = get_logger("transcribe.youtube")

_YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com")
_SUB_FORMATS = ("vtt", "srt")

DENO_INSTALL = {
    "Linux": "Install Deno: curl -fsSL https://deno.land/install.sh | sh  (then restart the shell)",
    "Darwin": "Install Deno: brew install deno  (or curl -fsSL https://deno.land/install.sh | sh)",
    "Windows": "Install Deno: winget install DenoLand.Deno  "
    "(or in PowerShell: irm https://deno.land/install.ps1 | iex)",
}


def is_youtube(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in _YOUTUBE_HOSTS)


def find_js_runtime() -> dict[str, dict[str, str]] | None:
    """Locate a JS runtime usable by yt-dlp, as a ``js_runtimes`` param value."""
    scripts = sysconfig.get_path("scripts")
    candidates = [
        ("deno", shutil.which("deno", path=scripts) if scripts else None),
        ("deno", shutil.which("deno")),
        ("deno", shutil.which("deno", path=str(Path.home() / ".deno" / "bin"))),
        ("node", shutil.which("node")),
    ]
    for name, path in candidates:
        if path:
            return {name: {"path": path}}
    return None


class _YdlLogger:
    """Routes yt-dlp messages to our stderr logger (never to stdout)."""

    def debug(self, msg: str) -> None:
        log.debug("%s", msg)

    def info(self, msg: str) -> None:
        log.debug("%s", msg)

    def warning(self, msg: str) -> None:
        log.warning("yt-dlp: %s", msg)

    def error(self, msg: str) -> None:
        log.debug("yt-dlp error: %s", msg)  # re-raised as DownloadError and reported once


class _ProgressHook:
    def __init__(self) -> None:
        self._last = 0.0

    def __call__(self, d: dict[str, Any]) -> None:
        if d.get("status") == "downloading":
            now = time.monotonic()
            if now - self._last < 2:
                return
            self._last = now
            done = d.get("downloaded_bytes") or 0
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                log.info("Downloading audio: %3.0f%% of %.1f MB", 100 * done / total, total / 1e6)
            else:
                log.info("Downloading audio: %.1f MB", done / 1e6)
        elif d.get("status") == "finished":
            log.info("Downloaded %s", Path(d.get("filename", "")).name)


def _map_download_error(exc: Exception, url: str) -> ToolError:
    msg = str(exc).removeprefix("ERROR: ").strip()
    lowered = msg.lower()
    if "unsupported url" in lowered:
        return InvalidInputError(msg, hint="Pass the URL of a single video.")
    if any(
        s in lowered
        for s in (
            "video unavailable",
            "private video",
            "has been removed",
            "does not exist",
            "not available",
        )
    ):
        return InvalidInputError(msg, hint="Check that the video exists and is public.")
    if "confirm you" in lowered and "bot" in lowered:
        return ToolError(
            msg,
            hint="YouTube is blocking anonymous requests from this IP. Retry later, use "
            "another network, or pass cookies with --cookies-from-browser firefox.",
        )
    return ToolError(
        msg,
        hint="YouTube changes often: update yt-dlp with `uv tool upgrade uvt-transcribe`. "
        "If the stable release is not enough, reinstall with the nightly build: add "
        '--with "yt-dlp[default]>=2026.10.10.dev0" to uv tool install.',
    )


def _source(info: dict[str, Any], url: str) -> SourceInfo:
    return SourceInfo(
        kind="youtube"
        if str(info.get("extractor_key", "")).lower().startswith("youtube")
        else "url",
        url=info.get("webpage_url") or url,
        id=info.get("id"),
        title=info.get("title"),
        channel=info.get("channel") or info.get("uploader"),
        uploader=info.get("uploader"),
        upload_date=info.get("upload_date"),
        duration=info.get("duration"),
    )


def pick_manual_subtitle(
    info: dict[str, Any], preferred: list[str] | None, language: str | None
) -> tuple[str, dict[str, Any]] | None:
    """Choose a manually created subtitle track (code, format entry) or None."""
    tracks = {
        code: [f for f in entries if f.get("ext") in _SUB_FORMATS and f.get("url")]
        for code, entries in (info.get("subtitles") or {}).items()
        if code != "live_chat"
    }
    tracks = {code: entries for code, entries in tracks.items() if entries}
    if not tracks:
        return None
    wanted = preferred or [x for x in (language, info.get("language"), "en") if x]
    for want in (w.lower() for w in wanted):
        for code, entries in tracks.items():
            c = code.lower()
            if c == want or c.startswith(want + "-"):
                return code, _best_sub_format(entries)
    if preferred:
        return None
    code = next(iter(tracks))
    return code, _best_sub_format(tracks[code])


def _best_sub_format(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return min(entries, key=lambda f: _SUB_FORMATS.index(f["ext"]))


def yt_transcribe(
    params: YtTranscribeParams,
    *,
    on_segment: SegmentCallback | None = None,
    backend: TranscriptionBackend | None = None,
) -> TranscriptionResult:
    """Transcribe an online video (YouTube or any site supported by yt-dlp)."""
    opts, cfg = resolve_options(params)
    url = opts.url.strip()
    if urlparse(url).scheme not in {"http", "https"}:
        raise InvalidInputError(f"Not an http(s) URL: {url!r}", hint="Pass a full video URL.")
    try:
        import yt_dlp
        from yt_dlp.utils import DownloadError
    except ImportError as exc:  # pragma: no cover - base dependency
        raise MissingDependencyError(f"yt-dlp is not importable: {exc}") from exc

    js_runtimes = find_js_runtime()
    if js_runtimes is None and is_youtube(url):
        raise MissingDependencyError(
            "No JavaScript runtime found: yt-dlp needs Deno (or Node >= 22) for YouTube.",
            hint="Reinstall the tool (the deno wheel is a dependency on supported platforms) "
            "or install Deno and make sure `deno` is on PATH.",
            os_hints=DENO_INSTALL,
        )

    out_dir = output_dir(opts)
    with tempfile.TemporaryDirectory(prefix="uvt-yt-", ignore_cleanup_errors=True) as tmp:
        ydl_opts: dict[str, Any] = {
            "quiet": True,
            "noprogress": True,
            "logger": _YdlLogger(),
            "progress_hooks": [_ProgressHook()],
            "noplaylist": True,
            "format": "bestaudio/best",
            "fixup": "never",
            "paths": {"home": tmp, "temp": tmp},
            "outtmpl": {"default": "%(id)s.%(ext)s"},
            "retries": 3,
        }
        if js_runtimes:
            ydl_opts["js_runtimes"] = js_runtimes
        if opts.cookies_from_browser:
            ydl_opts["cookiesfrombrowser"] = (opts.cookies_from_browser,)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            log.info("Fetching video info…")
            try:
                info = ydl.extract_info(url, download=False)
            except DownloadError as exc:
                raise _map_download_error(exc, url) from exc
            if info is None or info.get("_type") == "playlist":
                raise InvalidInputError(
                    "Playlists are not supported.", hint="Pass the URL of a single video."
                )
            source = _source(info, url)
            basename = safe_filename(opts.name or f"{source.title or 'video'} [{source.id}]")
            log.info("Video: %s (%s, %s s)", source.title, source.channel, source.duration)

            if opts.prefer_subs:
                result = _from_subtitles(ydl, info, opts, source)
                if result is not None:
                    result.artifacts = write_outputs(result, formats(opts), out_dir, basename)
                    return result

            try:
                info = ydl.process_ie_result(info, download=True)
            except DownloadError as exc:
                raise _map_download_error(exc, url) from exc
            downloads = info.get("requested_downloads") or []
            audio_path = Path(downloads[0]["filepath"]) if downloads else None
            if audio_path is None or not audio_path.is_file():
                raise ToolError("yt-dlp did not produce an audio file.", hint="Re-run with -v.")

        transcript, engine = run_backend(
            opts, cfg, audio_path, backend=backend, on_segment=on_segment
        )
        result = build_result(transcript, engine, source)
        result.artifacts = write_outputs(result, formats(opts), out_dir, basename)
        if opts.keep_audio:
            out_dir.mkdir(parents=True, exist_ok=True)
            kept = out_dir / f"{basename}{audio_path.suffix}"
            shutil.move(str(audio_path), kept)
            result.artifacts.append(Artifact.from_path("audio", kept))
        return result


def _from_subtitles(
    ydl: Any, info: dict[str, Any], opts: YtTranscribeParams, source: SourceInfo
) -> TranscriptionResult | None:
    choice = pick_manual_subtitle(info, opts.sub_langs, opts.language)
    if choice is None:
        log.info("No manual subtitles in the requested languages; transcribing the audio.")
        return None
    code, fmt = choice
    log.info("Using manual subtitles (%s, %s) instead of transcribing.", code, fmt["ext"])
    try:
        with ydl.urlopen(fmt["url"]) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - network errors of many kinds
        log.warning("Could not download subtitles (%s); transcribing instead.", exc)
        return None
    segments = parse_vtt(text)
    if not segments:
        log.warning("Subtitles were empty or unreadable; transcribing instead.")
        return None
    transcript = Transcript(segments=segments, language=code, duration=source.duration)
    return build_result(transcript, None, source, from_subtitles=True)
