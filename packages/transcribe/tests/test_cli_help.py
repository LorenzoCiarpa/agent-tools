"""Every console script starts and prints --help without heavy imports or hardware."""

import os
import subprocess
import sys

import pytest

MODULES = {
    "transcribe-file": "uvt_transcribe.cli.transcribe_file",
    "yt-transcribe": "uvt_transcribe.cli.yt_transcribe",
    "live-transcribe": "uvt_transcribe.cli.live_transcribe",
}


@pytest.mark.parametrize("command", sorted(MODULES))
def test_help(command):
    code = (
        "import sys\n"
        f"from {MODULES[command]} import main\n"
        "sys.argv = ['x', '--help']\n"
        "try:\n    main()\nexcept SystemExit as e:\n    code = e.code\n"
        "heavy = [m for m in ('faster_whisper', 'ctranslate2', 'yt_dlp', 'sounddevice', 'torch')"
        " if m in sys.modules]\n"
        "assert not heavy, heavy\n"
        "sys.exit(code)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "COLUMNS": "120"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "Examples" in proc.stdout and "Exit codes" in proc.stdout
