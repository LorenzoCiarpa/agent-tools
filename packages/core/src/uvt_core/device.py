"""Compute device detection without heavy dependencies.

Never imports torch or other ML frameworks: CUDA is probed through
``ctranslate2`` when it is installed (it is what faster-whisper runs on), then
through ``torch`` only if it is *already imported*, then through
``nvidia-smi``. Backends refine the result (e.g. faster-whisper maps ``mps``
to ``cpu`` because CTranslate2 has no Metal support).
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field

DeviceKind = Literal["cuda", "mps", "cpu"]
DeviceRequest = Literal["auto", "cuda", "mps", "cpu"]
DEVICE_ENV = "UVT_DEVICE"


class DeviceInfo(BaseModel):
    """The device a job actually ran on (reported in ``JobResult.device``)."""

    kind: DeviceKind
    name: str | None = None
    index: int | None = None
    compute_type: str | None = Field(
        default=None, description="Numeric precision used, e.g. float16, int8."
    )
    details: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() in {"arm64", "aarch64"}


def _nvidia_smi_names() -> list[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


@lru_cache(maxsize=1)
def cuda_device_count() -> int:
    """Number of usable CUDA devices (0 if none or undetectable)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) in {"", "-1"}:
        return 0
    if importlib.util.find_spec("ctranslate2") is not None:
        try:
            import ctranslate2

            return int(ctranslate2.get_cuda_device_count())
        except Exception:  # noqa: BLE001 - broken CUDA install => treat as no GPU
            return 0
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            return int(torch.cuda.device_count())
        except Exception:  # noqa: BLE001
            return 0
    return len(_nvidia_smi_names())


@lru_cache(maxsize=1)
def cuda_device_names() -> tuple[str, ...]:
    return tuple(_nvidia_smi_names())


def detect_device(requested: DeviceRequest | str | None = "auto") -> DeviceKind:
    """Resolve ``auto`` (or ``UVT_DEVICE``) to a concrete device kind.

    Explicit requests are returned unchanged: callers decide whether an
    unavailable explicit device is an error (it should be).
    """
    req = (requested or "auto").lower()
    if req == "auto":
        req = os.environ.get(DEVICE_ENV, "auto").lower()
    if req in {"cuda", "mps", "cpu"}:
        return req  # type: ignore[return-value]
    if cuda_device_count() > 0:
        return "cuda"
    if is_apple_silicon():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def cpu_name() -> str:
    """Marketing name of the CPU when the OS exposes it cheaply."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:  # noqa: PTH123
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine() or "CPU"


def describe(kind: DeviceKind, index: int = 0, compute_type: str | None = None) -> DeviceInfo:
    """Build a ``DeviceInfo`` with a human-readable name."""
    name: str | None
    if kind == "cuda":
        names = cuda_device_names()
        name = names[index] if index < len(names) else "CUDA device"
    elif kind == "mps":
        name = "Apple Silicon GPU (Metal)"
    else:
        name = cpu_name()
    return DeviceInfo(
        kind=kind,
        name=name,
        index=index if kind == "cuda" else None,
        compute_type=compute_type,
    )
