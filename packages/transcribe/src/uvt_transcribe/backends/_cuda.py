"""Make pip-installed NVIDIA libraries visible to CTranslate2.

CTranslate2 (>= 4.6.3) needs cuBLAS for CUDA 12 at runtime, not cuDNN. It
loads the library lazily by name, so if the optional ``[cuda]`` extra
installed ``nvidia-cublas-cu12`` into the tool's venv we preload it here.
After that no ``LD_LIBRARY_PATH`` / ``PATH`` tweaking is needed. Without the
extra, a system CUDA 12 toolkit (cuBLAS on the library path) works too.
"""

from __future__ import annotations

import ctypes
import os
import sys
from functools import lru_cache
from pathlib import Path

from uvt_core.logging import get_logger

log = get_logger("transcribe.cuda")

_LINUX_LIBS = ("libcublasLt.so.12", "libcublas.so.12")
_WINDOWS_LIBS = ("cublasLt64_12.dll", "cublas64_12.dll")


@lru_cache(maxsize=1)
def preload_cuda_libraries() -> tuple[str, ...]:
    """Load cuBLAS from the ``nvidia-cublas-cu12`` wheel if installed. Idempotent."""
    try:
        import nvidia.cublas as cublas  # type: ignore[import-not-found]
    except ImportError:
        return ()
    loaded: list[str] = []
    for base in map(Path, getattr(cublas, "__path__", [])):
        if sys.platform == "win32":
            lib_dir, names = base / "bin", _WINDOWS_LIBS
            if lib_dir.is_dir():
                os.add_dll_directory(str(lib_dir))
                os.environ["PATH"] = f"{lib_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        else:
            lib_dir, names = base / "lib", _LINUX_LIBS
        for name in names:
            path = lib_dir / name
            if not path.is_file():
                continue
            try:
                mode = getattr(ctypes, "RTLD_GLOBAL", 0)
                ctypes.CDLL(str(path), mode=mode)
                loaded.append(str(path))
            except OSError as exc:
                log.warning("Could not preload %s: %s", path, exc)
    if loaded:
        log.debug("Preloaded CUDA libraries: %s", ", ".join(loaded))
    return tuple(loaded)


CUDA_HINT = (
    "CTranslate2 needs NVIDIA cuBLAS for CUDA 12. Either install the 'cuda' extra "
    "(bundles cuBLAS, no system install needed):\n"
    '  uv tool install --reinstall "uvt-transcribe[cuda] @ '
    'git+https://github.com/LorenzoCiarpa/agent-tools#subdirectory=packages/transcribe"\n'
    "or install the CUDA 12 toolkit, or run on CPU with --device cpu."
)
