"""Thin facade selecting the mono-camera backend via `EVIO_MONO_BACKEND`.

Backend modules (e.g. `ids_peak`) import their SDK at module scope, so the
backend is only imported here, lazily, keeping this facade importable with
no SDK installed.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType


def _backend_module() -> ModuleType:
    backend = os.environ.get("EVIO_MONO_BACKEND", "ids_peak")
    return importlib.import_module(f"evio.io.mono_backends.{backend}")


def __getattr__(name: str) -> object:
    if name in ("MonoCameraSource", "get_mono_camera_serials"):
        return getattr(_backend_module(), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
