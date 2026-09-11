from __future__ import annotations

import builtins
import sys

import pytest


def test_mono_camera_facade_importable_without_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`evio.io.mono_camera` must import cleanly with no `ids-peak` package
    installed: the SDK import lives inside the backend module and is only
    touched when backend code actually runs.
    """
    for name in list(sys.modules):
        if name == "ids_peak" or name.startswith("ids_peak."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, "evio.io.mono_camera", raising=False)
    monkeypatch.delitem(
        sys.modules, "evio.io.mono_backends.ids_peak", raising=False
    )

    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "ids_peak" or name.startswith("ids_peak."):
            msg = f"No module named {name!r}"
            raise ModuleNotFoundError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    import evio.io.mono_camera  # noqa: F401
