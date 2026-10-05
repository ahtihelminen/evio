from __future__ import annotations

import builtins
import importlib
import sys
from types import SimpleNamespace

import pytest

from evio.io import mono_camera

pytestmark = pytest.mark.t1


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


def _intercept_imports(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the facade's backend imports and answer each with a stand-in."""
    imported: list[str] = []

    def record(name: str) -> SimpleNamespace:
        imported.append(name)
        return SimpleNamespace(MonoCameraSource=name)

    monkeypatch.setattr(importlib, "import_module", record)
    return imported


def test_mono_facade_imports_backend_named_by_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = _intercept_imports(monkeypatch)
    monkeypatch.setenv("EVIO_MONO_BACKEND", "u3v")

    source = mono_camera.MonoCameraSource

    assert imported == ["evio.io.mono_backends.u3v"]
    assert source == "evio.io.mono_backends.u3v"


def test_mono_facade_defaults_to_ids_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = _intercept_imports(monkeypatch)
    monkeypatch.delenv("EVIO_MONO_BACKEND", raising=False)

    source = mono_camera.MonoCameraSource

    assert imported == ["evio.io.mono_backends.ids_peak"]
    assert source == "evio.io.mono_backends.ids_peak"


def test_mono_facade_unknown_backend_fails_naming_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVIO_MONO_BACKEND", "no_such_backend")

    with pytest.raises(ModuleNotFoundError, match="no_such_backend"):
        _ = mono_camera.MonoCameraSource


def test_mono_facade_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="NoSuchThing"):
        _ = mono_camera.NoSuchThing
