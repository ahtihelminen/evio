"""The U3V mono backend releases the camera however its stop goes.

`u3v` is the in-house driver extension, absent on dev machines, so
`fakes.u3v_camera` stands in for it.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from fakes.u3v_camera import FakeU3vCamera, fake_u3v_module

pytestmark = pytest.mark.t1


def _source_class(
    monkeypatch: pytest.MonkeyPatch, camera: FakeU3vCamera,
) -> type:
    monkeypatch.setitem(sys.modules, "u3v", fake_u3v_module(camera))
    monkeypatch.delitem(
        sys.modules, "evio.io.mono_backends.u3v", raising=False,
    )
    module = importlib.import_module("evio.io.mono_backends.u3v")
    return module.MonoCameraSource


def test_exit_stops_then_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    camera = FakeU3vCamera()
    with _source_class(monkeypatch, camera)():
        pass
    assert camera.calls == ["start", "stop", "close"]


def test_exit_closes_the_camera_when_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = FakeU3vCamera(
        stop_error=RuntimeError("Gencp: bulk IN failed or short ack"),
    )
    with (
        pytest.raises(RuntimeError, match="bulk IN"),
        _source_class(monkeypatch, camera)(),
    ):
        pass
    assert camera.calls == ["start", "stop", "close"]
