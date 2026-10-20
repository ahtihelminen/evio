from __future__ import annotations

import importlib
import sys

import pytest
from fakes.u3v_camera import FakeU3vCamera, fake_u3v_module

pytestmark = pytest.mark.t1


def test_u3v_source_opens_a_camera_a_killed_recorder_left_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = FakeU3vCamera()
    monkeypatch.setitem(sys.modules, "u3v", fake_u3v_module(camera))
    monkeypatch.delitem(sys.modules, "evio.io.mono_backends.u3v", raising=False)
    backend = importlib.import_module("evio.io.mono_backends.u3v")

    with backend.MonoCameraSource():
        pass

    assert camera.calls[:2] == ["stop", "start"]
