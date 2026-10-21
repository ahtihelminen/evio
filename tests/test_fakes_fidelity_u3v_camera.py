"""Fidelity for the U3V camera double in `tests/fakes/u3v_camera.py`."""

from __future__ import annotations

import pytest
from fakes.u3v_camera import FakeU3vCamera, fake_u3v_module

pytestmark = pytest.mark.t1


def test_a_dead_camera_fails_stop_after_recording_it() -> None:
    camera = FakeU3vCamera(stop_error=RuntimeError("Gencp: bulk IN failed"))

    with pytest.raises(RuntimeError, match="bulk IN"):
        camera.stop()
    camera.close()

    assert camera.calls == ["stop", "close"]


def test_the_module_opens_the_given_camera_by_any_serial() -> None:
    camera = FakeU3vCamera()
    module = fake_u3v_module(camera)

    assert module.Camera.open("") is camera
    assert module.Camera.open("4110083899") is camera
