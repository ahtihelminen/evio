"""Placeholders exercising each camera fixture until the real hardware
tests land; on a laptop both skip, naming the missing driver or device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from evio.io.imx636_camera import Imx636CameraSource
    from evio.io.mono_backends.u3v import MonoCameraSource

pytestmark = pytest.mark.hardware


def test_imx636_camera_fixture_names_an_attached_camera(
    imx636_camera: Imx636CameraSource,
) -> None:
    assert imx636_camera.serial


def test_u3v_camera_fixture_names_an_attached_camera(
    u3v_camera: MonoCameraSource,
) -> None:
    assert u3v_camera.serial
