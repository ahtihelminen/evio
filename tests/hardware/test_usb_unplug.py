"""A software unplug makes the camera vanish from enumeration, and the
reattach brings it back. Skips without the camera or the udev rule in
`docs/hardware-tests.md`; the IMX636 serial is matched against the USB one.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
from support.usb_unplug import unplugged

if TYPE_CHECKING:
    from evio.io.imx636_camera import Imx636CameraSource

pytestmark = pytest.mark.hardware


def _enumerated(serial: str, *, timeout_s: float = 10.0) -> bool:
    from evio.io.imx636_camera import get_camera_serials

    deadline = time.monotonic() + timeout_s
    while serial not in get_camera_serials():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.2)
    return True


def test_unplug_hides_the_camera_and_reattach_restores_it(
    imx636_camera: Imx636CameraSource,
) -> None:
    from evio.io.imx636_camera import get_camera_serials

    serial = imx636_camera.serial
    with unplugged(serial=serial):
        assert serial not in get_camera_serials()
    assert _enumerated(serial)
