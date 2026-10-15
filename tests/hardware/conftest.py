"""Camera fixtures for the hardware tier.

Each yields an unopened source for the first enumerated device, so a test
opens and closes the camera itself. On a host without the driver or the
device, the fixture skips and names which one is missing. The drivers are
imported here, not at module level, because they are absent on a laptop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from evio.io.imx636_camera import Imx636CameraSource
    from evio.io.mono_backends.u3v import MonoCameraSource


@pytest.fixture
def imx636_camera() -> Imx636CameraSource:
    try:
        from evio.io import imx636_camera as driver
    except ImportError as error:
        pytest.skip(f"imx636 driver not importable: {error}")
    if not (serials := driver.get_camera_serials()):
        pytest.skip("no IMX636 camera enumerated")
    return driver.Imx636CameraSource(serial=serials[0])


@pytest.fixture
def u3v_camera() -> MonoCameraSource:
    try:
        from evio.io.mono_backends import u3v as driver
    except ImportError as error:
        pytest.skip(f"u3v driver not importable: {error}")
    if not (serials := driver.get_mono_camera_serials()):
        pytest.skip("no U3V mono camera enumerated")
    return driver.MonoCameraSource(serial=serials[0])
