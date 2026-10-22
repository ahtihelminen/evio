"""A software unplug makes one camera vanish from enumeration, and the
reattach brings it back. Skips without a camera or the access in
`docs/hardware-tests.md`. With several identical cameras attached, name the
one to unplug by its port path in `EVIO_UNPLUG_PORT`.
"""

from __future__ import annotations

import os

import pytest
from support.usb_unplug import imx636_serial_on

pytestmark = pytest.mark.hardware

PORT_ENV = "EVIO_UNPLUG_PORT"


@pytest.mark.usefixtures("imx636_camera")
def test_unplug_hides_a_camera_and_reattach_restores_it() -> None:
    imx636_serial_on(os.environ.get(PORT_ENV), timeout_s=10)
