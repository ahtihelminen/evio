"""A software unplug makes one camera vanish from enumeration, and the
reattach brings it back. Skips without a camera or the access in
`docs/hardware-tests.md`. With several identical cameras attached, name the
one to unplug by its port path in `EVIO_UNPLUG_PORT`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import pytest
from support.usb_unplug import unplugged

pytestmark = pytest.mark.hardware

# The IDS-built IMX636 cameras on the ops units; `lsusb` shows other vendors.
IMX636_VENDOR = "1409"
IMX636_PRODUCT = "8e00"
PORT_ENV = "EVIO_UNPLUG_PORT"


def _within(timeout_s: float, condition: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.2)
    return True


@pytest.mark.usefixtures("imx636_camera")
def test_unplug_hides_a_camera_and_reattach_restores_it() -> None:
    from evio.io.imx636_camera import get_camera_serials

    attached = set(get_camera_serials())
    with unplugged(
        vendor=IMX636_VENDOR,
        product=IMX636_PRODUCT,
        port=os.environ.get(PORT_ENV),
    ):
        assert _within(
            10, lambda: len(attached - set(get_camera_serials())) == 1,
        )
    assert _within(10, lambda: set(get_camera_serials()) == attached)
