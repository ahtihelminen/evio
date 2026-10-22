"""Unplug a USB device in software, for the hardware tier's recovery tests.

`unplugged(...)` writes `0` to the device's sysfs `authorized` attribute,
which unbinds its drivers as a pulled cable would, and writes `1` on exit,
also when the test body raised. Writing there needs permission, which a
plain user lacks; `docs/hardware-tests.md` has the udev rule that grants it.
Without it, or off Linux, the test skips saying so: nothing here asks a
human to touch a cable.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

SYSFS_USB = Path("/sys/bus/usb/devices")
# The IDS-built IMX636 cameras on the ops units; `lsusb` shows other vendors.
IMX636_VENDOR, IMX636_PRODUCT = "1409", "8e00"


def within(timeout_s: float, condition: Callable[[], bool]) -> bool:
    """Poll `condition` until it holds (True) or `timeout_s` passes (False)."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.2)
    return True


def _attribute(device: Path, name: str) -> str | None:
    path = device / name
    return path.read_text().strip() if path.is_file() else None


def find_usb_devices(
    sysfs: Path,
    *,
    vendor: str | None = None,
    product: str | None = None,
    serial: str | None = None,
    port: str | None = None,
) -> list[Path]:
    """Every device directory under `sysfs` matching all the ids given."""
    wanted = {"idVendor": vendor, "idProduct": product, "serial": serial}
    return [
        device
        for device in sorted(sysfs.iterdir())
        if _attribute(device, "idVendor") is not None
        and port in (None, device.name)
        and all(
            value is None or _attribute(device, name) == value
            for name, value in wanted.items()
        )
    ]


@contextmanager
def unplugged(
    *,
    vendor: str | None = None,
    product: str | None = None,
    serial: str | None = None,
    port: str | None = None,
    sysfs: Path = SYSFS_USB,
) -> Iterator[Path]:
    """Detach the one matching USB device for the body; yield its sysfs path.

    Several identical devices (six cameras of one model) match a vendor and
    product alike, so `port` (the sysfs name, such as `2-1.1`) picks one;
    without it, more than one match skips rather than guess.
    """
    if sys.platform != "linux" and sysfs == SYSFS_USB:
        pytest.skip(f"software unplug needs Linux sysfs, not {sys.platform}")
    found = find_usb_devices(
        sysfs, vendor=vendor, product=product, serial=serial, port=port,
    )
    if not found:
        pytest.skip(
            f"no USB device with vendor={vendor!r} product={product!r} "
            f"serial={serial!r} port={port!r} under {sysfs}"
        )
    if len(found) > 1:
        ports = ", ".join(device.name for device in found)
        pytest.skip(
            f"{len(found)} USB devices match ({ports}); name the one to "
            "unplug by its port path, see docs/hardware-tests.md"
        )
    (device,) = found
    authorized = device / "authorized"
    try:
        authorized.write_text("0")
    except PermissionError:
        pytest.skip(
            f"no write access to {authorized}; install the udev rule in "
            "docs/hardware-tests.md"
        )
    try:
        yield device
    finally:
        authorized.write_text("1")


def imx636_serial_on(port: str | None, timeout_s: float) -> str:
    """Unplug the IMX636 on `port` and return the serial that vanished; both
    the vanishing and the reattach must happen within `timeout_s`."""
    from evio.io.imx636_camera import get_camera_serials

    attached = set(get_camera_serials())
    gone: list[set[str]] = []

    def one_gone() -> bool:
        gone.append(attached - set(get_camera_serials()))
        return len(gone[-1]) == 1

    with unplugged(vendor=IMX636_VENDOR, product=IMX636_PRODUCT, port=port):
        assert within(timeout_s, one_gone)
    assert within(timeout_s, lambda: set(get_camera_serials()) == attached)
    (serial,) = gone[-1]
    return serial
