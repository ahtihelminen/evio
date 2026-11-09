"""Fidelity for the sysfs tree double in `tests/fakes/sysfs_usb.py`.

The real reader is `support.usb_unplug`: it lists a sysfs directory, reads
`idVendor`, `idProduct` and `serial` to match a device, and writes
`authorized` to unplug it. The tree is also compared with a sample of what
the kernel writes under `/sys/bus/usb/devices` for an IDS IMX636 on a hub.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes.sysfs_usb import add_usb_device
from support.usb_unplug import find_usb_devices, unplugged

pytestmark = pytest.mark.t1

# Captured from a Linux host (`ls` and `cat` under /sys/bus/usb/devices):
# the device directory is named after its port path, an interface directory
# `<port>:<config>.<interface>` sits beside it, and each id attribute is one
# lowercase value and a newline.
REAL_SAMPLE_DEVICE = "2-1.1"
REAL_SAMPLE_INTERFACE = "2-1.1:1.0"
REAL_SAMPLE_ATTRIBUTES = {
    "idVendor": "1409\n",
    "idProduct": "8e00\n",
    "serial": "4110083899\n",
    "authorized": "1\n",
}
# What the reader consumes; a rename in the double must break the match.
READ_BY_THE_READER = {"idVendor", "idProduct", "serial"}
WRITTEN_BY_THE_READER = {"authorized"}


@pytest.fixture
def device(tmp_path: Path) -> Path:
    return add_usb_device(
        tmp_path, REAL_SAMPLE_DEVICE, "1409", "8e00", serial="4110083899",
    )


def test_the_tree_has_the_real_layout(device: Path) -> None:
    root = device.parent

    assert sorted(path.name for path in root.iterdir()) == [
        REAL_SAMPLE_DEVICE, REAL_SAMPLE_INTERFACE,
    ]
    assert (root / REAL_SAMPLE_INTERFACE).is_dir()
    assert {
        path.name: path.read_text() for path in device.iterdir()
    } == REAL_SAMPLE_ATTRIBUTES


def test_an_interface_directory_carries_no_device_attribute(
    device: Path,
) -> None:
    interface = device.parent / REAL_SAMPLE_INTERFACE

    assert list(interface.iterdir()) == []


def test_the_double_provides_every_attribute_the_reader_consumes(
    device: Path,
) -> None:
    names = {path.name for path in device.iterdir()}

    assert names >= READ_BY_THE_READER | WRITTEN_BY_THE_READER


def test_the_reader_finds_the_device_by_each_attribute_it_reads(
    device: Path,
) -> None:
    root = device.parent

    assert find_usb_devices(root, vendor="1409") == [device]
    assert find_usb_devices(root, product="8e00") == [device]
    assert find_usb_devices(root, serial="4110083899") == [device]
    assert find_usb_devices(root, port=REAL_SAMPLE_DEVICE) == [device]


def test_the_reader_does_not_take_an_interface_for_a_device(
    device: Path,
) -> None:
    assert find_usb_devices(device.parent) == [device]


def test_a_device_without_a_serial_has_no_serial_file(tmp_path: Path) -> None:
    hub = add_usb_device(tmp_path, "usb1", "1d6b", "0002")

    assert not (hub / "serial").exists()
    assert find_usb_devices(tmp_path, serial="4110083899") == []
    assert find_usb_devices(tmp_path, vendor="1d6b") == [hub]


def test_the_reader_unplugs_through_the_authorized_attribute(
    device: Path,
) -> None:
    with unplugged(serial="4110083899", sysfs=device.parent):
        assert (device / "authorized").read_text() == "0"
    assert (device / "authorized").read_text() == "1"
