"""A `/sys/bus/usb/devices` tree under a test's own directory.

Shaped like the real one: each device a directory named after its port
path holding `idVendor`, `idProduct`, an optional `serial` and a writable
`authorized`, beside interface directories (`1-2:1.0`) that carry none of
those. Writing `authorized` only records the value, which is what a test of
the unplug helper checks; the kernel's unbind is not modelled.
"""

from __future__ import annotations

from pathlib import Path


def add_usb_device(
    root: Path, port: str, vendor: str, product: str, serial: str | None = None,
) -> Path:
    device = root / port
    device.mkdir(parents=True)
    (root / f"{port}:1.0").mkdir()
    (device / "idVendor").write_text(f"{vendor}\n")
    (device / "idProduct").write_text(f"{product}\n")
    if serial is not None:
        (device / "serial").write_text(f"{serial}\n")
    (device / "authorized").write_text("1\n")
    return device
