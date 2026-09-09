"""Unit tests for the count-comparison-and-warn decision, in isolation.

`describe_serial_read` takes plain data (the raw per-device serial strings
from `imx636.IMX636.list_cameras()`, empty string meaning unreadable) and
decides whether to warn. It lives in its own dependency-free module
specifically so it is testable without importing `imx636` or
`evio.io.imx636_camera` — no `sys.modules`/`sys.path` manipulation, no
stubbing, anywhere in this file. See SN-468: two prior attempts at this
fix failed review because their tests stubbed the `imx636` module import
and leaked global state between tests.
"""

from __future__ import annotations

from evio.io.camera_serial_report import describe_serial_read


def test_unreadable_serial_warns_with_both_counts_and_device_id() -> None:
    warning = describe_serial_read(["cam-111", "", "cam-333"])

    assert warning is not None
    assert "3" in warning  # matched device count
    assert "2" in warning  # readable-serial count
    assert "1" in warning  # index of the affected (empty-serial) device


def test_all_readable_serials_produce_no_warning() -> None:
    assert describe_serial_read(["cam-111", "cam-222", "cam-333"]) is None
    assert describe_serial_read([]) is None
