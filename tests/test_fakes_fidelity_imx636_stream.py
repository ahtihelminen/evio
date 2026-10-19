"""Fidelity for the streaming camera double in `tests/fakes/imx636_stream.py`.

`FakeStreamingCamera` stands in for the driver's `IMX636` in the stream
tests of `Imx636CameraSource`. These tests pin what it models; the
hardware tier checks `CD_DTYPE` against the real `imx636.CD_DTYPE`.
"""

from __future__ import annotations

import numpy as np
import pytest
from fakes.imx636_stream import CD_DTYPE, FakeStreamingCamera, cd_buffer

pytestmark = pytest.mark.t1


def test_poll_fires_the_callback_once_per_buffer_then_reports_nothing() -> None:
    received: list[np.ndarray] = []
    first, second = cd_buffer(np.arange(3)), cd_buffer(np.arange(3, 5))
    camera = FakeStreamingCamera([first, second])
    camera.set_cd_callback(received.append)

    assert [camera.poll(), camera.poll(), camera.poll()] == [1, 1, 0]
    assert received == [first, second]


def test_poll_without_a_callback_delivers_nothing() -> None:
    camera = FakeStreamingCamera([cd_buffer(np.arange(3))])

    assert camera.poll() == 0


def test_cd_dtype_is_the_layout_the_camera_source_reads() -> None:
    """`_on_cd_events` documents it: x int16, y int16, p int8, three pad
    bytes, t int64, so `t` sits 8 bytes in and a record is 16 bytes."""
    assert CD_DTYPE.itemsize == 16
    assert CD_DTYPE.fields["t"][1] == 8
    assert {name: CD_DTYPE[name] for name in ("x", "y", "p", "t")} == {
        "x": np.dtype("<i2"), "y": np.dtype("<i2"),
        "p": np.dtype("i1"), "t": np.dtype("<i8"),
    }


def test_the_script_can_restart_the_clock() -> None:
    stamps = np.array([3_786_227, 16, 24])

    assert cd_buffer(stamps)["t"].tolist() == stamps.tolist()
