"""A camera unplugged mid-recording: the source gives up within
`UNPLUG_TIMEOUT_S`, the recorder keeps every frame or event it was handed,
and after the reattach the camera opens and records again.

The unplug is `support.usb_unplug.unplugged`, so these skip without a camera
or without the access in `docs/hardware-tests.md`. The IMX636 cameras on a
unit are identical; name the one to unplug by its port in `EVIO_UNPLUG_PORT`.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from support.usb_unplug import (
    IMX636_PRODUCT,
    IMX636_VENDOR,
    imx636_serial_on,
    unplugged,
    within,
)

from evio.codec.dt8 import Dt8Codec, find_dat_data_offset
from evio.recording import FootageRecorder
from evio.recording.manifest import iter_manifest
from evio.recording.mono_footage_recorder import (
    MONO_GEOM,
    MONO_MAGIC,
    MONO_TS,
    MonoFootageRecorder,
    mono_frame_stride,
)

if TYPE_CHECKING:
    from evio.io.imx636_camera import Imx636CameraSource
    from evio.io.mono_backends.u3v import MonoCameraSource

pytestmark = pytest.mark.hardware

PORT_ENV = "EVIO_UNPLUG_PORT"
RECORD_S = 2.0
OPEN_TIMEOUT_S = 30.0  # an IMX636 takes about 3 s to open
UNPLUG_TIMEOUT_S = 5.0  # the most a source may take to notice the unplug
REATTACH_TIMEOUT_S = 15.0
U3V_FPS = 10.0
U3V_READ_TIMEOUT_MS = 1000


def _for(seconds: float) -> Callable[[], bool]:
    """A `stopped()` that turns true `seconds` after it is first asked, so
    the camera's open time does not eat into the recording."""
    deadline: list[float] = []

    def stopped() -> bool:
        if not deadline:
            deadline.append(time.monotonic() + seconds)
        return time.monotonic() >= deadline[0]

    return stopped


def _unplug_mid_recording(
    record: Callable[[Callable[[], bool]], None],
    written: Sequence[object],
    unplug: AbstractContextManager[Path],
) -> None:
    """Run `record` until the unplug ends it, which must be cleanly or with
    a device error that says what happened, within `UNPLUG_TIMEOUT_S`. The
    unplug comes `RECORD_S` after `record` first appends to `written`.

    The recording runs on a daemon thread so that a source which hangs fails
    the test instead of hanging the run, and the device is reattached either
    way. `record` gets a `stopped()` check that turns true when the test
    leaves, so a skipped unplug does not leave the camera recording.
    """
    stop = threading.Event()
    ended: list[RuntimeError | OSError | None] = []

    def run() -> None:
        try:
            record(stop.is_set)
        except (RuntimeError, OSError) as error:
            ended.append(error)
        else:
            ended.append(None)

    thread = threading.Thread(target=run, name="unplug-recording", daemon=True)
    thread.start()
    try:
        assert within(OPEN_TIMEOUT_S, lambda: bool(written) or bool(ended)), (
            "nothing recorded"
        )
        time.sleep(RECORD_S)
        assert thread.is_alive(), f"recording ended before the unplug: {ended}"
        with unplug as device:
            thread.join(UNPLUG_TIMEOUT_S)
            assert not thread.is_alive(), (
                f"source still running {UNPLUG_TIMEOUT_S}s after the unplug"
            )
    finally:
        stop.set()
        thread.join(UNPLUG_TIMEOUT_S)
    assert ended, "the source raised something other than a device error"
    if (error := ended[0]) is not None:
        assert str(error), f"{type(error).__name__} without a message"
    assert within(
        REATTACH_TIMEOUT_S, lambda: any(device.glob(f"{device.name}:*")),
    ), "device interfaces did not come back after the reattach"


def _entries(root: Path) -> list[tuple[Path, int, int]]:
    """(file, start_ts_ns, end_ts_ns) of every finalised file, in order."""
    found = []
    for manifest in sorted(root.glob("*/*/manifest.jsonl")):
        for entry in iter_manifest(manifest):
            path = manifest.parent / entry.filename
            assert entry.size_bytes == path.stat().st_size
            found.append((path, entry.start_ts_ns, entry.end_ts_ns))
    assert found, f"no finalised file under {root}"
    return found


def _imx636_record(
    source: Imx636CameraSource, root: Path, written: list[np.ndarray],
    stopped: Callable[[], bool],
) -> None:
    with source as camera:
        recorder = FootageRecorder(
            root=root, camera_id=camera.serial,
            width=camera.width, height=camera.height,
        )
        try:
            for batch in camera:
                recorder.write(batch)
                written.append(batch.timestamps)
                if stopped():
                    break
        finally:
            recorder.close()


def _assert_imx636_footage(
    root: Path, written: list[np.ndarray], width: int, height: int,
) -> None:
    decoded = []
    for dat, start_ns, end_ns in _entries(root):
        words = np.fromfile(dat, np.uint16, offset=find_dat_data_offset(dat))
        codec = Dt8Codec(width=width, height=height)
        timestamps = np.concatenate(
            [b.timestamps for b in codec.decode_chunk(words, final=True)],
        )
        assert (start_ns, end_ns) == (
            timestamps[0] * 1000, timestamps[-1] * 1000,
        )
        decoded.append(timestamps)
    expected = np.concatenate(written)
    assert expected.size > 0
    np.testing.assert_array_equal(np.concatenate(decoded), expected)


@pytest.mark.usefixtures("imx636_camera")
def test_imx636_recording_survives_an_unplug(tmp_path: Path) -> None:
    from evio.io.imx636_camera import Imx636CameraSource, get_camera_serials

    port = os.environ.get(PORT_ENV)
    serial = imx636_serial_on(port, REATTACH_TIMEOUT_S)
    source = Imx636CameraSource(serial=serial)
    before: list[np.ndarray] = []
    _unplug_mid_recording(
        lambda stopped: _imx636_record(
            source, tmp_path / "before", before, stopped,
        ),
        before,
        unplugged(vendor=IMX636_VENDOR, product=IMX636_PRODUCT, port=port),
    )
    assert source.cam is None
    _assert_imx636_footage(
        tmp_path / "before", before, source.width, source.height,
    )

    assert within(
        REATTACH_TIMEOUT_S, lambda: source.serial in get_camera_serials(),
    )
    after: list[np.ndarray] = []
    reopened = Imx636CameraSource(serial=source.serial)
    _imx636_record(reopened, tmp_path / "after", after, _for(RECORD_S))
    _assert_imx636_footage(
        tmp_path / "after", after, reopened.width, reopened.height,
    )


def _u3v_record(
    source: MonoCameraSource, root: Path, written: list[int],
    stopped: Callable[[], bool],
) -> None:
    with source as camera:
        recorder = MonoFootageRecorder(
            root=root, camera_id="mono-0",
            width=camera.width, height=camera.height, fps=U3V_FPS,
        )
        try:
            while not stopped():
                frame = camera.read(timeout_ms=U3V_READ_TIMEOUT_MS)
                ts_ns = time.time_ns()
                recorder.write(frame, ts_ns)
                written.append(ts_ns)
        finally:
            recorder.close()


def _assert_u3v_footage(
    root: Path, written: list[int], width: int, height: int,
) -> None:
    stride = mono_frame_stride(width, height, 1)
    stored = []
    for path, start_ns, end_ns in _entries(root):
        data = path.read_bytes()
        header = len(MONO_MAGIC) + MONO_GEOM.size
        assert data[:header] == MONO_MAGIC + MONO_GEOM.pack(width, height, 1)
        assert (len(data) - header) % stride == 0
        timestamps = [
            MONO_TS.unpack_from(data, at)[0]
            for at in range(header, len(data), stride)
        ]
        assert (start_ns, end_ns) == (timestamps[0], timestamps[-1])
        stored += timestamps
    assert written
    assert stored == written


def test_u3v_recording_survives_an_unplug(
    u3v_camera: MonoCameraSource, tmp_path: Path,
) -> None:
    u3v_camera.fps = U3V_FPS
    before: list[int] = []
    _unplug_mid_recording(
        lambda stopped: _u3v_record(
            u3v_camera, tmp_path / "before", before, stopped,
        ),
        before,
        unplugged(serial=u3v_camera.serial),
    )
    _assert_u3v_footage(
        tmp_path / "before", before, u3v_camera.width, u3v_camera.height,
    )

    after: list[int] = []
    _u3v_record(u3v_camera, tmp_path / "after", after, _for(RECORD_S))
    _assert_u3v_footage(
        tmp_path / "after", after, u3v_camera.width, u3v_camera.height,
    )
