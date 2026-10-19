"""A real IMX636 records through FootageRecorder into a valid file."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from evio.codec.dt8 import Dt8Codec, find_dat_data_offset
from evio.recording import FootageRecorder
from evio.recording.manifest import iter_manifest

if TYPE_CHECKING:
    from evio.io.imx636_camera import Imx636CameraSource

pytestmark = pytest.mark.hardware

_RECORD_S = 3.0
# A live camera buffers about 120 ms, so a consumer stall longer than that
# leaves a hole in the event stream. The rigs' cameras emit continuously (hot
# pixels alone give hundreds of thousands of events a second), so a hole this
# big is lost events; a truly silent camera would trip it too.
_MAX_EVENT_GAP_US = 50_000


def test_imx636_records_real_events_to_footage(
    imx636_camera: Imx636CameraSource, tmp_path: Path,
) -> None:
    with imx636_camera as camera:
        recorder = FootageRecorder(
            root=tmp_path, camera_id=camera.serial,
            width=camera.width, height=camera.height,
        )
        deadline = time.monotonic() + _RECORD_S
        for batch in camera:
            recorder.write(batch)
            if time.monotonic() >= deadline:
                break
        recorder.close()
    assert imx636_camera.cam is None
    with imx636_camera:  # reopening proves the USB claim was released
        pass

    (manifest,) = tmp_path.glob("*/*/manifest.jsonl")
    (entry,) = iter_manifest(manifest)
    dat = manifest.parent / entry.filename
    assert entry.size_bytes == dat.stat().st_size
    assert entry.sha256 == hashlib.sha256(dat.read_bytes()).hexdigest()
    assert entry.capture is not None
    assert (entry.capture["width"], entry.capture["height"]) == (
        imx636_camera.width, imx636_camera.height,
    )

    words = np.fromfile(dat, np.uint16, offset=find_dat_data_offset(dat))
    codec = Dt8Codec(width=imx636_camera.width, height=imx636_camera.height)
    timestamps = np.concatenate(
        [b.timestamps for b in codec.decode_chunk(words, final=True)],
    )
    assert timestamps.size > 0
    steps = np.diff(timestamps)
    assert np.all(steps >= 0)
    assert steps.max(initial=0) < _MAX_EVENT_GAP_US
    assert (entry.start_ts_ns, entry.end_ts_ns) == (
        timestamps[0] * 1000, timestamps[-1] * 1000,
    )
