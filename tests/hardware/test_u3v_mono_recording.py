"""Records a few seconds from a real U3V mono camera through
`MonoFootageRecorder` and checks the footage it leaves under `tmp_path`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from evio.recording.manifest import iter_manifest
from evio.recording.mono_footage_recorder import (
    MONO_GEOM,
    MONO_MAGIC,
    MONO_TS,
    MonoFootageRecorder,
    mono_frame_stride,
)

if TYPE_CHECKING:
    from evio.io.mono_backends.u3v import MonoCameraSource

pytestmark = pytest.mark.hardware

FPS = 10.0
RECORD_S = 3.0


def _frame_timestamps(path: Path, stride: int) -> list[int]:
    data = path.read_bytes()[len(MONO_MAGIC) + MONO_GEOM.size :]
    return [
        MONO_TS.unpack_from(data, at)[0]
        for at in range(0, len(data), stride)
    ]


def test_u3v_camera_records_real_frames_to_footage(
    u3v_camera: MonoCameraSource, tmp_path: Path,
) -> None:
    u3v_camera.fps = FPS
    camera_ts: list[int] = []
    with u3v_camera as camera:
        recorder = MonoFootageRecorder(
            root=tmp_path, camera_id="mono-0",
            width=camera.width, height=camera.height, fps=FPS,
        )
        started = time.monotonic()
        while (elapsed := time.monotonic() - started) < RECORD_S:
            frame, ts = camera.read(timeout_ms=2000, with_ts=True)
            assert frame.shape[:2] == (camera.height, camera.width)
            camera_ts.append(ts)
            recorder.write(frame, time.time_ns())
        recorder.close()

    [manifest] = tmp_path.glob("mono-0/*/manifest.jsonl")
    [entry] = iter_manifest(manifest)
    footage = manifest.parent / entry.filename
    assert entry.size_bytes == footage.stat().st_size
    assert entry.capture == {
        "format": "mono-raw", "width": camera.width, "height": camera.height,
        "channels": 1, "fps": FPS,
    }
    with footage.open("rb") as f:
        assert f.read(len(MONO_MAGIC)) == MONO_MAGIC
        assert MONO_GEOM.unpack(f.read(MONO_GEOM.size)) == (
            camera.width, camera.height, 1,
        )
    stride = mono_frame_stride(camera.width, camera.height, 1)
    timestamps = _frame_timestamps(footage, stride)
    assert np.all(np.diff(timestamps) > 0)
    assert np.all(np.diff(camera_ts) > 0)
    assert entry.start_ts_ns == timestamps[0]
    assert entry.end_ts_ns == timestamps[-1]
    expected = FPS * elapsed
    assert 0.5 * expected <= len(timestamps) <= 1.5 * expected + 1

    with u3v_camera as reopened:  # opens only if the first session released it
        assert reopened.read(timeout_ms=2000).shape[:2] == (
            camera.height, camera.width,
        )
