"""A recorder SIGKILLed mid-recording from a real camera leaves footage that
recovery re-indexes whole, and the camera opens again afterwards.

The child reports how many events or frames it had handed its recorder, so
the recovered footage is checked against that: every one of them is there,
in timestamp order. A recording that crosses a UTC hour leaves a finalised
file before the orphan; the counts run over both.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from support.crash import crash_recorder, survivors, written

from evio.codec.dt8 import Dt8Codec, find_dat_data_offset
from evio.recording import FootageRecorder, MonoFootageRecorder
from evio.recording.manifest import ManifestEntry, iter_manifest
from evio.recording.mono_footage_recorder import (
    MONO_GEOM,
    MONO_MAGIC,
    MONO_TS,
    mono_frame_stride,
)
from evio.recording.recovery import recover_mono_orphans, recover_orphans

if TYPE_CHECKING:
    from evio.io.imx636_camera import Imx636CameraSource
    from evio.io.mono_backends.u3v import MonoCameraSource

pytestmark = pytest.mark.hardware

RECORD_S = 3.0
U3V_FPS = 10.0
_DT8_RECORD_BYTES = 8
# As in test_imx636_recording: a hole this big is events the recorder lost.
_MAX_EVENT_GAP_US = 50_000


def _entries(directory: Path) -> dict[str, ManifestEntry]:
    return {
        entry.filename: entry
        for manifest in directory.glob("*/manifest.jsonl")
        for entry in iter_manifest(manifest)
    }


def _crash_and_recover(
    root: Path, pattern: str, recover: Callable[[Path], list[Path]],
    *source: str,
) -> tuple[list[Path], dict[Path, bytes]]:
    """Crash a recorder writing from `source` and recover `root`.

    Returns every footage file in time order and the orphans' bytes as the
    crash left them; checks recovery indexed exactly those orphans.
    """
    root.mkdir()
    directory = crash_recorder(root, *source, record_s=RECORD_S)
    assert not survivors(root)
    footage = sorted(directory.glob(pattern))
    finalised = _entries(directory)
    crashed = {p: p.read_bytes() for p in footage if p.name not in finalised}
    assert crashed

    assert recover(root) == list(crashed)
    entries = _entries(directory)
    for path in crashed:
        data = path.read_bytes()
        entry = entries[path.name]
        assert entry.size_bytes == len(data)
        assert entry.sha256 == hashlib.sha256(data).hexdigest()
        assert entry.capture is not None
        assert entry.capture["recovered"] is True
    return footage, crashed


def _record_once(write: Callable[[], None], close: Callable[[], None]) -> None:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        write()
    close()


def _one_entry(root: Path) -> ManifestEntry:
    (manifest,) = root.glob("*/*/manifest.jsonl")
    (entry,) = iter_manifest(manifest)
    return entry


def test_imx636_recording_recovers_after_sigkill(
    imx636_camera: Imx636CameraSource, tmp_path: Path,
) -> None:
    crashes = tmp_path / "crash"
    footage, crashed = _crash_and_recover(
        crashes, "*/*.dat", recover_orphans, "imx636", imx636_camera.serial,
    )
    count, last_ts = written(crashes)

    with imx636_camera as camera:  # opens only if the killed child released it
        after = FootageRecorder(
            root=tmp_path / "after", camera_id=camera.serial,
            width=camera.width, height=camera.height,
        )
        batches = iter(camera)
        _record_once(lambda: after.write(next(batches)), after.close)
    fresh = _one_entry(tmp_path / "after")
    assert fresh.end_ts_ns > fresh.start_ts_ns

    entries = _entries(footage[0].parents[1])
    timestamps = []
    for path in footage:
        data = path.read_bytes()
        if path in crashed:  # only the torn record and the zero extent are cut
            assert crashed[path].startswith(data)
            tail = crashed[path][len(data) + _DT8_RECORD_BYTES :]
            assert not tail.strip(b"\0")
        codec = Dt8Codec(width=imx636_camera.width, height=imx636_camera.height)
        words = np.fromfile(path, np.uint16, offset=find_dat_data_offset(path))
        decoded = np.concatenate(
            [b.timestamps for b in codec.decode_chunk(words, final=True)],
        )
        entry = entries[path.name]
        assert (entry.start_ts_ns, entry.end_ts_ns) == (
            decoded[0] * 1000, decoded[-1] * 1000,
        )
        timestamps.append(decoded)
    timestamps = np.concatenate(timestamps)
    assert timestamps.size >= count
    assert timestamps[count - 1] == last_ts
    steps = np.diff(timestamps)
    assert np.all(steps >= 0)
    assert steps.max(initial=0) < _MAX_EVENT_GAP_US


def _frame_timestamps(data: bytes) -> list[int]:
    width, height, channels = MONO_GEOM.unpack_from(data, len(MONO_MAGIC))
    stride = mono_frame_stride(width, height, channels)
    frames = data[len(MONO_MAGIC) + MONO_GEOM.size :]
    assert len(frames) % stride == 0
    return [
        MONO_TS.unpack_from(frames, at)[0]
        for at in range(0, len(frames), stride)
    ]


def test_u3v_recording_recovers_after_sigkill(
    u3v_camera: MonoCameraSource, tmp_path: Path,
) -> None:
    crashes = tmp_path / "crash"
    footage, crashed = _crash_and_recover(
        crashes, "*/*.mraw", recover_mono_orphans, "u3v", u3v_camera.serial,
    )
    count, last_ts = written(crashes)

    u3v_camera.fps = U3V_FPS
    with u3v_camera as camera:  # opens only if the killed child released it
        after = MonoFootageRecorder(
            root=tmp_path / "after", camera_id="mono-0",
            width=camera.width, height=camera.height, fps=U3V_FPS,
        )
        _record_once(
            lambda: after.write(camera.read(timeout_ms=2000), time.time_ns()),
            after.close,
        )
    fresh = _one_entry(tmp_path / "after")
    assert fresh.end_ts_ns > fresh.start_ts_ns
    stride = mono_frame_stride(camera.width, camera.height, 1)

    entries = _entries(footage[0].parents[1])
    timestamps = []
    for path in footage:
        data = path.read_bytes()
        if path in crashed:  # only the torn frame is cut
            assert crashed[path].startswith(data)
            assert len(crashed[path]) - len(data) < stride
        frames = _frame_timestamps(data)
        entry = entries[path.name]
        assert (entry.start_ts_ns, entry.end_ts_ns) == (frames[0], frames[-1])
        timestamps += frames
    assert len(timestamps) >= count
    assert timestamps[count - 1] == last_ts
    assert np.all(np.diff(timestamps) > 0)
