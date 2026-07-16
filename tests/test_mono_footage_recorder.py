from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pytest

from evio.recording import MonoFootageRecorder, iter_manifest
from evio.recording.mono_footage_recorder import (
    MONO_GEOM,
    MONO_MAGIC,
    MONO_TS,
    mono_frame_stride,
)


def _epoch(year: int, month: int, day: int, hour: int) -> float:
    return float(calendar.timegm((year, month, day, hour, 0, 0, 0, 0, 0)))


def _frame(width: int, height: int, value: int) -> np.ndarray:
    return np.full((height, width), value, dtype=np.uint8)


def test_write_creates_files_and_manifest_on_rotation(tmp_path: Path) -> None:
    clock = {"t": _epoch(2026, 6, 5, 10) + 30 * 60}  # 10:30 UTC
    recorder = MonoFootageRecorder(
        root=tmp_path, camera_id="mono-0",
        width=4, height=3, fps=30.0,
        time_provider=lambda: clock["t"],
    )
    recorder.write(_frame(4, 3, 1), ts_ns=1_000)
    recorder.write(_frame(4, 3, 2), ts_ns=2_000)
    first_path = recorder.current_path
    assert first_path is not None
    assert first_path.exists()

    clock["t"] += 3600  # +1 hour -> new bucket
    recorder.write(_frame(4, 3, 3), ts_ns=3_000)
    assert recorder.current_path != first_path

    recorder.close()
    manifest = tmp_path / "mono-0" / "2026-06-05" / "manifest.jsonl"
    entries = list(iter_manifest(manifest))
    assert len(entries) == 2
    assert entries[0].filename.startswith("100000Z-")
    assert entries[0].filename.endswith(".mraw")
    assert entries[1].filename.startswith("110000Z-")
    assert entries[0].start_ts_ns == 1_000
    assert entries[0].end_ts_ns == 2_000
    assert entries[0].capture == {
        "format": "mono-raw", "width": 4, "height": 3,
        "channels": 1, "fps": 30.0,
    }


def test_reenable_same_hour_writes_new_subindexed_file(
    tmp_path: Path,
) -> None:
    # Same defect as the event recorder (field 2026-07-15): a same-hour
    # re-enable must sub-index, never overwrite the earlier block.
    clock = {"t": _epoch(2026, 7, 15, 12) + 60}
    recorder = MonoFootageRecorder(
        root=tmp_path, camera_id="mono-0", width=4, height=3,
        time_provider=lambda: clock["t"],
    )
    recorder.write(_frame(4, 3, 1), ts_ns=1_000)
    first = recorder.current_path
    assert first is not None
    recorder.close()
    first_size = first.stat().st_size
    clock["t"] += 600.0
    recorder.write(_frame(4, 3, 2), ts_ns=2_000)
    second = recorder.current_path
    recorder.close()
    assert second is not None
    assert second.name == first.name.replace(".mraw", ".1.mraw")
    assert first.stat().st_size == first_size


def test_on_disk_layout_is_header_then_fixed_stride(tmp_path: Path) -> None:
    recorder = MonoFootageRecorder(
        root=tmp_path, camera_id="mono-0", width=4, height=3,
        time_provider=lambda: _epoch(2026, 6, 5, 12),
    )
    recorder.write(_frame(4, 3, 5), ts_ns=42)
    path = recorder.current_path
    assert path is not None
    recorder.close()

    raw = path.read_bytes()
    header = MONO_MAGIC + MONO_GEOM.pack(4, 3, 1)
    assert raw.startswith(header)
    assert len(raw) == len(header) + mono_frame_stride(4, 3, 1)
    ts = MONO_TS.unpack(raw[len(header):len(header) + MONO_TS.size])[0]
    assert ts == 42
    pixels = np.frombuffer(raw[len(header) + MONO_TS.size:], dtype=np.uint8)
    assert pixels.tolist() == [5] * 12


def test_write_rejects_wrong_frame_size(tmp_path: Path) -> None:
    recorder = MonoFootageRecorder(
        root=tmp_path, camera_id="mono-0", width=4, height=3,
        time_provider=lambda: _epoch(2026, 6, 5, 12),
    )
    with pytest.raises(ValueError, match="expected 12"):
        recorder.write(_frame(2, 2, 0), ts_ns=1)


def test_no_frames_writes_nothing(tmp_path: Path) -> None:
    recorder = MonoFootageRecorder(
        root=tmp_path, camera_id="mono-0", width=4, height=3,
        time_provider=lambda: _epoch(2026, 6, 5, 12),
    )
    recorder.close()
    assert not (tmp_path / "mono-0").exists()
