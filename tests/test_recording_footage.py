from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np

from evio.core.events import EventsBatch
from evio.recording import FootageRecorder, iter_manifest


def _epoch(year: int, month: int, day: int, hour: int) -> float:
    return float(calendar.timegm((year, month, day, hour, 0, 0, 0, 0, 0)))


def _batch(start_us: int, n: int) -> EventsBatch:
    return EventsBatch(
        timestamps=np.arange(start_us, start_us + n, dtype=np.int64),
        x_coords=np.full(n, 100, dtype=np.int16),
        y_coords=np.full(n, 200, dtype=np.int16),
        polarities=np.zeros(n, dtype=np.int8),
    )


def test_write_creates_file_and_manifest_on_rotation(tmp_path: Path) -> None:
    clock = {"t": _epoch(2026, 5, 21, 10) + 30 * 60}  # 10:30 UTC
    recorder = FootageRecorder(
        root=tmp_path, camera_id="event-0",
        width=1280, height=720,
        time_provider=lambda: clock["t"],
    )
    recorder.write(_batch(start_us=1_000_000, n=5))
    first_path = recorder.current_path
    assert first_path is not None
    assert first_path.exists()

    clock["t"] += 3600  # +1 hour
    recorder.write(_batch(start_us=2_000_000, n=5))
    second_path = recorder.current_path
    assert second_path != first_path

    recorder.close()
    manifest = tmp_path / "event-0" / "2026-05-21" / "manifest.jsonl"
    entries = list(iter_manifest(manifest))
    assert len(entries) == 2
    # Filename format: <hour>0000Z-<run_id>.dat — run_id is HHMMSSZ taken
    # from the recorder's construction time, so the prefix encodes the
    # bucket but each restart writes a fresh file.
    assert entries[0].filename.startswith("100000Z-")
    assert entries[0].filename.endswith(".dat")
    assert entries[1].filename.startswith("110000Z-")
    assert entries[1].filename.endswith(".dat")
    assert entries[0].end_ts_ns > entries[0].start_ts_ns


def test_object_id_tag_lands_in_manifest(tmp_path: Path) -> None:
    recorder = FootageRecorder(
        root=tmp_path, camera_id="event-0",
        width=1280, height=720,
        time_provider=lambda: _epoch(2026, 5, 21, 12),
    )
    recorder.write(_batch(start_us=1, n=3))
    recorder.mark_object_id("track-abc")
    recorder.mark_object_id("track-def")
    recorder.close()
    entries = list(iter_manifest(
        tmp_path / "event-0" / "2026-05-21" / "manifest.jsonl",
    ))
    assert entries[0].object_ids == ["track-abc", "track-def"]


def test_empty_batch_is_a_noop(tmp_path: Path) -> None:
    recorder = FootageRecorder(
        root=tmp_path, camera_id="event-0",
        width=1280, height=720,
        time_provider=lambda: _epoch(2026, 5, 21, 9),
    )
    empty = EventsBatch(
        timestamps=np.zeros(0, dtype=np.int64),
        x_coords=np.zeros(0, dtype=np.int16),
        y_coords=np.zeros(0, dtype=np.int16),
        polarities=np.zeros(0, dtype=np.int8),
    )
    recorder.write(empty)
    recorder.close()
    assert not (tmp_path / "event-0").exists()
