from __future__ import annotations

import calendar
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.core.recording import open_dat_dt8
from evio.recording import FootageRecorder, iter_manifest, recover_orphans
from evio.recording.footage_recorder import DAT_FILE_HEADER

pytestmark = pytest.mark.t1


def _epoch(year: int, month: int, day: int, hour: int) -> float:
    return float(calendar.timegm((year, month, day, hour, 0, 0, 0, 0, 0)))


def _orphan_bytes(n_records: int, *, trailing: int = 0) -> bytes:
    """A footage `.dat` body: header + whole records + a partial tail.

    Mimics a power cut: real records followed by `trailing` half-written
    bytes the recovery scan must drop. Record t32 ramps 1000, 2000, ... µs.
    """
    body = DAT_FILE_HEADER
    for i in range(n_records):
        body += struct.pack("<II", (i + 1) * 1000, i)
    return body + b"\xff" * trailing


def _write_orphan(day_dir: Path, name: str, body: bytes) -> Path:
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / name
    path.write_bytes(body)
    return path


def test_recover_truncates_partial_and_reindexes(tmp_path: Path) -> None:
    day_dir = tmp_path / "event-0" / "2026-06-05"
    orphan = _write_orphan(
        day_dir, "100000Z-AAA.dat", _orphan_bytes(5, trailing=3),
    )

    recovered = recover_orphans(tmp_path)

    assert recovered == [orphan]
    # Trailing partial record dropped: header + 5 whole 8-byte records.
    assert orphan.stat().st_size == len(DAT_FILE_HEADER) + 5 * 8
    entries = list(iter_manifest(day_dir / "manifest.jsonl"))
    assert len(entries) == 1
    assert entries[0].filename == "100000Z-AAA.dat"
    assert entries[0].start_ts_ns == 1_000_000
    assert entries[0].end_ts_ns == 5_000_000
    assert entries[0].object_ids == []
    assert entries[0].capture == {"recovered": True}
    # The truncated file is still a valid DT8 recording.
    rec = open_dat_dt8(str(orphan), width=1280, height=720)
    assert rec.timestamps.size == 5


def test_recover_handles_sub_indexed_names(tmp_path: Path) -> None:
    # A re-enabled recording block (same hour, same run) carries a
    # sub-index; orphan recovery must re-index it like any other .dat.
    day_dir = tmp_path / "event-0" / "2026-07-15"
    orphan = _write_orphan(
        day_dir, "120000Z-074303Z.1.dat", _orphan_bytes(3),
    )
    assert recover_orphans(tmp_path) == [orphan]
    entries = list(iter_manifest(day_dir / "manifest.jsonl"))
    assert entries[0].filename == "120000Z-074303Z.1.dat"
    assert entries[0].capture == {"recovered": True}


def test_recover_removes_header_only_file(tmp_path: Path) -> None:
    day_dir = tmp_path / "event-0" / "2026-06-05"
    orphan = _write_orphan(day_dir, "110000Z-BBB.dat", _orphan_bytes(0))

    recovered = recover_orphans(tmp_path)

    assert recovered == []
    assert not orphan.exists()
    assert not (day_dir / "manifest.jsonl").exists()


def test_recover_skips_files_already_in_manifest(tmp_path: Path) -> None:
    recorder = FootageRecorder(
        root=tmp_path, camera_id="event-0", width=1280, height=720,
        time_provider=lambda: _epoch(2026, 6, 5, 12),
    )
    recorder.write(EventsBatch(
        timestamps=np.arange(1_000_000, 1_000_004, dtype=np.int64),
        x_coords=np.full(4, 100, dtype=np.int16),
        y_coords=np.full(4, 200, dtype=np.int16),
        polarities=np.zeros(4, dtype=np.int8),
    ))
    recorder.close()
    manifest = tmp_path / "event-0" / "2026-06-05" / "manifest.jsonl"
    before = list(iter_manifest(manifest))
    assert len(before) == 1

    assert recover_orphans(tmp_path) == []
    assert len(list(iter_manifest(manifest))) == 1


def test_recover_ignores_foreign_file(tmp_path: Path) -> None:
    day_dir = tmp_path / "event-0" / "2026-06-05"
    foreign = _write_orphan(day_dir, "junk.dat", b"not an evio header at all")

    assert recover_orphans(tmp_path) == []
    assert foreign.exists()  # left untouched, not indexed
    assert not (day_dir / "manifest.jsonl").exists()


def test_recover_missing_root_is_noop(tmp_path: Path) -> None:
    assert recover_orphans(tmp_path / "does-not-exist") == []


def test_recover_before_cutoff_skips_live_recorder_files(
    tmp_path: Path,
) -> None:
    """`before` excludes files a concurrent recorder still owns (mtime >=)."""
    day_dir = tmp_path / "event-0" / "2026-06-05"
    orphan = _write_orphan(day_dir, "120000Z-CCC.dat", _orphan_bytes(5))
    cutoff = 1_000_000.0
    os.utime(orphan, (cutoff, cutoff))  # mtime exactly at the cutoff

    # mtime >= before → treated as a live-recorder file, left untouched.
    assert recover_orphans(tmp_path, before=cutoff) == []
    assert not (day_dir / "manifest.jsonl").exists()

    # A later boot with no live writer (no cutoff) re-indexes it.
    assert recover_orphans(tmp_path) == [orphan]
    assert len(list(iter_manifest(day_dir / "manifest.jsonl"))) == 1


def test_periodic_fsync_flushes_to_disk_before_close(tmp_path: Path) -> None:
    clock = {"t": _epoch(2026, 6, 5, 8)}
    recorder = FootageRecorder(
        root=tmp_path, camera_id="event-0", width=1280, height=720,
        fsync_interval_s=1.0, time_provider=lambda: clock["t"],
    )

    def _batch(start_us: int) -> EventsBatch:
        return EventsBatch(
            timestamps=np.arange(start_us, start_us + 4, dtype=np.int64),
            x_coords=np.full(4, 100, dtype=np.int16),
            y_coords=np.full(4, 200, dtype=np.int16),
            polarities=np.zeros(4, dtype=np.int8),
        )

    recorder.write(_batch(1_000_000))
    path = recorder.current_path
    assert path is not None
    clock["t"] += 2.0  # cross the fsync interval
    recorder.write(_batch(2_000_000))
    # fsync flushed the Python buffer, so bytes are on disk pre-close.
    assert path.stat().st_size > len(DAT_FILE_HEADER)
    recorder.close()
