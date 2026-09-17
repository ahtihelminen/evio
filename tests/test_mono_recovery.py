from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pytest

from evio.recording import (
    MonoFootageRecorder,
    iter_manifest,
    recover_mono_orphans,
)
from evio.recording.mono_footage_recorder import (
    MONO_GEOM,
    MONO_MAGIC,
    MONO_TS,
    mono_frame_stride,
)

pytestmark = pytest.mark.t1

_HEADER_LEN = len(MONO_MAGIC) + MONO_GEOM.size


def _epoch(year: int, month: int, day: int, hour: int) -> float:
    return float(calendar.timegm((year, month, day, hour, 0, 0, 0, 0, 0)))


def _mono_bytes(
    width: int, height: int, n_frames: int, *, trailing: int = 0,
) -> bytes:
    """A mono `.mraw` body: header + whole frames + a partial tail.

    Mimics a power cut: real frames followed by `trailing` half-written
    bytes the recovery scan must drop. Frame timestamps ramp 1000, 2000, ...
    """
    body = MONO_MAGIC + MONO_GEOM.pack(width, height, 1)
    payload = width * height
    for i in range(n_frames):
        body += MONO_TS.pack((i + 1) * 1000) + bytes([i % 256]) * payload
    return body + b"\xff" * trailing


def _write(day_dir: Path, name: str, body: bytes) -> Path:
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / name
    path.write_bytes(body)
    return path


def test_recover_truncates_partial_and_reindexes(tmp_path: Path) -> None:
    day_dir = tmp_path / "mono-0" / "2026-06-05"
    orphan = _write(
        day_dir, "100000Z-AAA.mraw", _mono_bytes(4, 3, 5, trailing=3),
    )

    recovered = recover_mono_orphans(tmp_path)

    assert recovered == [orphan]
    stride = mono_frame_stride(4, 3, 1)
    assert orphan.stat().st_size == _HEADER_LEN + 5 * stride
    entries = list(iter_manifest(day_dir / "manifest.jsonl"))
    assert len(entries) == 1
    assert entries[0].filename == "100000Z-AAA.mraw"
    assert entries[0].start_ts_ns == 1_000
    assert entries[0].end_ts_ns == 5_000
    assert entries[0].object_ids == []
    assert entries[0].capture == {
        "format": "mono-raw", "width": 4, "height": 3,
        "channels": 1, "recovered": True,
    }


def test_recover_removes_header_only_file(tmp_path: Path) -> None:
    day_dir = tmp_path / "mono-0" / "2026-06-05"
    orphan = _write(day_dir, "110000Z-BBB.mraw", _mono_bytes(4, 3, 0))

    recovered = recover_mono_orphans(tmp_path)

    assert recovered == []
    assert not orphan.exists()
    assert not (day_dir / "manifest.jsonl").exists()


def test_recover_unlinks_sub_header_stub(tmp_path: Path) -> None:
    # A rotate that died mid-header (ENOSPC, power cut) leaves a stub
    # shorter than the header: it can never be indexed, so recovery must
    # remove it instead of re-warning about it on every boot. A foreign
    # short file (magic-prefix mismatch) is left untouched.
    day_dir = tmp_path / "mono-0" / "2026-07-15"
    stub = _write(day_dir, "120000Z-074303Z.1.mraw", MONO_MAGIC[:10])
    foreign = _write(day_dir, "junk.mraw", b"short junk")

    assert recover_mono_orphans(tmp_path) == []
    assert not stub.exists()
    assert foreign.exists()
    assert not (day_dir / "manifest.jsonl").exists()


def test_recover_skips_files_already_in_manifest(tmp_path: Path) -> None:
    recorder = MonoFootageRecorder(
        root=tmp_path, camera_id="mono-0", width=4, height=3,
        time_provider=lambda: _epoch(2026, 6, 5, 12),
    )
    recorder.write(np.full((3, 4), 1, dtype=np.uint8), ts_ns=1_000)
    recorder.close()
    manifest = tmp_path / "mono-0" / "2026-06-05" / "manifest.jsonl"
    assert len(list(iter_manifest(manifest))) == 1

    assert recover_mono_orphans(tmp_path) == []
    assert len(list(iter_manifest(manifest))) == 1


def test_recover_ignores_foreign_file(tmp_path: Path) -> None:
    day_dir = tmp_path / "mono-0" / "2026-06-05"
    foreign = _write(day_dir, "junk.mraw", b"not an evio mono header at all!!")

    assert recover_mono_orphans(tmp_path) == []
    assert foreign.exists()  # left untouched, not indexed
    assert not (day_dir / "manifest.jsonl").exists()


def test_recover_missing_root_is_noop(tmp_path: Path) -> None:
    assert recover_mono_orphans(tmp_path / "does-not-exist") == []
