from __future__ import annotations

import calendar
import json
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


def test_bias_and_context_land_in_provenance_and_capture(
    tmp_path: Path,
) -> None:
    recorder = FootageRecorder(
        root=tmp_path, camera_id="event-0",
        width=1280, height=720, window_us=20_000, step_us=10_000,
        time_provider=lambda: _epoch(2026, 5, 21, 14),
    )
    recorder.write(_batch(start_us=1_000_000, n=4))
    recorder.note_bias("diff", 5, ts_ns=1_500_000_000)
    recorder.note_context(
        {
            "regime": "overcast",
            "target_noise_eps": 1e5,
            "biases": {"diff": 5, "refr": 20},
        },
        ts_ns=1_600_000_000,
    )
    recorder.close()

    prov = list((tmp_path / "event-0").glob("provenance-*.jsonl"))
    assert len(prov) == 1
    lines = [json.loads(x) for x in prov[0].read_text().splitlines()]
    assert [x["event"] for x in lines] == ["bias", "context"]
    assert lines[0] == {
        "event": "bias", "ts_ns": 1_500_000_000, "name": "diff", "value": 5,
    }
    assert lines[1]["regime"] == "overcast"

    entry = next(iter_manifest(
        tmp_path / "event-0" / "2026-05-21" / "manifest.jsonl",
    ))
    assert entry.capture is not None
    assert entry.capture["window_us"] == 20_000
    assert entry.capture["step_us"] == 10_000
    assert entry.capture["regime"] == "overcast"
    # note_bias + note_context both fold into the snapshot vector.
    assert entry.capture["biases"] == {"diff": 5, "refr": 20}


def test_recording_without_notes_has_capture_but_no_provenance(
    tmp_path: Path,
) -> None:
    recorder = FootageRecorder(
        root=tmp_path, camera_id="event-0",
        width=1280, height=720, window_us=20_000,
        time_provider=lambda: _epoch(2026, 5, 21, 15),
    )
    recorder.write(_batch(start_us=1, n=3))
    recorder.close()
    assert not list((tmp_path / "event-0").glob("provenance-*.jsonl"))
    entry = next(iter_manifest(
        tmp_path / "event-0" / "2026-05-21" / "manifest.jsonl",
    ))
    assert entry.capture is not None
    assert entry.capture["biases"] == {}
    assert entry.capture["window_us"] == 20_000


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


def test_degraded_disk_buffers_then_recovers(
    tmp_path: Path, monkeypatch: object,
) -> None:
    """A transient allocation failure NEVER stops recording (2026-07-03).

    While the disk path is broken, batches wait in the bounded pending
    buffer with ``degraded=True``; once allocation works again, every
    buffered batch lands, in order, with nothing lost.
    """
    import evio.recording.footage_recorder as fr

    real = fr._preallocate
    fail = {"on": True}

    def flaky(fd: int, offset: int, length: int) -> None:
        if fail["on"]:
            raise OSError(28, "No space left on device")
        real(fd, offset, length)

    monkeypatch.setattr(fr, "_preallocate", flaky)
    clock = {"t": _epoch(2026, 5, 21, 10)}
    recorder = FootageRecorder(
        root=tmp_path, camera_id="event-0", width=1280, height=720,
        time_provider=lambda: clock["t"],
    )
    recorder.write(_batch(start_us=1_000, n=5))  # alloc fails -> buffered
    assert recorder.degraded
    assert recorder.current_path is None or recorder.degraded is not False
    clock["t"] += 30.0  # past the retry backoff
    fail["on"] = False
    recorder.write(_batch(start_us=2_000, n=5))  # recovers + drains both
    assert not recorder.degraded
    assert recorder.dropped_bytes == 0
    recorder.close()
    dats = list(tmp_path.rglob("*.dat"))
    assert len(dats) == 1
    entries = list(iter_manifest(dats[0].parent / "manifest.jsonl"))
    assert entries[0].start_ts_ns == 1_000 * 1000  # first batch preserved
    assert entries[0].size_bytes == dats[0].stat().st_size


def test_reenable_same_hour_writes_new_subindexed_file(
    tmp_path: Path,
) -> None:
    """Field 2026-07-15: toggling recording off and back on inside one hour
    reused the same filename and truncated the earlier block (~150 GB lost
    fleet-wide). Every enable span must land in its own sub-indexed file."""
    clock = {"t": _epoch(2026, 7, 15, 12) + 60}
    recorder = FootageRecorder(
        root=tmp_path, camera_id="event-0", width=1280, height=720,
        time_provider=lambda: clock["t"],
    )
    recorder.write(_batch(start_us=1_000_000, n=500))
    first = recorder.current_path
    assert first is not None
    recorder.close()  # recording toggled OFF
    first_size = first.stat().st_size
    clock["t"] += 600.0  # toggled back ON inside the same hour
    recorder.write(_batch(start_us=2_000_000, n=5))
    second = recorder.current_path
    recorder.close()
    clock["t"] += 600.0  # and once more
    recorder.write(_batch(start_us=3_000_000, n=5))
    third = recorder.current_path
    recorder.close()
    assert second is not None and third is not None
    assert second.name == first.name.replace(".dat", ".1.dat")
    assert third.name == first.name.replace(".dat", ".2.dat")
    assert first.stat().st_size == first_size  # earlier block untouched
    entries = list(iter_manifest(first.parent / "manifest.jsonl"))
    assert [e.filename for e in entries] == [
        first.name, second.name, third.name,
    ]
    # Each row stamps its enable-span open wall time — the anchor replay
    # prefers over the (series-start) filename.
    base = _epoch(2026, 7, 15, 12)
    assert [e.wall_start_ns for e in entries] == [
        int((base + 60) * 1e9),
        int((base + 660) * 1e9),
        int((base + 1260) * 1e9),
    ]


def test_recovery_trims_preallocated_zero_tail(tmp_path: Path) -> None:
    """A crash mid-extent leaves a zero tail; recovery must trim it."""
    from evio.recording.footage_recorder import DAT_FILE_HEADER
    from evio.recording.recovery import recover_orphans

    day = tmp_path / "event-0" / "2026-05-21"
    day.mkdir(parents=True)
    path = day / "100000Z-093000Z.dat"
    record = (12345).to_bytes(4, "little") + (0x10000064).to_bytes(4, "little")
    path.write_bytes(DAT_FILE_HEADER + record * 3 + b"\x00" * (8 * 1000))
    recovered = recover_orphans(tmp_path)
    assert path in recovered
    assert path.stat().st_size == len(DAT_FILE_HEADER) + 8 * 3
