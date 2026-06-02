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
