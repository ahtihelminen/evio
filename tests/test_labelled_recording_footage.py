"""R5.4: an OPS capture plus rig ground truth becomes a labelled recording
with evio alone: FootageRecorder -> DT8 decode -> LabelledRecordingWriter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from evio.codec.dt8 import Dt8Codec, find_dat_data_offset
from evio.core.events import EventsBatch
from evio.recording import (
    FootageRecorder,
    LabelledRecordingReader,
    LabelledRecordingWriter,
    RecordingLabels,
    RecordingMeta,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.t1

WIDTH, HEIGHT = 640, 480


def _capture(root: Path, events: EventsBatch) -> Path:
    recorder = FootageRecorder(
        root=root, camera_id="cam0", width=WIDTH, height=HEIGHT,
        time_provider=lambda: 0.0,
    )
    recorder.write(events)
    path = recorder.current_path
    recorder.close()
    assert path is not None
    return path


def _decode(path: Path) -> EventsBatch:
    words = np.fromfile(path, dtype=np.uint16)
    words = words[find_dat_data_offset(path) // 2:]
    (batch,) = Dt8Codec(width=WIDTH, height=HEIGHT).decode_chunk(
        words, final=True,
    )
    return batch


def test_footage_with_rig_labels_becomes_a_labelled_recording(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(4)
    n = 1000
    captured = EventsBatch(
        timestamps=np.sort(rng.integers(0, 2_000_000, n)).astype(np.int64),
        x_coords=rng.integers(0, WIDTH, n).astype(np.int16),
        y_coords=rng.integers(0, HEIGHT, n).astype(np.int16),
        polarities=rng.integers(0, 2, n).astype(np.int8),
    )
    decoded = _decode(_capture(tmp_path / "footage", captured))
    # Rig rows as the OPS loop sees them: (time_us, track id, class, box).
    rig_tracks = {"01J9ZK0A": 0, "01J9ZK0B": 1}  # ULID -> int id, by caller
    rig_rows = [
        (100_000, "01J9ZK0A", 3, (10, 20, 50, 60)),
        (600_000, "01J9ZK0B", 3, (300, 200, 340, 230)),
        (900_000, "01J9ZK0A", 3, (12, 22, 52, 62)),
    ]
    labels = RecordingLabels(
        time_us=np.array([r[0] for r in rig_rows]),
        object_id=np.array([rig_tracks[r[1]] for r in rig_rows]),
        class_id=np.array([r[2] for r in rig_rows]),
        box=np.array([r[3] for r in rig_rows], dtype=np.float64),
    )
    meta = RecordingMeta(
        source_dataset="ops-capture", original_id="cam0", licence="internal",
        sensor="IMX636", width=WIDTH, height=HEIGHT,
        time_origin="sensor_power_on", box_frame="event",
    )
    path = tmp_path / "ops.h5"
    with LabelledRecordingWriter(path, meta, chunk_events=64) as writer:
        for start in range(0, decoded.timestamps.size, 300):
            window = slice(start, start + 300)
            writer.write_events(EventsBatch(
                decoded.timestamps[window], decoded.x_coords[window],
                decoded.y_coords[window], decoded.polarities[window],
            ))
        writer.write_labels(labels)

    with LabelledRecordingReader(path) as reader:
        everything = reader.events(0, 2_000_001)
        read_labels = reader.labels()
        assert reader.meta == meta
    np.testing.assert_array_equal(everything.timestamps, captured.timestamps)
    np.testing.assert_array_equal(everything.x_coords, captured.x_coords)
    np.testing.assert_array_equal(everything.y_coords, captured.y_coords)
    np.testing.assert_array_equal(everything.polarities, captured.polarities)
    np.testing.assert_array_equal(read_labels.box, labels.box)
    np.testing.assert_array_equal(read_labels.object_id, [0, 1, 0])
