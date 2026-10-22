"""The h5py-only reading example in the format docs runs and is correct."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.recording import (
    LabelledRecordingReader,
    LabelledRecordingWriter,
    RecordingMeta,
)

pytestmark = pytest.mark.t1

DOCS = Path(__file__).parent.parent / "docs/modules/labelled-recording.md"


def _run_example(path: Path, start_us: int, end_us: int) -> dict:
    block = re.search(
        r"<!-- docs-example -->\n```python\n(.*?)```",
        DOCS.read_text(),
        re.DOTALL,
    )
    assert block
    namespace = {"path": path, "start_us": start_us, "end_us": end_us}
    exec(block.group(1), namespace)
    return namespace


def _write(path: Path, timestamps: np.ndarray) -> None:
    rng = np.random.default_rng(8)
    n = timestamps.size
    batch = EventsBatch(
        timestamps=timestamps,
        x_coords=rng.integers(0, 346, n).astype(np.int16),
        y_coords=rng.integers(0, 260, n).astype(np.int16),
        polarities=rng.integers(0, 2, n).astype(np.int8),
    )
    meta = RecordingMeta("FRED", "id", "licence", "sensor", 346, 260,
                         "first_event", "event")
    with LabelledRecordingWriter(path, meta, chunk_events=8) as writer:
        writer.write_events(batch)


@pytest.mark.parametrize(
    ("start_us", "end_us", "hits"),
    [(120, 200, True), (-50, 0, False), (10_000, 20_000, False),
     (0, 10_000, True)],
)
def test_docs_example_reads_the_documented_layout(
    tmp_path: Path, start_us: int, end_us: int, hits: bool
) -> None:
    rng = np.random.default_rng(8)
    path = tmp_path / "rec.h5"
    _write(path, np.sort(rng.integers(0, 300, 90)).astype(np.int64))
    namespace = _run_example(path, start_us, end_us)
    with LabelledRecordingReader(path) as reader:
        want = reader.events(start_us, end_us)
    assert (want.timestamps.size > 0) == hits
    np.testing.assert_array_equal(namespace["t"], want.timestamps)
    np.testing.assert_array_equal(namespace["x"], want.x_coords)
    np.testing.assert_array_equal(namespace["y"], want.y_coords)
    np.testing.assert_array_equal(namespace["p"], want.polarities)


def test_docs_example_reads_an_empty_recording(tmp_path: Path) -> None:
    path = tmp_path / "empty.h5"
    _write(path, np.empty(0, np.int64))
    namespace = _run_example(path, 0, 100)
    assert namespace["t"].size == 0
    assert namespace["x"].size == 0
