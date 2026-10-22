"""The h5py-only reading example in the format docs runs and is correct."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.recording import (
    ExcludedSpans,
    LabelledRecordingReader,
    LabelledRecordingWriter,
    RecordingMeta,
)

pytestmark = pytest.mark.t1

DOCS = Path(__file__).parent.parent / "docs/modules/labelled-recording.md"


def _run_example(path: Path, start_us: int, end_us: int) -> dict:
    return _exec_block("docs-example", path=path, start_us=start_us,
                       end_us=end_us)


def _exec_block(marker: str, **names: object) -> dict:
    block = re.search(
        rf"<!-- {marker} -->\n```python\n(.*?)```",
        DOCS.read_text(),
        re.DOTALL,
    )
    assert block
    namespace = dict(names)
    exec(block.group(1), namespace)
    return namespace


def _write(
    path: Path, timestamps: np.ndarray, *, with_1_1_groups: bool = False,
) -> None:
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
        if with_1_1_groups:
            writer.write_bias([(0, "bias_diff", 1), (100, "bias_diff", 2),
                               (100, "bias_hpf", 3)])
            writer.write_excluded(ExcludedSpans(
                np.array([10], np.int64), np.array([20], np.int64)))


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


def test_docs_example_ignores_the_1_1_groups_of_a_newer_file(
    tmp_path: Path,
) -> None:
    times = np.sort(np.random.default_rng(8).integers(0, 300, 90))
    plain, newer = tmp_path / "plain.h5", tmp_path / "newer.h5"
    _write(plain, times.astype(np.int64))
    _write(newer, times.astype(np.int64), with_1_1_groups=True)
    want = _run_example(plain, 120, 200)
    got = _run_example(newer, 120, 200)
    assert got["t"].size > 0
    np.testing.assert_array_equal(got["t"], want["t"])
    np.testing.assert_array_equal(got["x"], want["x"])


@pytest.mark.parametrize(
    ("time_us", "want"),
    [(-1, {}), (0, {"bias_diff": 1}), (99, {"bias_diff": 1}),
     (100, {"bias_diff": 2, "bias_hpf": 3})],
)
def test_docs_bias_example_matches_the_reader(
    tmp_path: Path, time_us: int, want: dict[str, int],
) -> None:
    path = tmp_path / "rec.h5"
    _write(path, np.arange(0, 200, 5, dtype=np.int64), with_1_1_groups=True)
    namespace = _exec_block("docs-example-bias", path=path, time_us=time_us)
    assert namespace["bias"] == want
    with LabelledRecordingReader(path) as reader:
        assert reader.bias_at(time_us) == want
