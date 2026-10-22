"""The labelled recording format: lossless, seekable, plain-h5py readable."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.recording import (
    FORMAT_VERSION,
    LabelledRecordingReader,
    LabelledRecordingWriter,
    RecordingLabels,
    RecordingMeta,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.t1

META = RecordingMeta(
    source_dataset="FRED",
    original_id="seq-007",
    licence="CC-BY-4.0",
    sensor="DAVIS346",
    width=346,
    height=260,
    time_origin="first_event",
    box_frame="rgb",
)
SHUFFLE_ID, DEFLATE_ID = 2, 1


def _events(times: list[int] | np.ndarray, seed: int = 0) -> EventsBatch:
    rng = np.random.default_rng(seed)
    n = len(times)
    return EventsBatch(
        timestamps=np.asarray(times, dtype=np.int64),
        x_coords=rng.integers(0, 346, n).astype(np.int16),
        y_coords=rng.integers(0, 260, n).astype(np.int16),
        polarities=rng.integers(0, 2, n).astype(np.int8),
    )


def _labels(n: int, seed: int = 1) -> RecordingLabels:
    rng = np.random.default_rng(seed)
    return RecordingLabels(
        time_us=np.sort(rng.integers(0, 10_000, n)).astype(np.int64),
        object_id=rng.integers(0, 2**40, n).astype(np.int64),
        class_id=rng.integers(0, 5, n).astype(np.int32),
        box=rng.random((n, 4)) * 300 + 1 / 3,
    )


def _slice(batch: EventsBatch, rows: slice) -> EventsBatch:
    return EventsBatch(
        batch.timestamps[rows],
        batch.x_coords[rows],
        batch.y_coords[rows],
        batch.polarities[rows],
    )


def _write(
    path: Path,
    batch: EventsBatch,
    labels: RecordingLabels | None = None,
    *,
    chunk_events: int = 8,
    batch_size: int | None = None,
) -> None:
    step = batch_size or max(len(batch.timestamps), 1)
    with LabelledRecordingWriter(
        path, META, chunk_events=chunk_events,
    ) as writer:
        for start in range(0, max(len(batch.timestamps), 1), step):
            writer.write_events(_slice(batch, slice(start, start + step)))
        if labels is not None:
            writer.write_labels(labels)


def _read_all(path: Path) -> EventsBatch:
    with LabelledRecordingReader(path) as reader:
        return reader.events(-(2**62), 2**62)


def _assert_same(got: EventsBatch, want: EventsBatch) -> None:
    for name in ("timestamps", "x_coords", "y_coords", "polarities"):
        assert getattr(got, name).dtype == getattr(want, name).dtype
        np.testing.assert_array_equal(
            getattr(got, name), getattr(want, name),
        )


def test_round_trip_is_exact_for_events_labels_and_meta(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rec.h5"
    batch = _events(np.sort(np.random.default_rng(3).integers(0, 5000, 53)))
    labels = _labels(7)
    _write(path, batch, labels)
    with LabelledRecordingReader(path) as reader:
        assert reader.meta == META
        assert reader.n_events == 53
        got = reader.labels()
        _assert_same(reader.events(-(2**62), 2**62), batch)
    for name in ("time_us", "object_id", "class_id", "box"):
        assert getattr(got, name).dtype == getattr(labels, name).dtype
        np.testing.assert_array_equal(
            getattr(got, name), getattr(labels, name),
        )


def test_seek_equals_a_naive_filter_on_random_streams(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(5)
    for trial in range(6):
        chunk = int(rng.choice([1, 3, 4, 8]))
        times = np.sort(rng.integers(0, 40, int(rng.integers(0, 70))))
        batch = _events(times, seed=trial)
        path = tmp_path / f"rec{trial}.h5"
        _write(path, batch, chunk_events=chunk)
        with LabelledRecordingReader(path) as reader:
            bounds = [(rng.integers(-5, 50), rng.integers(-5, 50))
                      for _ in range(60)]
            bounds += [(-1, 100), (0, 0), (30, 10)]
            bounds += [(t, t + d) for t in times[::chunk] for d in (0, 1, 2)]
            for start, end in bounds:
                keep = (times >= start) & (times < end)
                _assert_same(
                    reader.events(int(start), int(end)),
                    _slice(batch, keep),
                )


def test_window_is_half_open_with_duplicates_across_a_chunk_boundary(
    tmp_path: Path,
) -> None:
    times = [1, 5, 5, 5, 5, 5, 9, 9]  # chunk 4 starts inside the run of 5s
    batch = _events(times)
    _write(tmp_path / "rec.h5", batch, chunk_events=4)
    with LabelledRecordingReader(tmp_path / "rec.h5") as reader:
        _assert_same(reader.events(5, 6), _slice(batch, slice(1, 6)))
        _assert_same(reader.events(5, 9), _slice(batch, slice(1, 6)))
        _assert_same(reader.events(6, 9), _slice(batch, slice(0, 0)))
        _assert_same(reader.events(9, 10), _slice(batch, slice(6, 8)))


def test_seek_does_not_decode_earlier_chunks(tmp_path: Path) -> None:
    path = tmp_path / "rec.h5"
    batch = _events(np.arange(0, 400, 2))
    _write(path, batch, chunk_events=10)
    with h5py.File(path, "r+") as f:
        f["events/dt_us"][:100] = 999_999  # poison chunks 0..9
    with LabelledRecordingReader(path) as reader:
        _assert_same(reader.events(300, 330), _slice(batch, slice(150, 165)))


def test_streaming_in_odd_batches_equals_one_shot(tmp_path: Path) -> None:
    batch = _events(np.sort(np.random.default_rng(9).integers(0, 900, 205)))
    labels = _labels(11)
    _write(tmp_path / "one.h5", batch, labels, chunk_events=16)
    streamed = tmp_path / "streamed.h5"
    with LabelledRecordingWriter(streamed, META, chunk_events=16) as writer:
        for start, stop in [(0, 0), (0, 1), (1, 40), (40, 41), (41, 205)]:
            writer.write_events(_slice(batch, slice(start, stop)))
        writer.write_labels(_slice_labels(labels, slice(0, 4)))
        writer.write_labels(_slice_labels(labels, slice(4, 11)))
    with h5py.File(tmp_path / "one.h5") as one, h5py.File(streamed) as two:
        names: list[str] = []
        one.visititems(lambda name, obj: names.append(name)
                       if isinstance(obj, h5py.Dataset) else None)
        assert names
        for name in names:
            np.testing.assert_array_equal(one[name][:], two[name][:])


def _slice_labels(labels: RecordingLabels, rows: slice) -> RecordingLabels:
    return RecordingLabels(
        labels.time_us[rows], labels.object_id[rows],
        labels.class_id[rows], labels.box[rows],
    )


def test_writer_buffers_less_than_one_chunk(tmp_path: Path) -> None:
    batch = _events(np.arange(100))
    with LabelledRecordingWriter(
        tmp_path / "rec.h5", META, chunk_events=8,
    ) as writer:
        written = 0
        for size in (3, 20, 1, 40, 0, 36):
            writer.write_events(_slice(batch, slice(written, written + size)))
            written += size
            assert writer.buffered_events == written % 8
    assert written == 100
    _assert_same(_read_all(tmp_path / "rec.h5"), batch)


def test_only_gzip_and_shuffle_filters_and_plain_h5py_reads(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rec.h5"
    _write(path, _events(np.arange(30)), _labels(5))
    datasets: list[h5py.Dataset] = []
    with h5py.File(path) as f:
        f.visititems(lambda _, obj: datasets.append(obj)
                     if isinstance(obj, h5py.Dataset) else None)
        assert {d.name for d in datasets} == {
            "/events/dt_us", "/events/x", "/events/y", "/events/polarity",
            "/index/t_us", "/labels/time_us", "/labels/object_id",
            "/labels/class_id", "/labels/box",
        }
        for dataset in datasets:
            plist = dataset.id.get_create_plist()
            filters = [plist.get_filter(i)[0]
                       for i in range(plist.get_nfilters())]
            assert filters == [SHUFFLE_ID, DEFLATE_ID], dataset.name
            assert dataset[:] is not None
        assert f.attrs["format_version"] == FORMAT_VERSION


def test_non_monotonic_timestamps_raise_and_are_not_written(
    tmp_path: Path,
) -> None:
    with LabelledRecordingWriter(
        tmp_path / "rec.h5", META, chunk_events=4,
    ) as writer:
        with pytest.raises(ValueError, match="non-decreasing"):
            writer.write_events(_events([1, 5, 4]))
        assert writer.buffered_events == 0
        writer.write_events(_events([1, 5, 5]))
        with pytest.raises(ValueError, match="non-decreasing"):
            writer.write_events(_events([4, 6]))
        assert writer.buffered_events == 3


def test_empty_recording_without_labels_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "rec.h5"
    _write(path, _events([]))
    with LabelledRecordingReader(path) as reader:
        assert reader.meta == META
        assert reader.n_events == 0
        assert reader.events(0, 10).timestamps.dtype == np.int64
        assert reader.events(0, 10).timestamps.size == 0
        assert list(reader.iter_events()) == []
        assert reader.labels().box.shape == (0, 4)


def test_recording_without_labels_keeps_its_events(tmp_path: Path) -> None:
    batch = _events([2, 3, 3])
    _write(tmp_path / "rec.h5", batch)
    _assert_same(_read_all(tmp_path / "rec.h5"), batch)
    with LabelledRecordingReader(tmp_path / "rec.h5") as reader:
        assert reader.labels().time_us.size == 0


def test_gaps_beyond_32_bit_and_large_start_times_survive(
    tmp_path: Path,
) -> None:
    start = 2**40 + 7
    times = [start, start, start + 2**33, start + 2**33 + 5, start + 2**36]
    batch = _events(times)
    _write(tmp_path / "rec.h5", batch, chunk_events=2)
    _assert_same(_read_all(tmp_path / "rec.h5"), batch)
    with LabelledRecordingReader(tmp_path / "rec.h5") as reader:
        _assert_same(
            reader.events(start + 2**33, start + 2**34),
            _slice(batch, slice(2, 4)),
        )


def test_iter_events_yields_one_chunk_at_a_time(tmp_path: Path) -> None:
    batch = _events(np.sort(np.random.default_rng(2).integers(0, 99, 21)))
    _write(tmp_path / "rec.h5", batch, chunk_events=8)
    with LabelledRecordingReader(tmp_path / "rec.h5") as reader:
        parts = list(reader.iter_events())
    assert [len(p.timestamps) for p in parts] == [8, 8, 5]
    _assert_same(
        EventsBatch(*(np.concatenate(cols) for cols in zip(
            *((p.timestamps, p.x_coords, p.y_coords, p.polarities)
              for p in parts), strict=True))),
        batch,
    )


@pytest.mark.parametrize(
    ("attr", "value", "readable"),
    [
        ("format_version", "2.0", False),
        ("format_version", "1.9", True),
        ("format", "something-else", False),
    ],
)
def test_reader_refuses_newer_major_and_foreign_files(
    tmp_path: Path, attr: str, value: str, readable: bool,
) -> None:
    path = tmp_path / "rec.h5"
    _write(path, _events([1, 2]))
    with h5py.File(path, "r+") as f:
        f.attrs[attr] = value
    if readable:
        _read_all(path)
    else:
        with pytest.raises(ValueError, match="format version"):
            LabelledRecordingReader(path)

