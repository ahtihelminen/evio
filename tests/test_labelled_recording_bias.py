"""Format 1.1: the bias log and the excluded spans of a labelled recording."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.recording import (
    FORMAT_VERSION,
    ExcludedSpans,
    LabelledRecordingReader,
    LabelledRecordingWriter,
    RecordingMeta,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

pytestmark = pytest.mark.t1

META = RecordingMeta("OPS", "mission-1", "internal", "IMX636", 1280, 720,
                     "sensor_power_on", "event")
SHUFFLE_ID, DEFLATE_ID = 2, 1
ROWS = [
    (0, "bias_diff", 10),
    (0, "bias_diff_on", 20),
    (0, "bias_diff_off", 30),
    (50, "bias_diff", 11),
    (50, "bias_diff", 12),
    (80, "bias_hpf", 5),
    (90, "bias_diff_on", 21),
]


def _write(
    path: Path,
    *,
    bias: Sequence[Sequence[tuple[int, str, int]]] = (),
    excluded: Sequence[ExcludedSpans] = (),
) -> None:
    with LabelledRecordingWriter(path, META, chunk_events=4) as writer:
        writer.write_events(EventsBatch(
            np.arange(0, 100, 10, dtype=np.int64),
            np.zeros(10, np.int16), np.zeros(10, np.int16),
            np.zeros(10, np.int8),
        ))
        for rows in bias:
            writer.write_bias(rows)
        for spans in excluded:
            writer.write_excluded(spans)


def _spans(*pairs: tuple[int, int]) -> ExcludedSpans:
    return ExcludedSpans(
        np.array([s for s, _ in pairs], np.int64),
        np.array([e for _, e in pairs], np.int64),
    )


def test_format_is_1_1() -> None:
    assert FORMAT_VERSION == "1.1"


def test_bias_log_round_trips_with_row_order_and_dtypes(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "rec.h5", bias=[ROWS])
    with LabelledRecordingReader(tmp_path / "rec.h5") as reader:
        log = reader.bias_log()
    assert len(log) == len(ROWS)
    assert log.time_us.dtype == np.int64
    assert log.name_id.dtype == np.int32
    assert log.value.dtype == np.int64
    assert log.names == (
        "bias_diff", "bias_diff_on", "bias_diff_off", "bias_hpf",
    )
    assert [(int(t), log.names[i], int(v)) for t, i, v in zip(
        log.time_us, log.name_id, log.value, strict=True)] == ROWS


def test_batched_writes_equal_one_shot(tmp_path: Path) -> None:
    _write(tmp_path / "one.h5", bias=[ROWS])
    _write(tmp_path / "many.h5",
           bias=[ROWS[:1], [], ROWS[1:4], ROWS[4:5], ROWS[5:]])
    with h5py.File(tmp_path / "one.h5") as one, h5py.File(
        tmp_path / "many.h5",
    ) as many:
        for name in ("names", "time_us", "name_id", "value"):
            np.testing.assert_array_equal(
                one[f"bias/{name}"][:], many[f"bias/{name}"][:],
            )


@pytest.mark.parametrize(
    ("time_us", "want"),
    [
        (-1, {}),
        (0, {"bias_diff": 10, "bias_diff_on": 20, "bias_diff_off": 30}),
        (49, {"bias_diff": 10, "bias_diff_on": 20, "bias_diff_off": 30}),
        (50, {"bias_diff": 12, "bias_diff_on": 20, "bias_diff_off": 30}),
        (79, {"bias_diff": 12, "bias_diff_on": 20, "bias_diff_off": 30}),
        (80, {"bias_diff": 12, "bias_diff_on": 20, "bias_diff_off": 30,
              "bias_hpf": 5}),
        (10_000, {"bias_diff": 12, "bias_diff_on": 21,
                  "bias_diff_off": 30, "bias_hpf": 5}),
    ],
)
def test_bias_at_is_the_vector_in_force(
    tmp_path: Path, time_us: int, want: dict[str, int],
) -> None:
    _write(tmp_path / "rec.h5", bias=[ROWS])
    with LabelledRecordingReader(tmp_path / "rec.h5") as reader:
        assert reader.bias_at(time_us) == want
        assert reader.bias_log().at(time_us) == want


def test_a_name_first_seen_late_is_absent_before_its_row(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "rec.h5", bias=[[(0, "a", 1)], [(5, "b", 2)]])
    with LabelledRecordingReader(tmp_path / "rec.h5") as reader:
        assert reader.bias_at(4) == {"a": 1}
        assert reader.bias_at(5) == {"a": 1, "b": 2}


def test_no_bias_rows_means_unknown(tmp_path: Path) -> None:
    _write(tmp_path / "rec.h5", bias=[[], []])
    with h5py.File(tmp_path / "rec.h5") as f:
        assert "bias" not in f
    with LabelledRecordingReader(tmp_path / "rec.h5") as reader:
        log = reader.bias_log()
        assert len(log) == 0
        assert log.names == ()
        assert log.time_us.dtype == np.int64
        assert reader.bias_at(0) == {}


def test_bias_time_must_not_go_back_and_a_bad_batch_is_not_written(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rec.h5"
    with LabelledRecordingWriter(path, META, chunk_events=4) as writer:
        with pytest.raises(ValueError, match="non-decreasing"):
            writer.write_bias([(5, "a", 1), (4, "a", 2)])
        writer.write_bias([(5, "a", 1), (5, "a", 2)])
        with pytest.raises(ValueError, match="non-decreasing"):
            writer.write_bias([(4, "a", 3)])
        writer.write_bias([(6, "b", 3)])
    with LabelledRecordingReader(path) as reader:
        assert reader.bias_log().value.tolist() == [1, 2, 3]
        assert reader.bias_log().names == ("a", "b")


def test_bias_tables_use_only_shuffle_and_gzip_and_plain_h5py_reads(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "rec.h5", bias=[ROWS],
           excluded=[_spans((1, 2))])
    with h5py.File(tmp_path / "rec.h5") as f:
        for name in ("bias/names", "bias/time_us", "bias/name_id",
                     "bias/value", "excluded/start_us", "excluded/end_us"):
            plist = f[name].id.get_create_plist()
            assert [plist.get_filter(i)[0]
                    for i in range(plist.get_nfilters())
                    ] == [SHUFFLE_ID, DEFLATE_ID], name
        assert f["bias/names"][:].tolist() == [
            b"bias_diff", b"bias_diff_on", b"bias_diff_off", b"bias_hpf",
        ]


def test_excluded_spans_round_trip_sorted_across_batches(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "rec.h5",
           excluded=[_spans((50, 60), (10, 20)), _spans((20, 30), (0, 5))])
    with LabelledRecordingReader(tmp_path / "rec.h5") as reader:
        spans = reader.excluded_spans()
    assert spans.start_us.dtype == spans.end_us.dtype == np.int64
    assert spans.start_us.tolist() == [0, 10, 20, 50]
    assert spans.end_us.tolist() == [5, 20, 30, 60]


@pytest.mark.parametrize(
    ("batches", "message"),
    [
        ([[(5, 5)]], "start < end"),
        ([[(6, 5)]], "start < end"),
        ([[(0, 10), (9, 20)]], "overlap"),
        ([[(0, 10), (2, 3)]], "overlap"),
        ([[(0, 10)], [(5, 15)]], "overlap"),
    ],
)
def test_bad_excluded_spans_raise(
    tmp_path: Path, batches: list[list[tuple[int, int]]], message: str,
) -> None:
    with LabelledRecordingWriter(
        tmp_path / "rec.h5", META, chunk_events=4,
    ) as writer, pytest.raises(ValueError, match=message):
        for pairs in batches:
            writer.write_excluded(_spans(*pairs))


def test_a_rejected_span_batch_is_not_kept(tmp_path: Path) -> None:
    path = tmp_path / "rec.h5"
    with LabelledRecordingWriter(path, META, chunk_events=4) as writer:
        writer.write_excluded(_spans((0, 10)))
        with pytest.raises(ValueError, match="overlap"):
            writer.write_excluded(_spans((20, 30), (5, 6)))
        writer.write_excluded(_spans((10, 12)))
    with LabelledRecordingReader(path) as reader:
        assert reader.excluded_spans().start_us.tolist() == [0, 10]


def test_no_excluded_spans_means_nothing_excluded(tmp_path: Path) -> None:
    _write(tmp_path / "rec.h5", excluded=[_spans()])
    with h5py.File(tmp_path / "rec.h5") as f:
        assert "excluded" not in f
    with LabelledRecordingReader(tmp_path / "rec.h5") as reader:
        spans = reader.excluded_spans()
    assert spans.start_us.size == 0
    assert spans.mask(np.array([0, 5])).tolist() == [False, False]
    assert spans.kept(0, 10) == [(0, 10)]


def test_mask_is_half_open() -> None:
    spans = _spans((10, 20), (30, 31))
    times = np.array([9, 10, 19, 20, 29, 30, 31, 100, -5])
    assert spans.mask(times).tolist() == [
        False, True, True, False, False, True, False, False, False,
    ]


@pytest.mark.parametrize(
    ("window", "want"),
    [
        ((0, 100), [(0, 10), (20, 30), (31, 100)]),
        ((12, 18), []),
        ((12, 25), [(20, 25)]),
        ((5, 10), [(5, 10)]),
        ((20, 30), [(20, 30)]),
        ((15, 35), [(20, 30), (31, 35)]),
        ((40, 40), []),
        ((50, 40), []),
    ],
)
def test_kept_returns_the_trusted_parts_of_a_window(
    window: tuple[int, int], want: list[tuple[int, int]],
) -> None:
    assert _spans((10, 20), (30, 31)).kept(*window) == want


def test_a_1_0_file_reads_with_no_bias_and_nothing_excluded(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "rec.h5", bias=[ROWS], excluded=[_spans((1, 2))])
    with h5py.File(tmp_path / "rec.h5", "r+") as f:
        del f["bias"], f["excluded"]
        f.attrs["format_version"] = "1.0"
    with LabelledRecordingReader(tmp_path / "rec.h5") as reader:
        assert reader.n_events == 10
        assert len(reader.bias_log()) == 0
        assert reader.bias_at(50) == {}
        assert reader.excluded_spans().start_us.size == 0
