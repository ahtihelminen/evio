"""`DatFileSource` over a synthetic `.dat`: its windows and its geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from support.dat import events_at, write_dat

from evio.io.dat_file import DatFileSource

pytestmark = pytest.mark.t1

# Events sit on and next to every window bound, so moving either bound by
# one microsecond moves an event into the neighbouring range.
_TIMESTAMPS = [100, 150, 199, 200, 201, 299, 300, 399, 400]


def test_dat_file_source_ranges_are_left_aligned_and_match_timestamps(
    tmp_path: Path,
) -> None:
    dat = write_dat(tmp_path / "events.dat", events_at(_TIMESTAMPS))
    source = DatFileSource(str(dat), window_length_us=100, step_us=100)
    ranges = list(source.ranges())

    assert [r.start_ts_us for r in ranges] == [100, 200, 300, 400]
    assert [r.end_ts_us for r in ranges] == [200, 300, 400, 500]
    timestamps = np.asarray(_TIMESTAMPS)
    for window in ranges:
        expected = timestamps[
            (timestamps >= window.start_ts_us)
            & (timestamps < window.end_ts_us)
        ]
        held = timestamps[window.start_idx:window.end_idx]
        np.testing.assert_array_equal(held, expected)


def test_dat_file_source_length_and_geometry(tmp_path: Path) -> None:
    dat = write_dat(tmp_path / "events.dat", events_at(_TIMESTAMPS))

    default = DatFileSource(str(dat), step_us=100)
    resized = DatFileSource(str(dat), step_us=50, width=640, height=480)

    assert len(default) == len(list(default.ranges())) == 4
    assert len(resized) == len(list(resized.ranges())) == 7
    assert (default.width, default.height) == (1280, 720)
    assert (resized.width, resized.height) == (640, 480)
