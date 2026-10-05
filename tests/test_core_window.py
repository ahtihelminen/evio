"""Rolling time windows over decoded event batches."""

from __future__ import annotations

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.core.window import RollingWindowBatchProcessor

pytestmark = pytest.mark.t1


def _batch(*timestamps: int) -> EventsBatch:
    """Events whose x coordinate is their timestamp, so fields are traceable."""
    stamps = np.array(timestamps, np.int64)
    return EventsBatch(
        timestamps=stamps,
        x_coords=stamps.astype(np.int16),
        y_coords=np.zeros(stamps.size, np.int16),
        polarities=np.ones(stamps.size, np.int8),
    )


def _stamps(windows: list[EventsBatch]) -> list[list[int]]:
    return [window.timestamps.tolist() for window in windows]


def test_rolling_window_is_left_aligned_and_advances_by_step() -> None:
    processor = RollingWindowBatchProcessor(window_length=10, step_us=5)

    windows = list(processor.process([_batch(0, 3, 9, 10, 12, 19, 20, 25)]))

    # 10 sits at exactly t + window_length of the first window: excluded.
    assert _stamps(windows) == [[0, 3, 9], [9, 10, 12], [10, 12, 19], [19, 20]]


def test_rolling_window_skips_empty_windows() -> None:
    processor = RollingWindowBatchProcessor(window_length=10, step_us=10)

    windows = list(processor.process([_batch(0, 5, 100, 105, 200)], final=True))

    assert _stamps(windows) == [[0, 5], [100, 105], [200]]


def test_rolling_window_joins_events_across_batches() -> None:
    processor = RollingWindowBatchProcessor(window_length=10, step_us=10)
    batches = [_batch(0, 4), _batch(6, 8), _batch(12, 15), _batch(25)]

    windows = list(processor.process(batches))

    assert _stamps(windows) == [[0, 4, 6, 8], [12, 15]]
    assert [window.x_coords.tolist() for window in windows] == _stamps(windows)


def test_rolling_window_final_flushes_and_resets() -> None:
    processor = RollingWindowBatchProcessor(window_length=10, step_us=10)

    first_run = list(processor.process([_batch(0, 5, 12)], final=True))
    # A processor still anchored at 0 would cut [1000, 1010) and [1010, 1020).
    second_run = list(processor.process([_batch(1005, 1012)], final=True))

    assert _stamps(first_run) == [[0, 5], [12]]
    assert _stamps(second_run) == [[1005, 1012]]
