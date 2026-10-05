"""Microstate vectors: event counts per (row, col, time_bin), by hand."""

from __future__ import annotations

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.detection.background.microstate import (
    microstate_from_event_tensor,
    microstate_from_events,
)
from evio.detection.grid import GridConfig

pytestmark = pytest.mark.t1

# 4x4 sensor, 2x2 cells of 2x2 pixels, a 100 us window in two 50 us bins.
_GRID = GridConfig(
    width=4, height=4, rows=2, cols=2, time_bins=2, window_us=100
)
_START_US = 1_000

# (x, y, offset_us) -> (row, col, time_bin)
_EVENTS = [
    (0, 0, 0),  # (0, 0, 0)
    (1, 1, 10),  # (0, 0, 0)
    (3, 0, 60),  # (0, 1, 1)
    (0, 3, 20),  # (1, 0, 0)
    (3, 3, 99),  # (1, 1, 1)
]


def _batch() -> EventsBatch:
    x_coords, y_coords, offsets = np.array(_EVENTS).T
    return EventsBatch(
        timestamps=(_START_US + offsets).astype(np.int64),
        x_coords=x_coords.astype(np.int16),
        y_coords=y_coords.astype(np.int16),
        polarities=np.ones(len(_EVENTS), dtype=np.int8),
    )


def test_microstate_counts_events_per_cell_and_time_bin() -> None:
    microstate = microstate_from_events(_batch(), _GRID)

    # Index = (row * cols + col) * time_bins + time_bin.
    expected = np.zeros(8, dtype=np.int32)
    expected[(0 * 2 + 0) * 2 + 0] = 2
    expected[(0 * 2 + 1) * 2 + 1] = 1
    expected[(1 * 2 + 0) * 2 + 0] = 1
    expected[(1 * 2 + 1) * 2 + 1] = 1
    np.testing.assert_array_equal(microstate, expected)
    assert microstate.sum() == len(_EVENTS)


def test_microstate_tensor_matches_event_path() -> None:
    event_tensor = np.zeros(
        (_GRID.width, _GRID.height, _GRID.window_us), dtype=np.int8
    )
    for x_coord, y_coord, offset_us in _EVENTS:
        event_tensor[x_coord, y_coord, offset_us] = 1

    from_tensor = microstate_from_event_tensor(
        event_tensor,
        rows=_GRID.rows,
        cols=_GRID.cols,
        time_bins=_GRID.time_bins,
    )

    np.testing.assert_array_equal(
        from_tensor, microstate_from_events(_batch(), _GRID)
    )
