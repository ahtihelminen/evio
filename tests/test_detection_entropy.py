"""Per-cell Shannon and Bernoulli entropy on hand-placed events."""

from __future__ import annotations

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.detection.features.entropy import (
    cell_bernoulli_entropy,
    cell_shannon_entropy_axis,
)
from evio.detection.grid import GridConfig

pytestmark = pytest.mark.t1


def _batch(events: list[tuple[int, int, int]]) -> EventsBatch:
    x_coords, y_coords, timestamps = np.array(events).T
    return EventsBatch(
        timestamps=timestamps.astype(np.int64),
        x_coords=x_coords.astype(np.int16),
        y_coords=y_coords.astype(np.int16),
        polarities=np.ones(len(events), dtype=np.int8),
    )


def test_entropy_zero_for_single_bin_and_log2n_for_uniform() -> None:
    # Two 2x2-pixel cells side by side, four 100 us time bins.
    grid = GridConfig(
        width=4, height=2, rows=1, cols=2, time_bins=4, window_us=400
    )

    # axis="t": bins are the cell's pixels. Cell 0 hits one pixel four
    # times; cell 1 hits each of its four pixels once.
    over_pixels = _batch(
        [(0, 0, t) for t in (0, 1, 2, 3)]
        + [(2, 0, 0), (3, 0, 1), (2, 1, 2), (3, 1, 3)]
    )
    np.testing.assert_allclose(
        cell_shannon_entropy_axis(over_pixels, grid, axis="t"),
        [0.0, np.log2(4)],
    )

    # axis="x": bins are (time_bin, y). Both cells stay on one row; cell 0
    # fires inside one time bin, cell 1 once in each of the four.
    over_time = _batch(
        [(0, 0, t) for t in (0, 10, 20, 30)]
        + [(2, 0, t) for t in (0, 100, 200, 300)]
    )
    np.testing.assert_allclose(
        cell_shannon_entropy_axis(over_time, grid, axis="x"),
        [0.0, np.log2(4)],
    )


def test_bernoulli_entropy_is_bounded_and_zero_when_empty() -> None:
    # One pixel per cell, so a cell's entropy is that pixel's Bernoulli
    # entropy. Four 100 us bins; p = fraction of bins the pixel fired in.
    grid = GridConfig(
        width=4, height=1, rows=1, cols=4, time_bins=4, window_us=400
    )
    events = _batch(
        [(0, 0, t) for t in (0, 100, 200, 300)]  # p = 1    -> 0
        + [(1, 0, 0), (1, 0, 150)]  # p = 1/2  -> 1
        + [(2, 0, 50)]  # p = 1/4  -> 0.811
        # pixel 3 never fires: empty cell -> 0
    )

    entropy = cell_bernoulli_entropy(events, grid)

    quarter = -(0.25 * np.log2(0.25) + 0.75 * np.log2(0.75))
    np.testing.assert_allclose(entropy, [0.0, 1.0, quarter, 0.0], rtol=1e-5)
    assert np.all((entropy >= 0.0) & (entropy <= 1.0))
    assert entropy[3] == 0.0
