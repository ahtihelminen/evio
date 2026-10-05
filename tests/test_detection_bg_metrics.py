"""Per-cell running statistics and 8-neighbour coactivity, checked by hand."""

from __future__ import annotations

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.detection.background.metrics import (
    compute_coactivity,
    compute_coactivity_pixels,
    update_stats_ema,
    update_stats_welford,
)
from evio.detection.grid import GridConfig

pytestmark = pytest.mark.t1

# Four 3x3 patterns, one per region: an isolated cell, two adjacent cells,
# a full block and the block's ring (the block minus its centre).
_ISOLATED = [(0, 0)]
_PAIR = [(0, 0), (0, 1)]
_BLOCK = [(row, col) for row in range(3) for col in range(3)]
_RING = [cell for cell in _BLOCK if cell != (1, 1)]
_PATTERNS = (_ISOLATED, _PAIR, _BLOCK, _RING)


def _zeroed_stats(shape: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    return tuple(np.zeros(shape) for _ in range(4))


def test_welford_matches_numpy_mean_and_variance() -> None:
    samples = np.array(
        [[[1.0, 4.0]], [[3.0, -2.0]], [[8.0, 0.5]], [[2.0, 7.0]]]
    )
    mean, var, m2, count = _zeroed_stats((1, 2))
    everywhere = np.ones((1, 2), dtype=bool)

    for sample in samples:
        update_stats_welford(sample, everywhere, mean, var, m2, count)

    np.testing.assert_allclose(mean, samples.mean(axis=0))
    np.testing.assert_allclose(var, samples.var(axis=0, ddof=1))
    np.testing.assert_array_equal(count, [[4, 4]])


def test_stats_updates_leave_unmasked_cells_untouched() -> None:
    values = np.array([10.0, 20.0])
    only_first = np.array([True, False])
    welford_state = [np.array([1.0, 5.0]) for _ in range(4)]
    ema_state = [np.array([1.0, 5.0]) for _ in range(3)]

    update_stats_welford(values, only_first, *welford_state)
    update_stats_ema(values, only_first, *ema_state, alpha=0.5)

    for array in welford_state + ema_state:
        assert array[1] == 5.0
        assert array[0] != 1.0


def test_ema_moves_mean_by_alpha_fraction() -> None:
    mean, var, _, count = _zeroed_stats(1)
    mean[0] = 10.0
    mask = np.array([True])

    update_stats_ema(np.array([20.0]), mask, mean, var, count, alpha=0.25)
    assert mean[0] == 10.0 + 0.25 * (20.0 - 10.0)  # 12.5

    update_stats_ema(np.array([20.0]), mask, mean, var, count, alpha=0.25)
    assert mean[0] == 12.5 + 0.25 * (20.0 - 12.5)  # 14.375


def test_coactivity_counts_active_neighbours() -> None:
    # Isolated: 0. Pair: 1 each. Block: 8 at the centre + 5 per edge cell
    # + 3 per corner = 40. Ring: 4 per edge cell + 2 per corner = 24.
    expected = [0.0, 2.0, 40.0, 24.0]

    time_bins = 2
    microstate = np.zeros((len(_PATTERNS), 3, 3, time_bins), dtype=np.int32)
    for region, pattern in enumerate(_PATTERNS):
        for index, (row, col) in enumerate(pattern):
            microstate[region, row, col, index % time_bins] = 1
    by_subcell = compute_coactivity(
        microstate.reshape(len(_PATTERNS), -1),
        sub_rows=3,
        sub_cols=3,
        time_bins=time_bins,
    )

    pixels = [
        (3 * region + col, row)
        for region, pattern in enumerate(_PATTERNS)
        for row, col in pattern
    ]
    x_coords, y_coords = np.array(pixels, dtype=np.int16).T
    events = EventsBatch(
        timestamps=np.arange(len(pixels), dtype=np.int64),
        x_coords=x_coords,
        y_coords=y_coords,
        polarities=np.ones(len(pixels), dtype=np.int8),
    )
    region_grid = GridConfig(
        width=12, height=3, rows=1, cols=4, time_bins=1, window_us=1_000
    )
    by_pixel = compute_coactivity_pixels(events, region_grid=region_grid)

    for scores in (by_subcell, by_pixel):
        np.testing.assert_array_equal(scores, expected)
        # Removing the centre drops its own 8 and 1 from each neighbour.
        assert scores[2] - scores[3] == 2 * 8
