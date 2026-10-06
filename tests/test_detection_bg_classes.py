"""BackgroundClassesClassifier calibrates on event time, then classifies."""

from __future__ import annotations

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.detection.background.classes import BackgroundClassesClassifier
from evio.detection.grid import GridConfig

pytestmark = pytest.mark.t1

# 8x8 sensor in 2x2 cells of 4x4 pixels.
_GRID = GridConfig(
    width=8, height=8, rows=2, cols=2, time_bins=1, window_us=400
)


def _batch(start_us: int, end_us: int) -> EventsBatch:
    """Busy top-left cell, a few events in the others, spanning the times."""
    pixels = [(x, y) for x in range(4) for y in range(4)] + [(5, 1), (2, 6)]
    x_coords, y_coords = np.array(pixels, dtype=np.int16).T
    return EventsBatch(
        timestamps=np.linspace(start_us, end_us, len(pixels)).astype(np.int64),
        x_coords=x_coords,
        y_coords=y_coords,
        polarities=np.ones(len(pixels), dtype=np.int8),
    )


def test_bg_classifier_calibrates_then_yields_class_map() -> None:
    classifier = BackgroundClassesClassifier(
        grid=_GRID, calibration_time_us=1_000
    )

    # Elapsed event time: 400 us, then 900 us -- both short of 1000.
    for start_us, end_us in ((0, 400), (500, 900)):
        result = classifier.update(_batch(start_us, end_us))
        assert result.calibrating
        assert result.class_map is None
        assert result.calibration_target_us == 1_000

    # 1100 us of event time have now passed.
    calibrated = classifier.update(_batch(1_000, 1_100))
    assert calibrated.class_map is not None
    assert calibrated.class_map.dtype == np.int32
    assert calibrated.class_map.shape == (_GRID.rows, _GRID.cols)

    after = classifier.update(_batch(1_200, 1_300))
    assert not after.calibrating
    np.testing.assert_array_equal(after.class_map, calibrated.class_map)


def test_bg_classifier_calibrating_is_false_on_the_completing_call() -> None:
    classifier = BackgroundClassesClassifier(
        grid=_GRID, calibration_time_us=1_000
    )
    classifier.update(_batch(0, 400))

    completing = classifier.update(_batch(500, 1_100))
    assert completing.class_map is not None
    assert not completing.calibrating
