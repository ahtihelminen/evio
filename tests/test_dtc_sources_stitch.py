"""Stitching windows into a mosaic: offsets, order, extent, purity."""

from __future__ import annotations

import numpy as np
import pytest

from evio.dtc.sources import seam_columns, stitch
from evio.dtc.types import Step
from tests.support.dtc import empty_events, events, step_of, window

pytestmark = pytest.mark.t1

STEP_US = 10_000
LAYOUT = [("a", 0, 0), ("b", 1280, 0)]


def two_tiles() -> Step:
    a = window("a", events([5, 30], [1, 2], [3, 4]), 0, STEP_US)
    b = window("b", events([10, 30], [7, 8], [9, 10], polarity=0), 0, STEP_US)
    return step_of(0, STEP_US, a, b)


def test_events_move_into_mosaic_space_in_time_order() -> None:
    mosaic = stitch(two_tiles(), LAYOUT, "mosaic").windows["mosaic"]

    assert mosaic.events.timestamps.tolist() == [5, 10, 30, 30]
    assert mosaic.events.x_coords.tolist() == [1, 1287, 2, 1288]
    assert mosaic.events.y_coords.tolist() == [3, 9, 4, 10]
    assert mosaic.events.polarities.tolist() == [1, 0, 1, 0]


def test_step_keeps_time_span_and_holds_one_window() -> None:
    mosaic = stitch(two_tiles(), LAYOUT, "mosaic")

    assert list(mosaic.windows) == ["mosaic"]
    assert (mosaic.index, mosaic.t_start_us, mosaic.t_end_us) == (0, 0, STEP_US)
    assert mosaic.windows["mosaic"].stream == "mosaic"
    assert mosaic.windows["mosaic"].extent == (2560, 720)


def test_extent_is_unknown_when_a_tile_extent_is_unknown() -> None:
    a = window("a", empty_events(), 0, STEP_US, extent=None)
    b = window("b", empty_events(), 0, STEP_US)

    mosaic = stitch(step_of(0, STEP_US, a, b), LAYOUT, "m")

    assert mosaic.windows["m"].extent is None


def test_vertical_offset_grows_the_extent() -> None:
    a = window("a", empty_events(), 0, STEP_US)
    b = window("b", empty_events(), 0, STEP_US)

    mosaic = stitch(
        step_of(0, STEP_US, a, b), [("a", 0, 0), ("b", 0, 720)], "m"
    )

    assert mosaic.windows["m"].extent == (1280, 1440)


def test_empty_windows_stitch_to_an_empty_window() -> None:
    a = window("a", empty_events(), 0, STEP_US)
    b = window("b", events([4], [1], [1]), 0, STEP_US)

    empty_b = window("b", empty_events(), 0, STEP_US)
    both = stitch(step_of(0, STEP_US, a, empty_b), LAYOUT, "m").windows["m"]
    one = stitch(step_of(0, STEP_US, a, b), LAYOUT, "m").windows["m"]

    assert len(both.events.timestamps) == 0
    assert one.events.x_coords.tolist() == [1281]


def test_input_windows_are_not_mutated() -> None:
    step = two_tiles()
    before = {k: w.events.x_coords.copy() for k, w in step.windows.items()}

    stitch(step, LAYOUT, "mosaic")

    for key, window_ in step.windows.items():
        assert np.array_equal(window_.events.x_coords, before[key])


def test_seam_columns_are_the_interior_joins() -> None:
    layout = [(f"c{i}", i * 1280, 0) for i in range(3)]
    widths = {f"c{i}": 1280 for i in range(3)}

    assert seam_columns(layout, widths) == (1280, 2560)
    assert seam_columns(LAYOUT[:1], {"a": 1280}) == ()
