"""Slicing batches into steps, zipping streams and driving a pipeline."""

from __future__ import annotations

import copy
from collections.abc import Iterator

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.dtc.runner import run, steps_from_batches, zip_steps
from evio.dtc.types import Step
from tests.fakes.dtc_counting_pipeline import CountingPipeline
from tests.support.dtc import empty_events, events

pytestmark = pytest.mark.t1

STEP_US = 100


def _times(step: Step, stream: str = "cam") -> list[int]:
    return step.windows[stream].events.timestamps.tolist()


def _slice(
    batches: list[EventsBatch], extent: tuple[int, int] | None = None
) -> list[Step]:
    return list(steps_from_batches(batches, "cam", STEP_US, extent))


def _batch(*times: int) -> EventsBatch:
    return events(list(times), [1] * len(times), [2] * len(times))


def test_steps_anchor_at_the_first_event_and_are_half_open() -> None:
    steps = _slice([_batch(1000, 1050, 1099, 1100, 1250)])

    assert [(s.index, s.t_start_us, s.t_end_us) for s in steps] == [
        (0, 1000, 1100),
        (1, 1100, 1200),
        (2, 1200, 1300),
    ]
    assert [_times(s) for s in steps] == [[1000, 1050, 1099], [1100], [1250]]


def test_event_exactly_on_a_boundary_opens_the_next_step() -> None:
    steps = _slice([_batch(0, 100)])

    assert [_times(s) for s in steps] == [[0], [100]]


def test_a_single_event_makes_one_full_span_step() -> None:
    (step,) = _slice([_batch(500)])

    assert (step.index, step.t_start_us, step.t_end_us) == (0, 500, 600)
    assert _times(step) == [500]


def test_no_events_make_no_steps() -> None:
    assert _slice([]) == []
    assert _slice([empty_events(), empty_events()]) == []


def test_a_gap_longer_than_a_step_yields_empty_steps() -> None:
    steps = _slice([_batch(0, 10), _batch(350)])

    assert [s.index for s in steps] == [0, 1, 2, 3]
    assert [_times(s) for s in steps] == [[0, 10], [], [], [350]]
    assert steps[2].t_start_us == 200
    assert steps[2].windows["cam"].t_end_us == 300


def test_a_batch_straddling_steps_is_split_across_them() -> None:
    steps = _slice([_batch(0, 90), _batch(95, 105, 190), _batch(210)])

    assert [_times(s) for s in steps] == [
        [0, 90, 95],
        [105, 190],
        [210],
    ]


def test_the_split_matches_one_big_batch() -> None:
    times = [3, 40, 99, 100, 101, 250, 251, 399, 700]
    whole = _slice([_batch(*times)])
    parts = [times[:2], times[2:5], times[5:]]
    pieces = _slice([_batch(*part) for part in parts])

    assert [_times(s) for s in pieces] == [_times(s) for s in whole]


def test_the_last_step_is_emitted_on_exhaustion_with_its_full_span() -> None:
    steps = _slice([_batch(0, 150)])

    assert [(s.t_start_us, s.t_end_us) for s in steps] == [(0, 100), (100, 200)]
    assert _times(steps[-1]) == [150]


def test_empty_batches_between_data_are_ignored() -> None:
    steps = _slice([empty_events(), _batch(7), empty_events(), _batch(8)])

    assert [(s.t_start_us, _times(s)) for s in steps] == [(7, [7, 8])]


def test_windows_carry_stream_extent_and_span() -> None:
    (step,) = _slice([_batch(5)], extent=(64, 48))
    window = step.windows["cam"]

    assert window.stream == "cam"
    assert window.extent == (64, 48)
    assert (window.t_start_us, window.t_end_us) == (5, 105)


def test_all_columns_follow_the_split() -> None:
    batch = EventsBatch(
        np.array([0, 50, 120], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.int16),
        np.array([4, 5, 6], dtype=np.int16),
        np.array([1, 0, 1], dtype=np.int8),
    )
    first, second = _slice([batch])

    assert first.windows["cam"].events.x_coords.tolist() == [1, 2]
    assert first.windows["cam"].events.polarities.tolist() == [1, 0]
    assert second.windows["cam"].events.y_coords.tolist() == [6]


def test_input_arrays_are_never_mutated() -> None:
    batch = _batch(0, 50, 150, 250)
    before = copy.deepcopy(batch)

    _slice([batch])

    for name in vars(batch):
        assert np.array_equal(getattr(batch, name), getattr(before, name))


def test_windows_stay_valid_after_later_steps_are_produced() -> None:
    steps = _slice([_batch(0, 10), _batch(110), _batch(210)])

    assert [_times(s) for s in steps] == [[0, 10], [110], [210]]


def test_steps_are_lazy() -> None:
    consumed: list[int] = []

    def batches() -> Iterator[EventsBatch]:
        for batch in (_batch(0), _batch(150), _batch(900)):
            consumed.append(len(consumed))
            yield batch

    first = next(steps_from_batches(batches(), "cam", STEP_US))

    assert first.index == 0
    assert consumed == [0, 1]


def test_zip_aligns_equal_indices_into_one_step() -> None:
    left = steps_from_batches([_batch(0, 150)], "left", STEP_US, (8, 8), 0)
    right = steps_from_batches([_batch(20, 130)], "right", STEP_US, None, 0)

    zipped = list(zip_steps(left, right))

    assert [s.index for s in zipped] == [0, 1]
    assert list(zipped[0].windows) == ["left", "right"]
    assert _times(zipped[0], "right") == [20]
    assert _times(zipped[1], "left") == [150]
    assert (zipped[0].t_start_us, zipped[0].t_end_us) == (0, 100)
    assert zipped[0].windows["right"].t_start_us == 0


def test_zip_of_streams_that_end_apart_raises() -> None:
    left = steps_from_batches([_batch(0, 150)], "left", STEP_US)
    right = steps_from_batches([_batch(0)], "right", STEP_US)

    with pytest.raises(ValueError, match="shorter"):
        list(zip_steps(left, right))


def test_zip_of_streams_sharing_a_key_raises_rather_than_dropping_one() -> None:
    first = steps_from_batches([_batch(0)], "cam", STEP_US)
    second = steps_from_batches([_batch(0)], "cam", STEP_US)

    with pytest.raises(ValueError, match="cam"):
        list(zip_steps(first, second))


def test_zip_of_misaligned_grids_names_the_streams() -> None:
    left = steps_from_batches([_batch(0, 150)], "left", STEP_US)
    right = steps_from_batches([_batch(20, 130)], "right", STEP_US)

    with pytest.raises(ValueError, match=r"left.*right"):
        list(zip_steps(left, right))


def test_a_shared_start_gives_empty_leading_steps() -> None:
    steps = list(steps_from_batches([_batch(250)], "cam", STEP_US, None, 0))

    assert [(s.index, s.t_start_us, _times(s)) for s in steps] == [
        (0, 0, []),
        (1, 100, []),
        (2, 200, [250]),
    ]


def test_events_before_the_start_raise() -> None:
    with pytest.raises(ValueError, match="time-sorted"):
        list(steps_from_batches([_batch(5)], "cam", STEP_US, None, 10))


def test_a_batch_that_is_not_sorted_raises() -> None:
    with pytest.raises(ValueError, match="time-sorted"):
        _slice([_batch(0, 50, 40)])


def test_a_batch_older_than_the_previous_one_raises() -> None:
    with pytest.raises(ValueError, match="time-sorted"):
        _slice([_batch(0, 150), _batch(120)])


def test_run_equals_a_manual_loop() -> None:
    steps = _slice([_batch(0, 150, 420)])
    manual = CountingPipeline()

    results = list(run(CountingPipeline(), steps))

    assert results == [manual.step(step) for step in steps]
    assert [r.index for r in results] == [0, 1, 2, 3, 4]


def test_run_is_lazy_and_steps_in_order() -> None:
    pipeline = CountingPipeline()
    results = run(pipeline, _slice([_batch(0, 250)]))

    assert pipeline.seen == []
    next(results)
    assert pipeline.seen == [0]


def test_counting_pipeline_double_notices_reordered_steps() -> None:
    steps = _slice([_batch(0, 150, 250)])
    ordered, reversed_ = CountingPipeline(), CountingPipeline()

    in_order = [ordered.step(s).provenance for s in steps]
    backwards = [reversed_.step(s).index for s in reversed(steps)]

    assert in_order == ["1", "2", "3"]
    assert backwards == [2, 1, 0]
    ordered.reset()
    assert ordered.seen == []
