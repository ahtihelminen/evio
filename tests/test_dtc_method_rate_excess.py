"""The rate-excess detector: blobs of events beyond the learned background.

Synthetic steps: static clutter that never moves, uniform noise, and a 28 px
target made of a few hundred extra events.
"""

from __future__ import annotations

import numpy as np
import pytest

from evio.dtc.conformance import check_method
from evio.dtc.errors import ConfigError
from evio.dtc.facts import EXTENT
from evio.dtc.manifest import Scope, Stage
from evio.dtc.methods.rate_excess import (
    RateExcess,
    RateExcessConfig,
    peaks,
    window_sums,
)
from evio.dtc.protocols import Context
from evio.dtc.registry import METHODS
from evio.dtc.types import Candidates, State, Step
from tests.fakes.dtc_rate_excess_reference import (
    ReferenceRateExcess,
)
from tests.fakes.dtc_rate_excess_reference import peaks as reference_peaks
from tests.support import dtc

pytestmark = pytest.mark.t1

STEP_US = 40_000
STREAM = "cam"
EXTENT_PX = (640, 360)
WARMUP = 25
SEED = 3


def build(extent: tuple[int, int] = EXTENT_PX, **config: float) -> RateExcess:
    context = Context(STREAM, (STREAM,), {EXTENT: extent}, STEP_US)
    return RateExcess(RateExcessConfig(**config), context)


def step_with(
    rng: np.random.Generator,
    index: int,
    target: tuple[float, float] | None,
    extent: tuple[int, int] = EXTENT_PX,
) -> Step:
    width, height = extent
    x = np.concatenate(
        [
            rng.integers(0, width, 1500),
            rng.integers(150, 180, 3000),  # a leafy corner that never moves
        ]
    )
    y = np.concatenate(
        [rng.integers(0, height, 1500), rng.integers(250, 280, 3000)]
    )
    if target is not None:
        x = np.concatenate([x, rng.integers(-14, 14, 300) + int(target[0])])
        y = np.concatenate([y, rng.integers(-14, 14, 300) + int(target[1])])
    x, y = np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)
    start = index * STEP_US
    t = np.sort(rng.integers(start, start + STEP_US, x.size))
    batch = dtc.events(t.tolist(), x.tolist(), y.tolist())
    window = dtc.window(STREAM, batch, start, start + STEP_US, extent)
    return dtc.step_of(index, STEP_US, window)


def warmed_up(rng: np.random.Generator, **config: float) -> RateExcess:
    detector = build(**config)
    for index in range(WARMUP):
        assert detector.step(step_with(rng, index, None), ()).items == ()
    return detector


def found(
    detector: RateExcess,
    rng: np.random.Generator,
    targets: list[tuple[float, float] | None],
) -> list[Candidates]:
    return [
        detector.step(step_with(rng, WARMUP + i, target), ())
        for i, target in enumerate(targets)
    ]


def test_registered_as_a_stream_detector_requiring_the_extent() -> None:
    manifest = METHODS["rate-excess"].manifest
    assert manifest.stages == (Stage.DETECT,)
    assert manifest.scope is Scope.STREAM
    assert manifest.requires == (EXTENT,)


def test_calibrating_with_progress_while_the_background_is_learned() -> None:
    rng = np.random.default_rng(SEED)
    detector = build()
    outputs = [
        detector.step(step_with(rng, i, (400, 100)), ()) for i in range(WARMUP)
    ]
    assert all(o.items == () for o in outputs)
    assert all(o.status.state is State.CALIBRATING for o in outputs)
    assert outputs[0].status.progress == pytest.approx(1 / WARMUP)
    assert outputs[-1].status.progress == 1.0
    after = found(detector, rng, [None])[0]
    assert after.status.state is State.OK


def test_a_moving_target_is_found_where_it_is() -> None:
    rng = np.random.default_rng(SEED)
    detector = warmed_up(rng)
    targets = [(60.0 + 10 * i, 60.0 + 5 * i) for i in range(40)]
    close = 0
    for target, output in zip(
        targets, found(detector, rng, targets), strict=True
    ):
        if output.items:
            cx, cy = output.items[0].region.center
            close += np.hypot(cx - target[0], cy - target[1]) <= 14
    assert close >= 36


def test_a_candidate_carries_its_z_as_score_and_feature() -> None:
    rng = np.random.default_rng(SEED)
    (output,) = found(warmed_up(rng), rng, [(400, 100)])
    candidate = output.items[0]
    assert candidate.stream == STREAM
    assert candidate.score > 8
    assert candidate.features == {"z": candidate.score}


def test_static_clutter_alone_raises_no_alarm() -> None:
    rng = np.random.default_rng(SEED)
    outputs = found(warmed_up(rng), rng, [None] * 150)
    assert sum(bool(o.items) for o in outputs) <= 2


def test_a_target_that_hovers_is_not_absorbed_into_the_background() -> None:
    rng = np.random.default_rng(SEED)
    outputs = found(warmed_up(rng), rng, [(400, 100)] * 80)
    assert sum(bool(o.items) for o in outputs[-20:]) >= 18


def test_the_strongest_blob_comes_first() -> None:
    rng = np.random.default_rng(SEED)
    (output,) = found(warmed_up(rng), rng, [(400, 100)])
    scores = [c.score for c in output.items]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("target", [(30, 30), (610, 330)])
def test_a_target_at_the_sensor_edge_is_found(
    target: tuple[float, float],
) -> None:
    rng = np.random.default_rng(SEED)
    outputs = found(warmed_up(rng), rng, [target] * 10)
    assert sum(bool(o.items) for o in outputs) >= 8


@pytest.mark.parametrize("window_cells", [4, 5, 6, 7])
def test_any_window_size_finds_the_target(window_cells: int) -> None:
    rng = np.random.default_rng(SEED)
    detector = warmed_up(rng, window_cells=window_cells)
    outputs = found(detector, rng, [(400, 100)] * 10)
    assert sum(bool(o.items) for o in outputs) >= 8


def test_window_sums_stay_exact_next_to_a_huge_value() -> None:
    values = np.full((90, 160), 0.02, np.float32)
    values[0, 0] = 5e7  # a hot cell far up the integral image
    assert window_sums(values, 5)[50, 100] == pytest.approx(25 * 0.02, rel=1e-3)


def test_an_extent_that_is_not_a_multiple_of_the_cell_works() -> None:
    extent = (203, 154)
    rng = np.random.default_rng(SEED)
    detector = build(extent)
    for index in range(WARMUP):
        detector.step(step_with(rng, index, None, extent), ())
    output = detector.step(step_with(rng, WARMUP, (190, 140), extent), ())
    assert output.items


def test_events_outside_the_extent_are_ignored() -> None:
    batch = dtc.events([0, 1, 2], [-5, 64, 3], [3, 3, 99])
    window = dtc.window(STREAM, batch, 0, STEP_US, (64, 48))
    detector = build((64, 48), warmup_steps=1)
    assert detector.step(dtc.step_of(0, STEP_US, window), ()).items == ()


def test_an_extent_smaller_than_the_window_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="smaller than"):
        build((16, 64))


def test_reset_zeroes_the_learned_background() -> None:
    rng = np.random.default_rng(SEED)
    detector = warmed_up(rng)
    detector.reset()
    assert not detector._mean.any()
    output = detector.step(step_with(rng, 0, (400, 100)), ())
    assert output.status.state is State.CALIBRATING


def test_conforms_on_a_small_extent() -> None:
    assert check_method("rate-excess", facts={EXTENT: (128, 96)}) == []
    assert check_method("rate-excess", facts={EXTENT: (130, 98)}) == []


def build_pair(
    extent: tuple[int, int], **config: float
) -> tuple[RateExcess, ReferenceRateExcess]:
    context = Context(STREAM, (STREAM,), {EXTENT: extent}, STEP_US)
    config_ = RateExcessConfig(**config)
    return RateExcess(config_, context), ReferenceRateExcess(config_, context)


@pytest.mark.parametrize(
    ("extent", "window_cells"),
    [((160, 120), 5), ((203, 154), 4), ((130, 98), 7), ((97, 61), 6)],
)
def test_the_fast_detector_matches_the_reference(
    extent: tuple[int, int], window_cells: int
) -> None:
    rng = np.random.default_rng(SEED)
    fast, reference = build_pair(extent, window_cells=window_cells)
    width, height = extent
    blobs = [(width * 0.3, height * 0.4), (width - 6, height - 5), (4, 3)]
    for index in range(WARMUP + 20):
        step = step_with(rng, index, None, extent)
        if index >= WARMUP:
            blob = blobs[index % len(blobs)]
            step = step_with(rng, index, blob, extent)
        got, want = fast.step(step, ()), reference.step(step, ())
        assert got.status == want.status
        assert [c.region.box for c in got.items] == [
            c.region.box for c in want.items
        ]
        assert [c.score for c in got.items] == pytest.approx(
            [c.score for c in want.items], rel=1e-5
        )
        np.testing.assert_allclose(fast._mean, reference._mean, rtol=1e-5)
        np.testing.assert_allclose(fast._spread, reference._spread, rtol=1e-4)
    assert want.items


def test_the_fast_z_map_matches_the_reference_beside_a_hot_cell() -> None:
    rng = np.random.default_rng(SEED)
    extent = (160, 120)
    fast, reference = build_pair(extent)
    for index in range(WARMUP):
        step = step_with(rng, index, None, extent)
        fast.step(step, ())
        reference.step(step, ())
    fast._mean[3, 3] = reference._mean[3, 3] = 5e4
    counts = fast._count_map(
        step_with(rng, WARMUP, (80, 60), extent).windows[STREAM].events
    )
    np.testing.assert_allclose(
        fast._z_map(counts), reference._z_map(counts), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("size", [3, 4, 5, 6])
@pytest.mark.parametrize("hot", [0, 40, 1500])
def test_peaks_match_the_reference_however_many_pass(
    size: int, hot: int
) -> None:
    rng = np.random.default_rng(SEED)
    z = rng.normal(0, 1, (37, 53))
    z.flat[rng.choice(z.size, hot, replace=False)] += 20
    z[5:8, 5:8] = z.max()  # a plateau: equal values all count as summits
    assert peaks(z, size, 8.0) == reference_peaks(z, size, 8.0)


def test_the_fast_detector_matches_on_empty_and_stray_events() -> None:
    rng = np.random.default_rng(SEED)
    extent = (130, 98)
    fast, reference = build_pair(extent)
    for index in range(WARMUP + 6):
        step = step_with(rng, index, None, extent)
        if index % 3 == 0:
            window = step.windows[STREAM]
            if index >= WARMUP:
                x = np.array([-5, 0, 129, 130, 4000], np.int16)
                y = np.array([3, 97, 98, 5, -1], np.int16)
                batch = dtc.events([0] * 5, x.tolist(), y.tolist())
            else:
                batch = dtc.empty_events()
            step = dtc.step_of(
                index,
                STEP_US,
                dtc.window(STREAM, batch, window.t_start_us, window.t_end_us),
            )
        got, want = fast.step(step, ()), reference.step(step, ())
        assert [c.region.box for c in got.items] == [
            c.region.box for c in want.items
        ]
        np.testing.assert_allclose(fast._mean, reference._mean, rtol=1e-5)
        np.testing.assert_allclose(fast._spread, reference._spread, rtol=1e-4)


def test_a_reset_detector_repeats_a_fresh_one() -> None:
    extent = (130, 98)

    def run(detector: RateExcess) -> list[Candidates]:
        rng = np.random.default_rng(SEED)
        return [
            detector.step(
                step_with(rng, i, (60, 40) if i >= WARMUP else None, extent),
                (),
            )
            for i in range(WARMUP + 5)
        ]

    reused = build(extent)
    first = run(reused)
    reused.reset()
    assert run(reused) == first == run(build(extent))
