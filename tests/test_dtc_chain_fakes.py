"""Fidelity for the chain test doubles.

`FakeTimer` stands in for a `Timer`; the toys in `tests/fakes/dtc_toys.py`
stand in for real detectors, trackers and classifiers. These tests fail when
a double stops keeping the property its chain tests rely on.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fakes.dtc_timer import FakeTimer
from fakes.dtc_toys import LOG, TOYS, registered

from evio.dtc.evidence import PROVIDERS
from evio.dtc.protocols import Context
from evio.dtc.registry import METHODS, create
from evio.dtc.timing import StageTiming, SystemTimer, TimingStats
from evio.dtc.types import Candidates, Step
from tests.support import dtc

pytestmark = pytest.mark.t1

STEP_US = 10_000
CONTEXT = Context("cam0", ("cam0", "cam1"), {}, STEP_US)


@pytest.fixture
def toys() -> Iterator[None]:
    with registered():
        yield


def step_with(index: int, count: int) -> Step:
    start = index * STEP_US
    batch = dtc.blob_events(start, start + STEP_US, 40, 30, count=count)
    return dtc.step_of(
        index, STEP_US, dtc.window("cam0", batch, start, start + STEP_US)
    )


def test_the_fake_timer_has_the_surface_of_the_system_timer() -> None:
    surface = {"wall_ns", "cpu_ns"}
    assert surface <= set(dir(FakeTimer()))
    assert surface <= set(dir(SystemTimer()))


def test_a_span_between_two_reads_is_exactly_one_tick() -> None:
    timer = FakeTimer(tick_ns=700, cpu_tick_ns=300)
    wall, cpu = timer.wall_ns(), timer.cpu_ns()
    assert timer.wall_ns() - wall == 700
    assert timer.cpu_ns() - cpu == 300


def test_the_fake_timer_is_strictly_increasing() -> None:
    timer = FakeTimer()
    reads = [timer.wall_ns() for _ in range(5)]
    assert reads == sorted(set(reads))


def test_fake_timer_spans_feed_timing_stats_like_real_ones() -> None:
    timer = FakeTimer(tick_ns=5_000)
    stats = TimingStats(window_us=STEP_US)
    for _ in range(3):
        start = timer.wall_ns()
        stats.record(StageTiming(timer.wall_ns() - start, 0))
    assert stats.percentile_ns(50) == 5_000
    assert stats.real_time_factor == pytest.approx(5_000 / 10_000_000)


def test_registered_toys_exist_inside_the_block_and_vanish_after() -> None:
    with registered():
        assert set(TOYS) <= set(METHODS)
        assert "toy-count" in PROVIDERS
    assert not set(TOYS) & set(METHODS)
    assert "toy-count" not in PROVIDERS


@pytest.mark.usefixtures("toys")
def test_toys_sharing_a_class_keep_their_own_manifest() -> None:
    assert METHODS["toy-classifier"].manifest.cadence == 2
    assert METHODS["toy-classifier-every-step"].manifest.cadence == 1
    assert METHODS["toy-batch-classifier"].manifest.evidence == ()


@pytest.mark.usefixtures("toys")
def test_the_toy_detector_emits_ranked_boxes_of_its_stream() -> None:
    detector = create("toy-detector", {"boxes": 3}, CONTEXT)
    items = detector.step(step_with(0, 50), ()).items
    assert [c.score for c in items] == [50, 49, 48]
    assert {c.stream for c in items} == {"cam0"}
    assert LOG["priors"] == [("cam0", ())]


@pytest.mark.usefixtures("toys")
def test_the_toy_tracker_keeps_live_ids_equal_to_its_track_ids() -> None:
    step = step_with(0, 50)
    found = create("toy-detector", {"boxes": 2}, CONTEXT).step(step, ())
    tracks = create("toy-tracker", {}, CONTEXT).step(step, found)
    assert len(tracks.items) == 2
    assert tracks.live_ids == {t.id for t in tracks.items}
    assert tracks.priors == tuple(t.region for t in tracks.items)
    assert create("toy-tracker", {}, CONTEXT).step(
        step, Candidates()
    ).live_ids == frozenset()


@pytest.mark.usefixtures("toys")
def test_the_exploder_raises_only_for_its_stream_and_step() -> None:
    cam0 = create("toy-exploder", {"stream": "cam0", "at": 1}, CONTEXT)
    cam1 = create("toy-exploder", {"stream": "cam1", "at": 1}, CONTEXT)
    cam0.step(step_with(0, 5), ())
    assert cam1.step(step_with(1, 5), ()).items
    with pytest.raises(ValueError, match="boom at 1"):
        cam0.step(step_with(1, 5), ())


@pytest.mark.usefixtures("toys")
def test_the_count_provider_reports_the_events_of_the_window() -> None:
    step = step_with(0, 37)
    found = create("toy-detector", {}, CONTEXT).step(step, ())
    tracks = create("toy-tracker", {}, CONTEXT).step(step, found)
    provider = PROVIDERS["toy-count"](CONTEXT)
    assert provider.compute(step, tracks) == {"cam0:0": 37}
    assert LOG["provider"] == [("cam0", 0)]
