"""Cues: priors from outside the chain, carried on the step to every stage
that wants them and routed by stream."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from fakes.dtc_timer import FakeTimer
from fakes.dtc_toys import LOG, registered

from evio.dtc.chain import build
from evio.dtc.errors import ConfigError
from evio.dtc.facts import EXTENT
from evio.dtc.result import Pipeline
from evio.dtc.runner import steps_from_batches, with_cues, zip_steps
from evio.dtc.sources import stitch
from evio.dtc.taps import TapBus
from evio.dtc.types import Cue, Region, Step
from tests.support.dtc import events, step_of, window
from tests.support.dtc_chain import STEP_US, STREAMS, spec_of, step_at

pytestmark = pytest.mark.t1

RADAR = Cue("radar", "cam0", Region((10, 10, 20, 20)), source="radar0")
SITE = Cue("trees", None, Region((0, 0, 50, 50)), source="site-map")
OTHER = Cue("hand-off", "cam1", Region(point=(5, 5)), track_id="cam9:3")


@pytest.fixture(autouse=True)
def toys() -> Iterator[None]:
    with registered():
        yield


def make(*methods: str, bus: TapBus | None = None) -> Pipeline:
    return build(
        spec_of(*methods), {EXTENT: (1280, 720)}, streams=STREAMS,
        step_us=STEP_US, timer=FakeTimer(), bus=bus,
    )


def cued(index: int, *cues: Cue) -> Step:
    return replace(step_at(index, *STREAMS), cues=cues)


def test_a_step_restricted_to_a_stream_keeps_its_cues_and_global_ones() -> None:
    step = cued(0, RADAR, SITE, OTHER)

    assert step.only("cam0").cues == (RADAR, SITE)
    assert step.only("cam1").cues == (SITE, OTHER)


def test_a_step_without_cues_has_none() -> None:
    assert step_at(0, *STREAMS).cues == ()


def test_each_stream_s_method_sees_only_its_cues() -> None:
    make("toy-detector").step(cued(0, RADAR, SITE, OTHER))

    assert dict(LOG["cues"]) == {"cam0": (RADAR, SITE), "cam1": (SITE, OTHER)}


def test_a_joint_method_sees_every_cue() -> None:
    make("toy-joint-detector").step(cued(0, RADAR, SITE, OTHER))

    assert LOG["joint-cues"] == [(RADAR, SITE, OTHER)]


def test_a_cue_for_a_stream_not_built_is_refused() -> None:
    with pytest.raises(ConfigError, match="cam7"):
        make("toy-detector").step(cued(0, replace(RADAR, stream="cam7")))


def test_the_cues_of_a_step_are_published_on_a_tap() -> None:
    bus, seen = TapBus(), []
    bus.subscribe("cues", lambda index, cues: seen.append((index, cues)))

    make("toy-detector", bus=bus).step(cued(0, RADAR))

    assert seen == [(0, (RADAR,))]


def test_recorded_cues_are_put_back_on_their_steps() -> None:
    steps = [step_at(i, *STREAMS) for i in range(3)]

    replayed = list(with_cues(steps, {1: (RADAR,), 2: (SITE, OTHER)}))

    assert [s.cues for s in replayed] == [(), (RADAR,), (SITE, OTHER)]


def test_zipped_streams_keep_the_cues_of_both() -> None:
    def cued_steps(stream: str, cue: Cue) -> Iterator:
        batch = events([5, 15], [1, 2], [3, 4])
        for step in steps_from_batches([batch], stream, STEP_US, None, 0):
            yield replace(step, cues=(cue,))

    first = next(zip_steps(
        cued_steps("left", replace(RADAR, stream="left")),
        cued_steps("right", replace(OTHER, stream="right")),
    ))

    assert [c.kind for c in first.cues] == ["radar", "hand-off"]


def test_a_mosaic_moves_tile_cues_into_its_own_space() -> None:
    a = window("a", events([5], [1], [3]), 0, STEP_US)
    b = window("b", events([5], [7], [9]), 0, STEP_US)
    step = replace(step_of(0, STEP_US, a, b), cues=(
        Cue("radar", "b", Region((10, 10, 20, 20))),
        Cue("trees", None, Region((0, 0, 5, 5))),
        Cue("lost", "c", Region(point=(1, 1))),
    ))

    mosaic = stitch(step, [("a", 0, 0), ("b", 1280, 0)], "m")

    assert [(c.kind, c.stream) for c in mosaic.cues] == [
        ("radar", "m"), ("trees", None),
    ]
    assert mosaic.cues[0].region.box == (1290, 10, 1300, 20)
