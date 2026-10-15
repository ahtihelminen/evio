"""The Kalman `TrackManager`, adapted as a DTC tracker."""

from __future__ import annotations

import re

import pytest

from evio.dtc.conformance import check_method
from evio.dtc.errors import ConfigError
from evio.dtc.facts import EXTENT
from evio.dtc.manifest import Scope, Stage
from evio.dtc.methods.trackmanager import (
    TrackManagerOptions,
    TrackManagerTracker,
)
from evio.dtc.protocols import Context
from evio.dtc.registry import METHODS, parse_config
from evio.dtc.types import Candidate, Candidates, Region, Tracks
from tests.support import dtc

pytestmark = pytest.mark.t1

STEP_US = 10_000
STREAM = "cam"
ULID = re.compile(r"[0-9A-Z]{26}")
BOX = (10.0, 10.0, 30.0, 30.0)


def build(config: TrackManagerOptions | None = None) -> TrackManagerTracker:
    context = Context(STREAM, (STREAM,), {EXTENT: (64, 48)}, STEP_US)
    return TrackManagerTracker(config or TrackManagerOptions(), context)


def seen(region: Region | None, score: float = 0.8) -> Candidates:
    items = (Candidate(STREAM, region, score),) if region else ()
    return Candidates(items)


def run(
    tracker: TrackManagerTracker, candidates: list[Candidates]
) -> list[Tracks]:
    outputs = []
    for i, found in enumerate(candidates):
        window = dtc.window(
            STREAM, dtc.empty_events(), i * STEP_US, (i + 1) * STEP_US
        )
        outputs.append(tracker.step(dtc.step_of(i, STEP_US, window), found))
    return outputs


def test_registered_as_a_stream_tracker() -> None:
    manifest = METHODS["trackmanager"].manifest
    assert manifest.stages == (Stage.TRACK,)
    assert manifest.scope is Scope.STREAM


def test_confirms_a_track_after_three_sightings() -> None:
    outputs = run(build(), [seen(Region(BOX))] * 3)
    assert [len(o.items) for o in outputs] == [0, 0, 1]
    (track,) = outputs[-1].items
    assert track.id == "cam-0"
    assert track.region.box == BOX
    assert 0 < track.confidence <= 1
    assert "speed_mean" in track.features
    assert outputs[-1].live_ids == {"cam-0"}


def test_priors_are_the_boxes_of_confirmed_tracks() -> None:
    outputs = run(build(), [seen(Region(BOX))] * 3)
    assert outputs[1].priors == ()
    (prior,) = outputs[-1].priors
    assert prior.box == BOX


def test_coasting_track_is_live_but_not_observed() -> None:
    config = TrackManagerOptions(max_miss_frames=2)
    sightings = [seen(Region(BOX))] * 3 + [seen(None)] * 4
    outputs = run(build(config), sightings)
    assert outputs[3].items == ()
    assert outputs[3].live_ids == {"cam-0"}
    assert outputs[-1].live_ids == frozenset()


def test_point_only_candidate_becomes_a_tiny_box() -> None:
    outputs = run(build(), [seen(Region(point=(20.0, 20.0)))] * 3)
    x0, y0, x1, y1 = outputs[-1].items[0].region.box
    assert (x1 - x0, y1 - y0) == (2, 2)


def test_ids_are_numbered_in_spawn_order_per_stream() -> None:
    far = Region((40.0, 30.0, 60.0, 46.0))
    outputs = run(build(), [seen(Region(BOX))] * 3 + [seen(far)] * 3)
    assert outputs[-1].live_ids == {"cam-0", "cam-1"}
    assert [t.id for t in outputs[-1].items] == ["cam-1"]


def test_ulid_ids_pass_through_when_not_deterministic() -> None:
    tracker = build(TrackManagerOptions(deterministic_ids=False))
    (track,) = run(tracker, [seen(Region(BOX))] * 3)[-1].items
    assert ULID.fullmatch(track.id)


def test_reset_restarts_the_numbering() -> None:
    tracker = build()
    sightings = [seen(Region(BOX))] * 3
    first = run(tracker, sightings)
    tracker.reset()
    assert run(tracker, sightings) == first


def test_sensor_size_comes_from_the_fact_unless_given() -> None:
    assert build().manager_config.sensor_width_px == 64
    given = TrackManagerOptions(sensor_width_px=100)
    assert build(given).manager_config.sensor_width_px == 100


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ConfigError):
        parse_config(TrackManagerOptions, {"max_miss": 3})


def test_passes_the_conformance_kit() -> None:
    assert check_method("trackmanager", facts={EXTENT: (128, 96)}) == []
