"""Tracking blob centres and ranking tracks by the evidence behind them."""

from __future__ import annotations

import pytest

from evio.dtc.conformance import check_method
from evio.dtc.facts import EXTENT
from evio.dtc.manifest import Stage
from evio.dtc.methods.blob_tracker import BlobTracker, BlobTrackerConfig
from evio.dtc.protocols import Context
from evio.dtc.registry import METHODS
from evio.dtc.types import Candidate, Candidates, Region, Tracks
from tests.support import dtc

pytestmark = pytest.mark.t1

STEP_US = 40_000
STREAM = "cam"


def blob(x: float, y: float, z: float = 15.0) -> Candidate:
    box = (x - 14, y - 14, x + 14, y + 14)
    return Candidate(STREAM, Region(box), z, {"z": z})


def flight(
    steps: int, start: float = 100.0, speed: float = 10.0
) -> list[list[Candidate]]:
    return [[blob(start + speed * i, 200.0)] for i in range(steps)]


class Rig:
    """A tracker fed one list of blobs per step."""

    def __init__(self, **config: float) -> None:
        context = Context(STREAM, (STREAM,), {EXTENT: (1280, 720)}, STEP_US)
        self.tracker = BlobTracker(BlobTrackerConfig(**config), context)
        self.index = 0

    def update(self, blobs: list[Candidate]) -> Tracks:
        start = self.index * STEP_US
        window = dtc.window(STREAM, dtc.empty_events(), start, start + STEP_US)
        step = dtc.step_of(self.index, STEP_US, window)
        self.index += 1
        return self.tracker.step(step, Candidates(tuple(blobs)))


def ids(tracks: Tracks) -> set[str]:
    return {track.id for track in tracks.items}


def test_registered_as_a_tracker() -> None:
    assert METHODS["blob-tracker"].manifest.stages == (Stage.TRACK,)


def test_a_blob_seen_step_after_step_becomes_one_confirmed_track() -> None:
    rig = Rig()
    outputs = [rig.update(blobs) for blobs in flight(12)]
    assert outputs[0].items == ()
    assert len({i for out in outputs[6:] for i in ids(out)}) == 1
    assert all(len(out.items) == 1 for out in outputs[6:])


def test_ids_are_prefixed_with_the_stream() -> None:
    rig = Rig()
    outputs = [rig.update(blobs) for blobs in flight(8)]
    assert ids(outputs[-1]) == {"cam-t1"}
    assert outputs[-1].live_ids == {"cam-t1"}
    assert outputs[-1].items[0].stream == STREAM


def test_the_track_box_follows_the_blob() -> None:
    rig = Rig()
    for blobs in flight(12):
        output = rig.update(blobs)
    cx, cy = output.items[0].region.center
    assert abs(cx - (100 + 10 * 11)) <= 8
    assert abs(cy - 200) <= 8


def test_blobs_that_never_repeat_never_confirm() -> None:
    rig = Rig()
    positions = [(100, 100), (900, 500), (300, 650), (1100, 80), (500, 300)]
    outputs = [rig.update([blob(x, y)]) for x, y in positions * 3]
    assert all(out.items == () for out in outputs)


def test_a_track_coasts_through_missed_steps_and_keeps_its_id() -> None:
    rig = Rig()
    seen = set()
    for i, blobs in enumerate(flight(16)):
        output = rig.update([] if i in (9, 10) else blobs)
        seen |= ids(output)
        if i == 9:
            assert output.live_ids == {"cam-t1"}
    assert seen == {"cam-t1"}


def test_a_track_that_is_lost_for_good_is_dropped() -> None:
    rig = Rig()
    for blobs in flight(10):
        rig.update(blobs)
    outputs = [rig.update([]) for _ in range(12)]
    assert outputs[-1].items == ()
    assert outputs[-1].live_ids == frozenset()


def test_two_separate_targets_get_two_ids() -> None:
    rig = Rig()
    for i in range(12):
        output = rig.update([blob(100 + 10 * i, 200), blob(900 - 10 * i, 500)])
    assert len(ids(output)) == 2


def test_the_better_supported_track_comes_first() -> None:
    rig = Rig()
    for i in range(12):
        output = rig.update(
            [blob(100 + 10 * i, 200, z=9.0), blob(900 - 10 * i, 500, z=25.0)]
        )
    first, second = output.items
    assert first.region.center[1] > 400
    assert first.confidence > second.confidence


def test_two_tracks_on_one_drone_become_one_the_older() -> None:
    rig = Rig()
    for blobs in flight(8):
        rig.update(blobs)
    for i in range(8, 20):
        x = 100.0 + 10 * i
        output = rig.update([blob(x, 200.0), blob(x + 22, 200.0)])
    assert ids(output) == {"cam-t1"}


def test_a_drone_that_vanishes_and_returns_keeps_its_id() -> None:
    rig = Rig()
    for blobs in flight(10):
        rig.update(blobs)
    for _ in range(20):
        rig.update([])
    for i in range(8):
        output = rig.update([blob(300.0 + 10 * i, 200.0)])
    assert ids(output) == {"cam-t1"}


def test_a_blob_that_steals_a_lost_id_gives_it_back_when_it_dies() -> None:
    rig = Rig()
    for blobs in flight(10):
        rig.update(blobs)
    for _ in range(15):
        rig.update([])
    rig.update([blob(500.0, 400.0)])  # clutter, seen once
    for _ in range(12):
        rig.update([])
    for i in range(8):
        output = rig.update([blob(300.0 + 10 * i, 200.0)])
    assert ids(output) == {"cam-t1"}


def test_priors_are_the_predicted_boxes_of_live_tracks() -> None:
    rig = Rig()
    for blobs in flight(8):
        output = rig.update(blobs)
    (track,) = output.items
    (prior,) = output.priors
    assert prior.center[0] > track.region.center[0]
    assert prior.center[1] == pytest.approx(track.region.center[1], abs=1)


def test_a_steady_flight_has_straight_steady_features_in_px_per_s() -> None:
    rig = Rig()
    for blobs in flight(14):
        output = rig.update(blobs)
    features = output.items[0].features
    assert features["straightness"] == pytest.approx(1.0, abs=0.01)
    assert features["speed_mean"] == pytest.approx(250.0, rel=0.1)
    assert features["speed_cv"] < 0.2
    assert features["vel_mag"] == pytest.approx(250.0, rel=0.1)
    assert features["age_frames"] == 14
    assert features["miss_count"] == 0
    assert features["bbox_diag"] == pytest.approx(28 * 2**0.5, rel=0.05)
    assert output.items[0].region.velocity_px_s[0] == features["vel_mag"]


def test_a_hovering_blob_has_no_speed() -> None:
    rig = Rig()
    for _ in range(10):
        output = rig.update([blob(300.0, 300.0)])
    assert output.items[0].features["speed_mean"] == pytest.approx(0.0)
    assert output.items[0].features["straightness"] == 0.0


def test_the_centre_history_is_bounded_by_the_config() -> None:
    rig = Rig(history_len=3)
    for blobs in flight(10):
        rig.update(blobs)
    (live,) = rig.tracker._tracks
    assert len(live.path) == 3


def test_reset_forgets_every_track() -> None:
    rig = Rig()
    for blobs in flight(8):
        rig.update(blobs)
    rig.tracker.reset()
    assert rig.update(flight(1)[0]).live_ids == {"cam-t1"}


def test_conforms_on_a_small_extent() -> None:
    assert check_method("blob-tracker", facts={EXTENT: (128, 96)}) == []
