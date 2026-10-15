"""The kinematic classifier turns motion features into drone probabilities."""

from __future__ import annotations

import pytest

from evio.dtc.conformance import check_method
from evio.dtc.facts import EXTENT
from evio.dtc.manifest import Stage
from evio.dtc.methods.kinematic import Kinematic, KinematicConfig
from evio.dtc.protocols import Context
from evio.dtc.registry import METHODS
from evio.dtc.types import Region, Track, Tracks
from tests.support import dtc

pytestmark = pytest.mark.t1

STEP = dtc.step_of(0, 10_000, dtc.window("cam", dtc.empty_events(), 0, 10_000))
DRONE = {
    "speed_mean": 300.0,
    "straightness": 0.95,
    "speed_cv": 0.1,
    "age_frames": 40.0,
}
CLUTTER = {
    "speed_mean": 5.0,
    "straightness": 0.1,
    "speed_cv": 1.5,
    "age_frames": 3.0,
}


def classify(features: dict[str, dict[str, float]]) -> dict[str, float]:
    context = Context("cam", ("cam",), {EXTENT: (64, 48)}, 10_000)
    classifier = Kinematic(KinematicConfig(), context)
    tracks = Tracks(
        tuple(
            Track("cam", name, Region(point=(1.0, 1.0)), 1.0)
            for name in features
        )
    )
    verdicts = classifier.step(STEP, tracks, {"kinematics": features})
    return {v.track_id: v.probability for v in verdicts.items}


def test_registered_as_classify_with_kinematics_evidence() -> None:
    manifest = METHODS["kinematic"].manifest
    assert manifest.stages == (Stage.CLASSIFY,)
    assert manifest.evidence == ("kinematics",)


def test_steady_straight_old_track_scores_drone() -> None:
    assert classify({"t": DRONE})["t"] > 0.9


def test_erratic_short_slow_track_scores_no_drone() -> None:
    assert classify({"t": CLUTTER})["t"] < 0.1


def test_missing_feature_is_neutral() -> None:
    partial = {"speed_mean": 300.0}
    assert classify({"a": partial, "b": {}}) == {"a": 0.5, "b": 0.5}


def test_track_without_evidence_is_neutral() -> None:
    context = Context("cam", ("cam",), {EXTENT: (64, 48)}, 10_000)
    track = Track("cam", "t", Region(point=(1.0, 1.0)), 1.0)
    verdicts = Kinematic(KinematicConfig(), context).step(
        STEP, Tracks((track,)), {}
    )
    assert [v.probability for v in verdicts.items] == [0.5]


def test_weights_come_from_config() -> None:
    context = Context("cam", ("cam",), {EXTENT: (64, 48)}, 10_000)
    track = Track("cam", "t", Region(point=(1.0, 1.0)), 1.0)
    flat = KinematicConfig(
        bias=0.0,
        speed_weight=0.0,
        straightness_weight=0.0,
        speed_cv_weight=0.0,
        age_weight=0.0,
    )
    verdicts = Kinematic(flat, context).step(
        STEP, Tracks((track,)), {"kinematics": {"t": DRONE}}
    )
    assert verdicts.items[0].probability == 0.5


def test_extreme_features_saturate_instead_of_overflowing() -> None:
    erratic = CLUTTER | {"speed_cv": 1e6}
    assert classify({"t": erratic})["t"] == 0.0


def test_passes_the_conformance_kit() -> None:
    assert check_method("kinematic", facts={EXTENT: (128, 96)}) == []
