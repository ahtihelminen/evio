"""Rotor LLR arm: one-sided calibration, neutrality, fusion composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from evio.detection.kinematic_llr import KinematicLLR
from evio.detection.rotor_llr import RotorLLR, fit_rotor_llr
from evio.detection.verdict_gate import FusionGate, SequentialGate

pytestmark = pytest.mark.t1

if TYPE_CHECKING:
    from pathlib import Path

_THRESHOLD = 0.5


def _fit_separable() -> RotorLLR:
    rng = np.random.default_rng(0)
    drone = rng.uniform(0.5, 1.0, 200)
    clutter = rng.uniform(0.0, 0.1, 200)
    scores = np.concatenate([drone, clutter])
    labels = np.concatenate([np.ones(200), np.zeros(200)])
    return fit_rotor_llr(scores.tolist(), labels.tolist())


def test_fit_separable_orders_and_one_sided() -> None:
    model = _fit_separable()
    assert model.llr({"periodicity_score": 0.9}) > 1.0
    assert model.llr({"periodicity_score": 0.0}) == 0.0  # one-sided clamp
    assert model.llr({}) == 0.0                          # absent = neutral
    assert model.llr(None) == 0.0


def test_weight_scales_and_round_trip(tmp_path: Path) -> None:
    model = _fit_separable()
    path = tmp_path / "rotor.json"
    model.save(path)
    loaded = RotorLLR.load(path)
    f = {"periodicity_score": 0.8}
    assert abs(loaded.llr(f) - model.llr(f)) < 1e-9
    doubled = RotorLLR.load(path, weight=2.0)
    assert abs(doubled.llr(f) - 2.0 * model.llr(f)) < 1e-9


def test_non_finite_score_is_neutral() -> None:
    model = _fit_separable()
    assert model.llr({"periodicity_score": float("nan")}) == 0.0


def _zero_kinematic() -> KinematicLLR:
    """A kinematic arm that always contributes 0 (isolates the rotor arm)."""
    return KinematicLLR(
        feature_names=("speed_mean",),
        mean=np.array([0.0]), std=np.array([1.0]),
        coef=np.array([0.0]), intercept=0.0, prior_logit=0.0,
    )


def test_fusion_with_rotor_arm_speeds_firing() -> None:
    rotor = _fit_separable()
    fused = FusionGate(
        [_zero_kinematic(), rotor], SequentialGate(alpha=0.05, beta=0.1),
    )
    plain = FusionGate(
        [_zero_kinematic()], SequentialGate(alpha=0.05, beta=0.1),
    )
    with_rotor = {"speed_mean": 0.0, "periodicity_score": 0.95}
    without = {"speed_mean": 0.0}
    fired_with = fired_without = None
    for i in range(1, 12):
        if fired_with is None and fused.update(
                "cam", "t1", 0.65, _THRESHOLD, features=with_rotor):
            fired_with = i
        if fired_without is None and plain.update(
                "cam", "t2", 0.65, _THRESHOLD, features=without):
            fired_without = i
    assert fired_with is not None
    assert fired_without is None or fired_with < fired_without


def test_rotor_absence_never_hurts() -> None:
    """One-sided arm: a zero score must not slow firing vs no arm at all."""
    rotor = _fit_separable()
    fused = FusionGate(
        [_zero_kinematic(), rotor], SequentialGate(alpha=0.05, beta=0.1),
    )
    plain = FusionGate(
        [_zero_kinematic()], SequentialGate(alpha=0.05, beta=0.1),
    )
    feats = {"speed_mean": 0.0, "periodicity_score": 0.0}
    for prob in (0.8, 0.8, 0.8):
        a = fused.update("cam", "t1", prob, _THRESHOLD, features=feats)
        b = plain.update("cam", "t2", prob, _THRESHOLD,
                         features={"speed_mean": 0.0})
        assert a == b
