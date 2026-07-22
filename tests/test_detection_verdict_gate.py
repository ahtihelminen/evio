"""SPRT SequentialGate (0a): accumulation, latching, non-fatal dropout, KILL.

The headline property vs the static M-of-K gate is that a single weak/dropped
window does not delete a confirmed track — the gate latches DRONE and only a
decisive run of contrary evidence (S ≤ B) kills it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from evio.detection.kinematic_llr import KinematicLLR
from evio.detection.verdict_gate import (
    FusionGate,
    PersistenceGate,
    SequentialGate,
    build_verdict_gate,
    prob_to_log_odds,
)

if TYPE_CHECKING:
    from pathlib import Path

_THRESHOLD = 0.5  # logit(0.5) = 0, so increments are just logit(prob)/T


def test_prob_to_log_odds_basics() -> None:
    assert abs(prob_to_log_odds(0.5)) < 1e-9
    assert prob_to_log_odds(0.9) > 0.0
    assert prob_to_log_odds(0.1) < 0.0
    # Temperature scales the log-odds linearly.
    assert abs(
        prob_to_log_odds(0.9, temperature=2.0) * 2.0 - prob_to_log_odds(0.9),
    ) < 1e-9


def test_accumulates_to_fire() -> None:
    # prob 0.8 → logit ≈ 1.386 per window; A=log(18)≈2.89, so it takes 3.
    gate = SequentialGate(alpha=0.05, beta=0.1)
    assert gate.update("cam", "t1", 0.8, _THRESHOLD) is False
    assert gate.update("cam", "t1", 0.8, _THRESHOLD) is False
    assert gate.update("cam", "t1", 0.8, _THRESHOLD) is True


def test_strong_single_window_fires() -> None:
    # prob 0.95 → logit ≈ 2.944 > A in one window.
    gate = SequentialGate(alpha=0.05, beta=0.1)
    assert gate.update("cam", "t1", 0.95, _THRESHOLD) is True


def test_latched_drone_survives_one_weak_window() -> None:
    gate = SequentialGate(alpha=0.05, beta=0.1)
    for _ in range(3):
        gate.update("cam", "t1", 0.8, _THRESHOLD)
    # A single weak (below-threshold) window must NOT un-declare the track —
    # this is exactly what the static M-of-K gate fails to guarantee.
    assert gate.update("cam", "t1", 0.45, _THRESHOLD) is True


def test_sustained_low_kills_and_resets() -> None:
    gate = SequentialGate(alpha=0.05, beta=0.1)
    assert gate.update("cam", "t1", 0.2, _THRESHOLD) is False
    assert gate.update("cam", "t1", 0.2, _THRESHOLD) is False  # S ≤ B → KILL
    # After KILL the accumulator is reset, so one moderate window can't fire.
    assert gate.update("cam", "t1", 0.8, _THRESHOLD) is False


def test_prune_drops_track_state() -> None:
    gate = SequentialGate(alpha=0.05, beta=0.1)
    assert gate.update("cam", "t1", 0.95, _THRESHOLD) is True  # latched
    gate.prune("cam", set())  # t1 no longer alive
    # Fresh state: a single weak window neither fires nor is latched.
    assert gate.update("cam", "t1", 0.45, _THRESHOLD) is False


def test_thresholds_match_wald() -> None:
    import math
    gate = SequentialGate(alpha=0.05, beta=0.1)
    assert abs(gate.upper - math.log(0.9 / 0.05)) < 1e-12
    assert abs(gate.lower - math.log(0.1 / 0.95)) < 1e-12


def test_factory_dispatch() -> None:
    assert isinstance(build_verdict_gate("mofk", k=4, m=2), PersistenceGate)
    assert isinstance(build_verdict_gate("sprt"), SequentialGate)
    with pytest.raises(ValueError, match="unknown verdict gate"):
        build_verdict_gate("nope")


def test_factory_sprt_is_drop_in() -> None:
    # Same call-site contract as PersistenceGate: update(...)->bool, prune(...).
    gate = build_verdict_gate("sprt")
    assert gate.update("cam", "t1", 0.95, _THRESHOLD) is True
    gate.prune("cam", {"t1"})  # live id kept → no error, state retained


def _one_feature_llr(coef: float) -> KinematicLLR:
    """A trivial 1-feature model where llr({"f": v}) == coef * v."""
    return KinematicLLR(
        feature_names=("f",),
        mean=np.array([0.0]), std=np.array([1.0]),
        coef=np.array([coef]), intercept=0.0, prior_logit=0.0,
    )


def test_sprt_extra_llr_is_additive() -> None:
    base = SequentialGate(alpha=0.05, beta=0.1)
    boosted = SequentialGate(alpha=0.05, beta=0.1)
    assert base.update("cam", "t1", 0.6, _THRESHOLD) is False
    # +2.0 of extra evidence on top of logit(0.6)≈0.405 clears A≈2.89.
    assert boosted.update("cam", "t1", 0.6, _THRESHOLD, extra_llr=2.5) is True


def test_fusion_drone_like_motion_speeds_firing() -> None:
    plain = SequentialGate(alpha=0.05, beta=0.1)
    fused = FusionGate([_one_feature_llr(1.0)], SequentialGate(
        alpha=0.05, beta=0.1))
    feats = {"f": 3.0}  # +3.0 kinematic LLR per window
    # logit(0.7)≈0.85 alone stays under A≈2.89; +3.0 kinematic clears it.
    assert plain.update("cam", "t1", 0.7, _THRESHOLD) is False
    assert fused.update("cam", "t1", 0.7, _THRESHOLD, features=feats) is True


def test_fusion_clutter_motion_delays_firing() -> None:
    fused = FusionGate([_one_feature_llr(1.0)], SequentialGate(
        alpha=0.05, beta=0.1))
    # Strongly clutter-like motion (-3.0) drags a moderate window negative.
    assert fused.update(
        "cam", "t1", 0.7, _THRESHOLD, features={"f": -3.0},
    ) is False


def test_fusion_without_features_matches_sprt() -> None:
    plain = SequentialGate(alpha=0.05, beta=0.1)
    fused = FusionGate([_one_feature_llr(5.0)], SequentialGate(
        alpha=0.05, beta=0.1))
    for prob in (0.8, 0.8, 0.8):
        a = plain.update("cam", "t1", prob, _THRESHOLD)
        b = fused.update("cam", "t2", prob, _THRESHOLD)  # no features
        assert a == b


def test_fusion_kill_authority_preserved() -> None:
    # Even with a positive kinematic bias, sustained low appearance KILLs.
    fused = FusionGate([_one_feature_llr(1.0)], SequentialGate(
        alpha=0.05, beta=0.1))
    feats = {"f": 0.5}
    assert fused.update("cam", "t1", 0.05, _THRESHOLD, features=feats) is False
    assert fused.update("cam", "t1", 0.05, _THRESHOLD, features=feats) is False
    # After KILL a single moderate window can't immediately re-fire.
    assert fused.update("cam", "t1", 0.7, _THRESHOLD, features=feats) is False


def test_factory_fusion_requires_path_then_loads(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires kinematic_llr_path"):
        build_verdict_gate("fusion")
    model_path = tmp_path / "kin.json"
    _one_feature_llr(1.0).save(model_path)
    gate = build_verdict_gate("fusion", kinematic_llr_path=model_path)
    assert isinstance(gate, FusionGate)
    # Drop-in contract: update(...)->bool, prune(...).
    assert gate.update("cam", "t1", 0.95, _THRESHOLD, features={"f": 1.0})
    gate.prune("cam", {"t1"})


def test_fusion_clamp_bounds_arm_evidence() -> None:
    """A transient hugely-negative arm must not KILL a latched track."""
    unclamped = FusionGate([_one_feature_llr(1.0)], SequentialGate(
        alpha=0.05, beta=0.1))
    clamped = FusionGate([_one_feature_llr(1.0)], SequentialGate(
        alpha=0.05, beta=0.1), clamp=2.0)
    for gate in (unclamped, clamped):
        assert gate.update("cam", "t1", 0.95, _THRESHOLD,
                           features={"f": 0.0}) is True  # latched
    sour = {"f": -8.0}  # one bad kinematic window
    assert unclamped.update("cam", "t1", 0.6, _THRESHOLD,
                            features=sour) is False  # dragged to KILL
    assert clamped.update("cam", "t1", 0.6, _THRESHOLD,
                          features=sour) is True    # latch survives


def test_fusion_clamp_is_negative_only() -> None:
    """Positive arm evidence stays unbounded — it carries weak-appearance
    drones over the line (2026-07-03 real-footage finding)."""
    gate = FusionGate([_one_feature_llr(1.0)], SequentialGate(
        alpha=0.05, beta=0.1), clamp=2.0)
    # Huge positive evidence fires in ONE window even at modest prob.
    assert gate.update("cam", "t1", 0.6, _THRESHOLD,
                       features={"f": 10.0}) is True
