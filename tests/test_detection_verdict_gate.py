"""SPRT SequentialGate (0a): accumulation, latching, non-fatal dropout, KILL.

The headline property vs the static M-of-K gate is that a single weak/dropped
window does not delete a confirmed track — the gate latches DRONE and only a
decisive run of contrary evidence (S ≤ B) kills it.
"""

from __future__ import annotations

import pytest

from evio.detection.verdict_gate import (
    PersistenceGate,
    SequentialGate,
    build_verdict_gate,
    prob_to_log_odds,
)

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
