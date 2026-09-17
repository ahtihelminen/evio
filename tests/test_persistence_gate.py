"""K-of-M persistence gate for per-track drone verdicts."""

from __future__ import annotations

import pytest

from evio.detection.verdict_gate import PersistenceGate

pytestmark = pytest.mark.t1

THRESH = 0.5


def test_k1_m1_is_single_window_threshold() -> None:
    gate = PersistenceGate(k=1, m=1)
    assert gate.update("cam", "t1", 0.9, THRESH) is True
    assert gate.update("cam", "t1", 0.1, THRESH) is False


def test_k4_m2_needs_two_of_last_four() -> None:
    gate = PersistenceGate(k=4, m=2)
    # One high window is not enough.
    assert gate.update("cam", "t1", 0.99, THRESH) is False
    assert gate.update("cam", "t1", 0.10, THRESH) is False
    # Second crossing within the window fires the gate.
    assert gate.update("cam", "t1", 0.80, THRESH) is True


def test_ring_slides_so_old_crossings_age_out() -> None:
    gate = PersistenceGate(k=3, m=2)
    gate.update("cam", "t1", 0.9, THRESH)  # [0.9]
    gate.update("cam", "t1", 0.9, THRESH)  # [0.9, 0.9] -> fires
    # Three low windows push both highs out of the maxlen-3 ring.
    assert gate.update("cam", "t1", 0.1, THRESH) is True  # [0.9,0.9,0.1]
    assert gate.update("cam", "t1", 0.1, THRESH) is False  # [0.9,0.1,0.1]
    assert gate.update("cam", "t1", 0.1, THRESH) is False  # [0.1,0.1,0.1]


def test_m_clamped_to_k_and_floors_at_one() -> None:
    gate = PersistenceGate(k=2, m=9)
    assert gate.k == 2
    assert gate.m == 2  # clamped to k
    floored = PersistenceGate(k=0, m=0)
    assert floored.k == 1
    assert floored.m == 1


def test_threshold_is_applied_per_call() -> None:
    gate = PersistenceGate(k=1, m=1)
    assert gate.update("cam", "t1", 0.6, 0.5) is True
    assert gate.update("cam", "t1", 0.6, 0.9) is False


def test_serial_and_track_isolation() -> None:
    gate = PersistenceGate(k=2, m=2)
    gate.update("camA", "t1", 0.9, THRESH)
    # Same track id on a different camera shares no history.
    assert gate.update("camB", "t1", 0.9, THRESH) is False
    # camA's own second crossing still fires independently.
    assert gate.update("camA", "t1", 0.9, THRESH) is True


def test_prune_drops_dead_tracks_only() -> None:
    gate = PersistenceGate(k=2, m=1)
    gate.update("cam", "t1", 0.9, THRESH)
    gate.update("cam", "t2", 0.9, THRESH)
    gate.update("other", "t1", 0.9, THRESH)
    gate.prune("cam", {"t2"})
    # t1 on cam was pruned -> its ring restarts empty (one crossing needed).
    assert ("cam", "t1") not in gate._rings
    # t2 on cam survived; other-serial t1 untouched by a cam prune.
    assert ("cam", "t2") in gate._rings
    assert ("other", "t1") in gate._rings
