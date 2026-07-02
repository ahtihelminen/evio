"""Kinematic LLR fusion model: fit, calibration sign, persistence, assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from evio.detection.kinematic_llr import (
    KINEMATIC_FEATURES,
    KinematicLLR,
    assemble_features,
    fit_kinematic_llr,
)

_NAMES = ("a", "b")


def _make_dataset(n: int = 200) -> tuple[list[dict[str, float]], list[float]]:
    """Two features where high 'a' ⇒ drone; 'b' is pure noise."""
    rng = np.random.default_rng(0)
    rows: list[dict[str, float]] = []
    labels: list[float] = []
    for i in range(n):
        drone = i % 2 == 0
        a = rng.normal(3.0 if drone else -3.0, 1.0)
        b = rng.normal(0.0, 1.0)
        rows.append({"a": float(a), "b": float(b)})
        labels.append(1.0 if drone else 0.0)
    return rows, labels


def test_fit_orders_llr_by_class() -> None:
    rows, labels = _make_dataset()
    model = fit_kinematic_llr(rows, labels, feature_names=_NAMES)
    drone_like = model.llr({"a": 3.0, "b": 0.0})
    clutter_like = model.llr({"a": -3.0, "b": 0.0})
    assert drone_like > 0.0 > clutter_like
    assert drone_like > clutter_like


def test_noise_feature_barely_moves_llr() -> None:
    rows, labels = _make_dataset()
    model = fit_kinematic_llr(rows, labels, feature_names=_NAMES)
    # 'b' carries no class information, so sweeping it shifts llr far less
    # than sweeping the informative 'a'.
    span_b = abs(model.llr({"a": 0.0, "b": 3.0})
                 - model.llr({"a": 0.0, "b": -3.0}))
    span_a = abs(model.llr({"a": 3.0, "b": 0.0})
                 - model.llr({"a": -3.0, "b": 0.0}))
    assert span_a > span_b * 3.0


def test_missing_feature_falls_back_to_mean() -> None:
    rows, labels = _make_dataset()
    model = fit_kinematic_llr(rows, labels, feature_names=_NAMES)
    # Omitting 'b' must equal supplying b at its training mean (neutral).
    with_mean = model.llr({"a": 1.0, "b": float(model.mean[1])})
    without_b = model.llr({"a": 1.0})
    assert abs(with_mean - without_b) < 1e-9


def test_non_finite_feature_is_neutralised() -> None:
    rows, labels = _make_dataset()
    model = fit_kinematic_llr(rows, labels, feature_names=_NAMES)
    finite = model.llr({"a": float(model.mean[0]), "b": 0.0})
    with_inf = model.llr({"a": float("inf"), "b": 0.0})
    assert abs(finite - with_inf) < 1e-9


def test_weight_scales_llr_linearly() -> None:
    rows, labels = _make_dataset()
    model = fit_kinematic_llr(rows, labels, feature_names=_NAMES, weight=1.0)
    heavy = fit_kinematic_llr(rows, labels, feature_names=_NAMES, weight=2.0)
    f = {"a": 2.0, "b": 0.5}
    assert abs(heavy.llr(f) - 2.0 * model.llr(f)) < 1e-9


def test_save_load_round_trip(tmp_path: Path) -> None:
    rows, labels = _make_dataset()
    model = fit_kinematic_llr(rows, labels, feature_names=_NAMES, weight=1.5)
    path = tmp_path / "kin.json"
    model.save(path)
    loaded = KinematicLLR.load(path)
    f = {"a": 1.7, "b": -0.3}
    assert abs(loaded.llr(f) - model.llr(f)) < 1e-9
    assert loaded.feature_names == _NAMES
    # weight override at load time.
    reweighted = KinematicLLR.load(path, weight=3.0)
    assert abs(reweighted.llr(f) - 2.0 * model.llr(f)) < 1e-9


def test_fit_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        fit_kinematic_llr([], [], feature_names=_NAMES)


@dataclass
class _FakeDet:
    m_coh: float = 0.4
    best_psr: float = 5.0
    event_count: int = 20
    z_score: float = 2.5
    periodicity_score: float | None = None


def test_assemble_features_merges_detection_and_track() -> None:
    track_feats = {"speed_mean": 120.0, "straightness": 0.9}
    feats = assemble_features(_FakeDet(), track_feats)
    assert feats["speed_mean"] == 120.0
    assert feats["straightness"] == 0.9
    assert feats["m_coh"] == 0.4
    assert feats["best_psr"] == 5.0
    assert feats["z_score"] == 2.5
    assert feats["log_event_count"] == pytest.approx(np.log1p(20))


def test_assemble_features_tolerates_missing_track() -> None:
    feats = assemble_features(_FakeDet(), None)
    # Detection-side keys always present; track-side simply absent.
    assert "m_coh" in feats
    assert "speed_mean" not in feats
    assert "periodicity_score" not in feats  # None = not evaluated


def test_assemble_features_carries_rotor_score() -> None:
    feats = assemble_features(_FakeDet(periodicity_score=0.8), None)
    assert feats["periodicity_score"] == 0.8


def test_default_feature_contract_is_stable() -> None:
    # The live gate and the offline trainer share this ordered contract.
    assert KINEMATIC_FEATURES[0] == "speed_mean"
    assert "m_coh" in KINEMATIC_FEATURES
    assert len(set(KINEMATIC_FEATURES)) == len(KINEMATIC_FEATURES)
