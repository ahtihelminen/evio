"""Classify-cadence: re-score a mature track at most once per N windows.

The transformer is mocked (no checkpoint) so these pin only the cadence
gating in ``TrackTransformerClassifier`` — the E0 Sub-gate A throughput fix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.detection import track_classifier as tc
from evio.detection.live_camera import SEQ_LEN
from evio.detection.track_classifier import TrackTransformerClassifier
from evio.detection.types import Detection


def _det(track_id: str) -> Detection:
    return Detection(
        x0=10, y0=10, x1=20, y1=20, vx_px_s=0.0, vy_px_s=0.0,
        m_coh=0.8, best_psr=1.0, event_count=100, bg_class=0,
        row_start=0, row_end=1, col_start=0, col_end=1,
        track_id=track_id, patch=np.zeros((32, 32, 3), dtype=np.uint8),
    )


def _batch() -> EventsBatch:
    return EventsBatch(
        timestamps=np.array([0, 10_000], dtype=np.int64),
        x_coords=np.array([1, 2], dtype=np.int16),
        y_coords=np.array([1, 2], dtype=np.int16),
        polarities=np.array([1, 0], dtype=np.int8),
    )


def _clf(
    cadence: int, monkeypatch: pytest.MonkeyPatch,
) -> tuple[TrackTransformerClassifier, dict[str, int]]:
    clf = TrackTransformerClassifier(
        Path("/nonexistent.pt"), classify_cadence=cadence,
    )
    # Force a v2-style loaded model without a checkpoint, and count scores.
    clf._bundle = object()
    clf._is_v3 = False
    clf._is_v4 = False
    calls = {"n": 0}

    def _fake(
        bundle: object, patches: np.ndarray,
        events: object = None, mask: object = None, *, is_v4: bool = False,
    ) -> np.ndarray:
        calls["n"] += 1
        return np.full(patches.shape[0], 0.9, dtype=np.float32)

    monkeypatch.setattr(tc, "predict_drone_probs", _fake)
    return clf, calls


def _run(clf: TrackTransformerClassifier, n_windows: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for _ in range(n_windows):
        out = clf.classify("cam", [_det("a")], _batch())
    return out


def test_cadence_1_scores_every_ready_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clf, calls = _clf(1, monkeypatch)
    out = _run(clf, SEQ_LEN + 5)  # ready on windows SEQ_LEN..SEQ_LEN+5 = 6
    assert calls["n"] == 6
    assert out["a"] == pytest.approx(0.9)  # verdict present


def test_cadence_n_divides_the_load(monkeypatch: pytest.MonkeyPatch) -> None:
    clf, calls = _clf(5, monkeypatch)
    out = _run(clf, SEQ_LEN + 10)  # 11 ready windows; scored at +0,+5,+10
    assert calls["n"] == 3
    # The verdict is still returned every window (sticky between re-scores).
    assert out["a"] == pytest.approx(0.9)


def test_cadence_never_scores_before_ring_fills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clf, calls = _clf(1, monkeypatch)
    _run(clf, SEQ_LEN - 1)  # ring not full yet
    assert calls["n"] == 0
