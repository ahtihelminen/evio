"""Fidelity for the detection chain's own doubles (split from SN-577).

`FakeDet` and `FakeStatefulBundle` stand in for pieces of evio's detection
chain. Each double gets one positive test pinning the property it must
share with the real thing, and one negative-control test: a deliberately
broken variant, and the assertion that would have caught it inverted to
prove it still discriminates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from fakes.kinematic_llr import FakeDet
from fakes.track_classifier import FakeStatefulBundle

from evio.detection.kinematic_llr import assemble_features
from evio.detection.live_camera import SEQ_LEN
from evio.detection.types import Detection

pytestmark = pytest.mark.t1


def _detection(**overrides: object) -> Detection:
    fields = {
        "x0": 10, "y0": 10, "x1": 14, "y1": 14,
        "vx_px_s": 1.0, "vy_px_s": 1.0, "m_coh": 5.0, "best_psr": 8.0,
        "event_count": 20, "bg_class": 0,
        "row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1,
    }
    fields.update(overrides)
    return Detection(**fields)


# --- FakeDet -----------------------------------------------------------


def test_fake_det_matches_a_real_detection_on_assemble_features() -> None:
    """`assemble_features` reads exactly five `Detection` fields (`m_coh`,
    `best_psr`, `event_count`, `z_score`, `periodicity_score`). `FakeDet`
    is a separate dataclass with the same field names and defaults --
    nothing keeps the two in sync if `Detection` changes. Run both
    through `assemble_features` with matching values, including the
    documented "None = not evaluated" edge case for `periodicity_score`
    (`evio/detection/types.py`).
    """
    fake = FakeDet()
    real = _detection(
        m_coh=fake.m_coh, best_psr=fake.best_psr,
        event_count=fake.event_count, z_score=fake.z_score,
        periodicity_score=fake.periodicity_score,
    )
    assert assemble_features(fake, None) == assemble_features(real, None)
    assert "periodicity_score" not in assemble_features(real, None)

    fake_scored = FakeDet(periodicity_score=0.8)
    real_scored = _detection(
        m_coh=fake_scored.m_coh, best_psr=fake_scored.best_psr,
        event_count=fake_scored.event_count, z_score=fake_scored.z_score,
        periodicity_score=fake_scored.periodicity_score,
    )
    assert (
        assemble_features(fake_scored, None)
        == assemble_features(real_scored, None)
    )


def test_a_fake_det_defaulting_periodicity_to_zero_would_fail_this() -> None:
    """The deliberate break the issue names: skip the documented
    None-vs-evaluated edge case by defaulting to 0.0 instead of `None`."""
    @dataclass
    class BrokenFakeDet:
        m_coh: float = 0.4
        best_psr: float = 5.0
        event_count: int = 20
        z_score: float = 2.5
        periodicity_score: float | None = 0.0  # bug: should default None

    broken = BrokenFakeDet()
    real = _detection(
        m_coh=broken.m_coh, best_psr=broken.best_psr,
        event_count=broken.event_count, z_score=broken.z_score,
        periodicity_score=None,
    )
    assert assemble_features(broken, None) != assemble_features(real, None)


# --- FakeStatefulBundle -----------------------------------------------------


def test_fake_stateful_bundle_preserves_h_track_shape_against_h_in() -> None:
    """R5 threading contract (`evio/detection/temporal_transformer_v3.py`,
    `evio/detection/openvino_transformer.py`): `h_track` rides out at the
    same (tracks, state_dim) shape `h_in` rode in at -- the classifier's
    `_score_stateful` indexes `h_out[i]` per input track
    (`evio/detection/track_classifier.py`). Running the real OpenVINO or
    torch bundle needs a compiled IR or a checkpoint, neither practical
    for a T1 test, so this pins the documented shape contract instead.
    """
    n, state_dim = 3, FakeStatefulBundle._state_dim
    patches = np.zeros((n, SEQ_LEN, 32, 32, 3), dtype=np.uint8)
    h_in = np.zeros((n, state_dim), dtype=np.float32)

    out = FakeStatefulBundle().predict(patches, None, None, h_in)

    assert out["h_track"].shape == h_in.shape
    assert out["drone_logits"].shape[0] == n


def test_a_bundle_dropping_a_track_from_h_track_would_fail_this() -> None:
    """The deliberate break: return one fewer row than tracks fed in, as
    if a track were silently dropped from the state batch."""
    n, state_dim = 3, FakeStatefulBundle._state_dim
    patches = np.zeros((n, SEQ_LEN, 32, 32, 3), dtype=np.uint8)
    h_in = np.zeros((n, state_dim), dtype=np.float32)

    class DroppingBundle(FakeStatefulBundle):
        def predict(
            self, patches: np.ndarray, events: object, mask: object,
            h_in: np.ndarray,
        ) -> dict[str, np.ndarray]:
            out = super().predict(patches, events, mask, h_in)
            out["h_track"] = out["h_track"][:-1]
            return out

    out = DroppingBundle().predict(patches, None, None, h_in)
    assert out["h_track"].shape != h_in.shape
