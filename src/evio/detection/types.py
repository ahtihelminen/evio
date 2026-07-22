"""Domain types for detections.

`Detection` is the single-window primitive produced by any detection algorithm
(currently TBD; future algorithms or other camera types will produce the same
type). `Classification` is the optional identity-of-the-thing-detected, added
by a downstream classifier stage when the input supports it (e.g. mono-camera
patches; not all detections will carry one).

These types intentionally live above any specific algorithm so that consumers
(classifier, SAPIENT adapter, UI overlay, evaluation) don't need to import
from a TBD-internal module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class Classification:
    """An optional identity label attached to a Detection."""

    label: str
    confidence: float


@dataclass(frozen=True, slots=True)
class Detection:
    """A validated detection with motion coherence."""

    x0: int
    y0: int
    x1: int
    y1: int
    vx_px_s: float
    vy_px_s: float
    m_coh: float
    best_psr: float
    event_count: int
    bg_class: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    z_score: float = 0.0
    timestamp_us: int = 0
    camera_id: str = ""
    classification: Classification | None = None
    range_m: float | None = None
    track_id: str = ""
    # Fixed-size event-density crop inside the bbox, used by the Re-ID path
    # in `evio.detection.reid` to match an occluded-then-reappearing drone
    # back to its dormant track. None means caller didn't extract one;
    # tracker falls back to legacy spawn behaviour.
    patch: np.ndarray | None = None
    # Learned L2-normed ReID embedding (v3/v4 transformer head). When the
    # tracker runs with `reid_use_embedding`, this replaces the raw-patch
    # signature for appearance matching; None falls back to `patch` (a track
    # is blind to its own embedding until its classify ring fills).
    embedding: np.ndarray | None = None
    # Sub-pixel center. When populated, the tracker uses these instead of
    # the integer-bbox midpoint, enabling detectors that localise finer
    # than one pixel (Gaussian-fit, CenterNet offset, IWE quadratic fit).
    cx: float | None = None
    cy: float | None = None
    # Soft R1 confidence in [0, 1] (0b R1 hardening): sigmoid((m_coh-c)/s).
    # When populated, the consumer-side keep-gate (`evio.detection.r1_gate`)
    # keeps a weak candidate that overlaps a live track's predicted ROI
    # instead of R1 hard-dropping it, so one dropout window is non-fatal.
    cand_conf: float | None = None
    # Rotor-periodicity evidence (`evio.detection.rotor_periodicity`, SN-31):
    # blade-pass fundamental + comb score in [0, 1). None = not evaluated
    # (estimator off/blanked/gates unmet); 0.0 = evaluated, nothing found.
    # One-sided evidence — rides the fusion LLR bus, never m_coh.
    rotor_hz: float | None = None
    periodicity_score: float | None = None
    # IRSTD local-contrast score (`evio.detection.irstd` MPCM on the event
    # patch): diagnostic scalar for logs/dashboard; the LLR arm consumes the
    # full feature dict computed at the fusion seam.
    irstd_score: float | None = None
