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
