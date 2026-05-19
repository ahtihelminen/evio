"""Signature-based re-identification helpers for the Kalman tracker.

When a drone hides behind a building, into the sun, or behind the mast,
its Kalman track times out and the next sighting would normally spawn a
fresh `object_id`. This module exposes the primitives the tracker uses
to recognise the drone by its bbox appearance instead and reuse the
original track.

Three steps make a signature:

1. `extract_event_patch` clips the event burst to the detection's bbox,
   renders the polarity-up / polarity-down event-density frame, and
   resizes to a fixed (target_size) BGR uint8 patch.
2. `normalize_patch` converts that patch to a unit-norm grayscale
   feature vector.
3. Cosine similarity (`cosine_similarity`) compares two feature vectors;
   1.0 means identical, near 0 means orthogonal.

`SignatureBuffer` is a bounded queue of normalized patches per track
(`.signature()` returns the mean, renormalised). The tracker stores
this on every confirmed track, bakes it into the dormant pool on
eviction, and dot-products incoming detections against it.

A known limitation: two visually identical drones at a crossing moment
will swap IDs because no appearance method can resolve the ambiguity.
Trajectory continuity (the Kalman tracker) handles the spatially
separated case; appearance handles the occlusion case; the crossing
case is unavoidable.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from evio.rendering.frame import get_frame

if TYPE_CHECKING:
    from evio.core.events import EventsBatch

PatchArray = NDArray[np.uint8]
FeatureVector = NDArray[np.float32]


def extract_event_patch(
    batch: EventsBatch,
    bbox: tuple[int, int, int, int],
    target_size: tuple[int, int] = (32, 32),
) -> PatchArray:
    """Crop the event burst to `bbox` and render a fixed-size BGR patch.

    Returns a `(target_size[1], target_size[0], 3)` uint8 array. Resizing
    normalises scale so a close drone and a distant one produce
    comparable signatures.
    """
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    mask = (
        (batch.x_coords >= x0)
        & (batch.x_coords < x1)
        & (batch.y_coords >= y0)
        & (batch.y_coords < y1)
    )
    local_x = (batch.x_coords[mask] - x0).astype(np.int16, copy=False)
    local_y = (batch.y_coords[mask] - y0).astype(np.int16, copy=False)
    polarities_on = batch.polarities[mask] > 0
    width = x1 - x0
    height = y1 - y0
    frame = get_frame(
        window=(local_x, local_y, polarities_on),
        width=width, height=height,
    )
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def normalize_patch(patch: PatchArray) -> FeatureVector:
    """BGR uint8 → unit-norm grayscale float32 feature vector."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    vec = gray.flatten()
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return vec  # all-zero patch — return zeros, callers tolerate that
    return vec / norm


def cosine_similarity(a: FeatureVector, b: FeatureVector) -> float:
    """Dot product of two unit-norm vectors. Falls back to 0 on zero norms."""
    if a.shape != b.shape:
        msg = f"shape mismatch: {a.shape} vs {b.shape}"
        raise ValueError(msg)
    return float(np.dot(a, b))


class SignatureBuffer:
    """Bounded queue of normalized patches per track."""

    def __init__(self, capacity: int = 10) -> None:
        if capacity < 1:
            msg = f"capacity must be >= 1, got {capacity}"
            raise ValueError(msg)
        self._vectors: deque[FeatureVector] = deque(maxlen=capacity)

    def add(self, patch: PatchArray) -> None:
        self._vectors.append(normalize_patch(patch))

    def signature(self) -> FeatureVector | None:
        """Mean of stored normalized patches, re-normalised. None if empty."""
        if not self._vectors:
            return None
        mean = np.mean(self._vectors, axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < 1e-9:
            return mean
        return mean / norm

    def __len__(self) -> int:
        return len(self._vectors)


__all__ = [
    "SignatureBuffer",
    "cosine_similarity",
    "extract_event_patch",
    "normalize_patch",
]
