"""Camera extrinsics — each camera's orientation in the mast frame.

Conventions (documented once, used everywhere):
- Mast frame: +X east, +Y north, +Z up.
- Camera frame: +X right (image), +Y down (image), +Z forward (boresight).
- `yaw_deg` is compass-style: 0 = boresight points along mast +Y (north),
  positive rotates clockwise viewed from above (90 → east).
- `pitch_deg`: 0 = level horizon, positive tilts boresight up (90 = zenith).
- `roll_deg`: rotation around boresight; positive = image right edge drops.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CameraExtrinsics:
    """Pose of a camera in the mast frame."""

    yaw_deg: float
    pitch_deg: float
    roll_deg: float = 0.0
