"""Camera intrinsics — pinhole model parameters for a single camera."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics for one camera.

    Focal lengths are derived from FoV at use-time, so users can specify
    cameras by datasheet FoV rather than focal length. `cx` / `cy` default
    to image centre when unspecified.
    """

    width_px: int
    height_px: int
    h_fov_deg: float
    v_fov_deg: float
    cx: float | None = None
    cy: float | None = None

    @property
    def principal_x(self) -> float:
        return self.cx if self.cx is not None else self.width_px / 2

    @property
    def principal_y(self) -> float:
        return self.cy if self.cy is not None else self.height_px / 2
