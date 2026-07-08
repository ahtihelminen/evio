"""Reduce a Calibration to the per-camera FoV cones for `fov_publish`.

`project_test_site_calibration_fov`: the backend only needs six numbers per
camera to draw a coverage cone on the map -- position (lat/lng/alt), yaw, pitch,
horizontal + vertical FoV, and a max range. The full intrinsics/extrinsics stay
sensor-side. This is the thin pure reduction from evio's existing `Calibration`
to those six numbers, mirroring how `sapient.mapper` reduces to az/el.

Five of the six come straight from the calibration:
- position  <- mount latitude/longitude/altitude (all cameras share the mast)
- yaw_deg   <- extrinsics.yaw_deg + mount.world_yaw_deg (WORLD frame; extrinsics
               yaw is mast-frame, so world_yaw MUST be added), wrapped [0,360)
- pitch_deg <- extrinsics.pitch_deg
- fov_h_deg / fov_v_deg <- intrinsics

`max_range_m` is NOT calibration proper -- it is the operator-authored "useful
detection range" (§9 answer #7, `project_evio_testsite_compliance`).
Precedence: the caller's per-serial override dict, then the camera's optional
`max_range_m` key in the calibration YAML, then `DEFAULT_MAX_RANGE_M`.

The typed wire form of a cone is `test_site_protocol.capabilities.FovEntry`
(protocol v0.3.0); `evio.edge.fov_publish` converts `FovCone` to it. The dict
helpers below stay protocol-free so this module keeps zero testsite deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evio.geometry.calibration_io import Calibration

DEFAULT_MAX_RANGE_M = 800.0


@dataclass(frozen=True, slots=True)
class FovCone:
    """One camera's coverage cone, in the backend's six-number form."""

    id: str
    lat_deg: float
    lng_deg: float
    alt_m: float
    yaw_deg: float
    pitch_deg: float
    fov_h_deg: float
    fov_v_deg: float
    max_range_m: float

    @property
    def has_position(self) -> bool:
        """False for the null-island placeholder (mount pose not yet set).

        The single source of the rule: an unsurveyed mount declares its cones
        WITHOUT an origin (`position=None` on every surface) rather than
        claiming (0, 0).
        """
        return not (self.lat_deg == 0.0 and self.lng_deg == 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the calibration_fov `fovs[]` entry shape."""
        position = (
            {
                "lat_deg": self.lat_deg,
                "lng_deg": self.lng_deg,
                "alt_m": self.alt_m,
            }
            if self.has_position
            else None
        )
        return {
            "id": self.id,
            "position": position,
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "fov_h_deg": self.fov_h_deg,
            "fov_v_deg": self.fov_v_deg,
            "max_range_m": self.max_range_m,
        }


def build_fov_cones(
    calibration: Calibration,
    *,
    max_range_m: dict[str, float] | None = None,
    default_max_range_m: float = DEFAULT_MAX_RANGE_M,
) -> list[FovCone]:
    """Build one FoV cone per camera in the calibration.

    `max_range_m` maps camera serial -> useful range; serials not present use
    the camera's calibration `max_range_m`, else `default_max_range_m`.
    Cameras are emitted in calibration order.
    """
    ranges = max_range_m or {}
    mount = calibration.mount
    cones: list[FovCone] = []
    for serial, camera in calibration.cameras.items():
        world_yaw = (camera.extrinsics.yaw_deg + mount.world_yaw_deg) % 360.0
        cone_range = ranges.get(serial, camera.max_range_m)
        if cone_range is None:
            cone_range = default_max_range_m
        cones.append(
            FovCone(
                id=serial,
                lat_deg=mount.latitude_deg,
                lng_deg=mount.longitude_deg,
                alt_m=mount.altitude_m,
                yaw_deg=world_yaw,
                pitch_deg=camera.extrinsics.pitch_deg,
                fov_h_deg=camera.intrinsics.h_fov_deg,
                fov_v_deg=camera.intrinsics.v_fov_deg,
                max_range_m=cone_range,
            ),
        )
    return cones


def fov_response(
    cones: list[FovCone], *, dynamic: bool = False,
) -> dict[str, Any]:
    """The `GET /control/fov` response body (static ring -> dynamic=False)."""
    return {"dynamic": dynamic, "fovs": [cone.to_dict() for cone in cones]}
