"""Load and save the calibration YAML.

Schema:

    mount:
      latitude_deg: 60.123
      longitude_deg: 24.456
      altitude_m: 50.0
      world_yaw_deg: 0.0

    cameras:
      <camera-serial>:
        intrinsics:
          width_px: 1280
          height_px: 720
          h_fov_deg: 65.0
          v_fov_deg: 40.0
          cx: 640.0     # optional, defaults to width/2
          cy: 360.0     # optional, defaults to height/2
        extrinsics:
          yaw_deg: 0.0
          pitch_deg: 0.0
          roll_deg: 0.0
        max_range_m: 800.0   # optional — operator-tunable useful detection
                             # range for the fov_publish cone (NOT calibration;
                             # see evio.geometry.fov)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from evio.geometry.extrinsics import CameraExtrinsics
from evio.geometry.intrinsics import CameraIntrinsics
from evio.geometry.mount import MountPose


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    # Operator-authored useful detection range for the fov_publish cone
    # (project_test_site_calibration_fov); None -> the fov builder's default.
    max_range_m: float | None = None


@dataclass(frozen=True, slots=True)
class Calibration:
    """All geometry config for one appliance."""

    mount: MountPose
    cameras: dict[str, CameraCalibration]


def load_calibration(path: Path | str) -> Calibration:
    with Path(path).open() as f:
        data = yaml.safe_load(f)
    return _from_dict(data)


def save_calibration(calibration: Calibration, path: Path | str) -> None:
    with Path(path).open("w") as f:
        yaml.safe_dump(_to_dict(calibration), f, sort_keys=False)


def _from_dict(data: dict[str, Any]) -> Calibration:
    mount = MountPose(**data["mount"])
    cameras = {
        serial: CameraCalibration(
            intrinsics=CameraIntrinsics(**entry["intrinsics"]),
            extrinsics=CameraExtrinsics(**entry["extrinsics"]),
            max_range_m=entry.get("max_range_m"),
        )
        for serial, entry in data["cameras"].items()
    }
    return Calibration(mount=mount, cameras=cameras)


def _to_dict(calibration: Calibration) -> dict[str, Any]:
    return {
        "mount": asdict(calibration.mount),
        "cameras": {
            serial: _camera_to_dict(cam)
            for serial, cam in calibration.cameras.items()
        },
    }


def _camera_to_dict(cam: CameraCalibration) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "intrinsics": _strip_nones(asdict(cam.intrinsics)),
        "extrinsics": asdict(cam.extrinsics),
    }
    if cam.max_range_m is not None:
        entry["max_range_m"] = cam.max_range_m
    return entry


def _strip_nones(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}
