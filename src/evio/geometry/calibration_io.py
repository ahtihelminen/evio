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
          yaw_deg: 0.0      # mandatory; a null angle is rejected on load
          pitch_deg: 0.0    # mandatory; a null angle is rejected on load
          roll_deg: 0.0     # optional, defaults to 0; null is rejected
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
    return _from_dict(data, Path(path))


def require_cameras(
    calibration: Calibration, serials: list[str], path: Path | str,
) -> None:
    """Raise unless every requested serial is calibrated.

    A serial the calibration has never heard of means the operator pointed
    the runner at the wrong site file; say so at startup, naming the file,
    rather than letting the camera drop silently out of the panorama.
    """
    missing = [s for s in serials if s not in calibration.cameras]
    if missing:
        msg = (
            f"{path}: requested camera(s) {missing} are not in the "
            f"calibration (known: {sorted(calibration.cameras)})"
        )
        raise ValueError(msg)


PLACEHOLDER_SERIAL_PREFIX = "REPLACE_ME_"


def is_placeholder_serial(serial: str) -> bool:
    """True for an obviously-not-real-hardware sentinel serial.

    A shipped template must fail this check for every default camera serial,
    so a deployment that never overrides them fails fast and visibly instead
    of silently matching nothing on real hardware (SN-469).
    """
    return serial.startswith(PLACEHOLDER_SERIAL_PREFIX)


def check_live_cameras(configured: list[str], live: set[str]) -> list[str]:
    """Compare configured camera serials against what is actually on the bus.

    Raises, naming every configured serial that matched no live camera and
    every serial actually found, when any configured serial is missing --
    whether none match or only some do. A CAMERA_SERIALS/--serial value that
    doesn't match the connected hardware must fail loudly at startup rather
    than silently run with an empty or reduced camera set (SN-469).

    Returns the live serials that are present but not configured, so the
    caller can name them too -- a healthy connected camera outside the
    configured list is reported, not silently ignored.
    """
    missing = sorted(s for s in configured if s not in live)
    unconfigured = sorted(s for s in live if s not in configured)
    if missing:
        msg = (
            f"configured camera(s) not found on the bus: {missing} "
            f"(live cameras found: {sorted(live)})"
        )
        if unconfigured:
            msg += f"; found on the bus but not configured: {unconfigured}"
        raise ValueError(msg)
    return unconfigured


def save_calibration(calibration: Calibration, path: Path | str) -> None:
    with Path(path).open("w") as f:
        yaml.safe_dump(_to_dict(calibration), f, sort_keys=False)


def _from_dict(data: dict[str, Any], path: Path) -> Calibration:
    mount = MountPose(**data["mount"])
    cameras = {
        serial: CameraCalibration(
            intrinsics=CameraIntrinsics(**entry["intrinsics"]),
            extrinsics=_extrinsics(entry["extrinsics"], serial, path),
            max_range_m=entry.get("max_range_m"),
        )
        for serial, entry in data["cameras"].items()
    }
    return Calibration(mount=mount, cameras=cameras)


def _extrinsics(
    entry: dict[str, Any], serial: str, path: Path,
) -> CameraExtrinsics:
    """Build one camera's pose, rejecting a half-finished calibration.

    A `yaw_deg: null` used to load without complaint and then kill the
    appliance several frames later inside the panorama's yaw sort, with a
    message that named neither the file nor the camera. Under systemd that
    read as a crash-loop, so the angles are checked here instead.
    """
    for angle in ("yaw_deg", "pitch_deg"):
        if entry.get(angle) is None:
            msg = f"{path}: camera {serial!r} has no {angle}"
            raise ValueError(msg)
    # roll_deg stays optional (it defaults to level), but an explicit null
    # is an unfinished entry, not a default.
    if "roll_deg" in entry and entry["roll_deg"] is None:
        msg = f"{path}: camera {serial!r} has a null roll_deg"
        raise ValueError(msg)
    return CameraExtrinsics(**entry)


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
