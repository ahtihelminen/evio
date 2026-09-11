"""Load and save the calibration YAML.

Schema (today's single-file layout; see `evio.geometry.site_config` and
`evio.geometry.rig_config` for the two-file split of the same schema,
SN-486):

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
from evio.geometry.rig_config import (
    CameraCalibration,
    load_rig,
    parse_cameras,
    parse_yaml,
)
from evio.geometry.site_config import load_site, parse_mount

__all__ = [
    "Calibration",
    "CalibrationFault",
    "CameraCalibration",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "check_live_cameras",
    "is_placeholder_serial",
    "load_calibration",
    "load_calibration_from_files",
    "require_cameras",
    "save_calibration",
]


@dataclass(frozen=True, slots=True)
class CalibrationFault:
    """A camera-serial/config mismatch, reported instead of exiting (SN-481).

    `message` is the operator-facing text (unchanged from SN-469); the
    other fields let a future UI render specifics without re-parsing it.
    """

    message: str
    missing: tuple[str, ...] = ()
    live: tuple[str, ...] = ()
    unconfigured: tuple[str, ...] = ()
    placeholders: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Calibration:
    """All geometry config for one appliance."""

    mount: MountPose
    cameras: dict[str, CameraCalibration]
    # Set when boot's calibration-matching found a serial/config mismatch
    # (SN-481). The process still starts; this is reported fault state for
    # a later UI/API surface, not a reason to exit.
    fault: CalibrationFault | None = None


class CalibrationFileError(ValueError):
    """A combined calibration file failed to parse. The message names it."""


def load_calibration(path: Path | str) -> Calibration:
    data = parse_yaml(Path(path), CalibrationFileError)
    return _from_dict(data, Path(path))


def load_calibration_from_files(
    site_path: Path | str, rig_path: Path | str,
) -> Calibration:
    """Assemble a `Calibration` from a site file + a rig config (SN-486).

    Produces the exact same `Calibration(mount, cameras)` shape as
    `load_calibration`, so every existing consumer keeps working against it
    unchanged. Each file is parsed independently (`site_config.load_site`,
    `rig_config.load_rig`) and raises its own named error on malformation.
    """
    mount = load_site(site_path)
    cameras = load_rig(rig_path)
    return Calibration(mount=mount, cameras=cameras)


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
    mount = parse_mount(data, path)
    cameras = parse_cameras(data["cameras"], path)
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
