"""Load the rig file — the per-rig half of calibration (SN-486).

The rig config carries what stays fixed when an appliance moves to a new
site: each camera's serial, intrinsics, and extrinsics. This is the
`cameras:` block of today's single `calibration.yaml`, split out so the
stable rig half can be shared or reused independently of the site's own
location/heading — see `evio.geometry.site_config` for that half.

Schema:

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
        position: 0          # optional — explicit panorama strip slot
                             # (SN-550). Absent everywhere, yaw decides the
                             # order (the OPS appliance's panorama layout).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evio.geometry.extrinsics import CameraExtrinsics
from evio.geometry.intrinsics import CameraIntrinsics


class RigConfigError(ValueError):
    """A rig file failed to parse. The message names the file and field."""


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    # Operator-authored useful detection range for the fov_publish cone
    # (project_test_site_calibration_fov); None -> the fov builder's default.
    max_range_m: float | None = None
    # Operator-authored panorama strip slot (SN-550); None -> this camera's
    # place in the strip is decided by its extrinsic yaw, as it always was.
    position: int | None = None


def parse_yaml(path: Path, kind: type[ValueError]) -> object:
    """Parse `path` as YAML, reporting a syntax error as a config error.

    Each loader owns its file format, so a malformed file is a bad config --
    not a `yaml.YAMLError` leaking out to whatever happens to be calling.
    Semantic errors below already raise `ValueError` subclasses; without
    this, syntactically invalid files raised something callers could not
    anticipate and staged boot died on them (SN-520).
    """
    with path.open() as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            msg = f"{path}: invalid YAML ({exc})"
            raise kind(msg) from exc


def load_rig(path: Path | str) -> dict[str, CameraCalibration]:
    path = Path(path)
    data = parse_yaml(path, RigConfigError)
    if not isinstance(data, dict) or "cameras" not in data:
        msg = f"{path}: missing top-level 'cameras' block"
        raise RigConfigError(msg)
    return parse_cameras(data["cameras"], path)


def parse_cameras(
    cameras: dict[str, Any], path: Path,
) -> dict[str, CameraCalibration]:
    return {
        serial: CameraCalibration(
            intrinsics=_intrinsics(entry["intrinsics"], serial, path),
            extrinsics=_extrinsics(entry["extrinsics"], serial, path),
            max_range_m=entry.get("max_range_m"),
            position=_position(entry, serial, path),
        )
        for serial, entry in cameras.items()
    }


def _position(entry: dict[str, Any], serial: str, path: Path) -> int | None:
    """The optional explicit panorama slot, rejected unless it is an int.

    Checked here for the same reason the angles below are: a string slot
    loads fine and then dies inside the panorama's order sort, far from the
    file that caused it.
    """
    position = entry.get("position")
    if position is None or isinstance(position, int):
        return position
    msg = f"{path}: camera {serial!r} has a non-integer position {position!r}"
    raise RigConfigError(msg)


def _intrinsics(
    entry: dict[str, Any], serial: str, path: Path,
) -> CameraIntrinsics:
    try:
        return CameraIntrinsics(**entry)
    except TypeError as exc:
        msg = (
            f"{path}: camera {serial!r} has an invalid intrinsics block "
            f"({exc})"
        )
        raise RigConfigError(msg) from exc


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
            raise RigConfigError(msg)
    # roll_deg stays optional (it defaults to level), but an explicit null
    # is an unfinished entry, not a default.
    if "roll_deg" in entry and entry["roll_deg"] is None:
        msg = f"{path}: camera {serial!r} has a null roll_deg"
        raise RigConfigError(msg)
    try:
        return CameraExtrinsics(**entry)
    except TypeError as exc:
        msg = (
            f"{path}: camera {serial!r} has an invalid extrinsics block "
            f"({exc})"
        )
        raise RigConfigError(msg) from exc
