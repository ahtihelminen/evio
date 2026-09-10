"""Load the site file — the per-deployment half of calibration (SN-486).

The site file carries what changes when an appliance moves to a new
location: where it sits and which way it faces. This is the `mount:`
block of today's single `calibration.yaml`, split out so it can be
edited (and re-surveyed) independently of the rig's own camera setup —
see `evio.geometry.rig_config` for that half.

Schema:

    mount:
      latitude_deg: 60.123
      longitude_deg: 24.456
      altitude_m: 50.0
      world_yaw_deg: 0.0
"""

from __future__ import annotations

from pathlib import Path

import yaml

from evio.geometry.mount import MountPose


class SiteConfigError(ValueError):
    """A site file failed to parse. The message names the file and field."""


def load_site(path: Path | str) -> MountPose:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    return parse_mount(data, path)


def parse_mount(data: object, path: Path) -> MountPose:
    if not isinstance(data, dict) or "mount" not in data:
        msg = f"{path}: missing top-level 'mount' block"
        raise SiteConfigError(msg)
    try:
        return MountPose(**data["mount"])
    except TypeError as exc:
        msg = f"{path}: invalid 'mount' block ({exc})"
        raise SiteConfigError(msg) from exc
