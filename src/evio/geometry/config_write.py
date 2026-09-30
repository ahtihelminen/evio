"""Write the rig config and the site file (SN-489).

The calibration page submits JSON; these writers validate it through the
same loaders that read it back (`rig_config.parse_cameras`,
`site_config.parse_mount`) and then serialise the parsed objects, so what
lands on disk is exactly what the loader will hand every consumer.

Two properties the field depends on:

- **Ordinary file IO only.** The config directory is writable by the
  service's own user through an ordinary directory permission. Nothing
  here shells out, and no `sudoers` grant exists or is needed for it
  (PROJECT.md constraint; README "Setup").
- **Atomic.** The YAML goes to a temporary file in the same directory and
  is renamed over the target, so a crash or power cut mid-write leaves the
  previous config intact rather than a truncated one.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from evio.geometry.calibration_io import (
    PLACEHOLDER_SERIAL_PREFIX,
    camera_to_dict,
)
from evio.geometry.rig_config import CameraCalibration, parse_cameras
from evio.geometry.site_config import parse_mount

RIG_FILENAME = "rig.yaml"
SITE_FILENAME = "site.yaml"

_SENTINEL = re.compile(re.escape(PLACEHOLDER_SERIAL_PREFIX) + r"\w*")


class ConfigWriteError(ValueError):
    """A submission was refused; nothing was written.

    The message names the offending field, camera or sentinel, so the
    calibration page can show the operator what to fix.
    """


def write_rig(config_dir: Path | str, submission: dict[str, Any]) -> Path:
    """Validate a rig-config submission and write `rig.yaml` atomically."""
    path = Path(config_dir) / RIG_FILENAME
    cameras = submission.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        msg = f"{path}: submission needs a non-empty 'cameras' block"
        raise ConfigWriteError(msg)
    _reject_sentinels(submission, path)
    try:
        parsed = parse_cameras(cameras, path)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        msg = f"{path}: invalid rig config ({exc})"
        raise ConfigWriteError(msg) from exc
    _write_yaml(path, {
        "cameras": {s: rig_entry(c) for s, c in parsed.items()},
    })
    return path


def rig_entry(cam: CameraCalibration) -> dict[str, Any]:
    """`camera_to_dict` plus the rig-only explicit panorama slot (SN-550).

    The slot lives in the rig config, not in the combined calibration file
    `camera_to_dict` serialises, so it is added here rather than there.
    """
    entry = camera_to_dict(cam)
    if cam.position is not None:
        entry["position"] = cam.position
    return entry


def write_site(config_dir: Path | str, submission: dict[str, Any]) -> Path:
    """Validate a site-file submission and write `site.yaml` atomically.

    Only the `mount:` block is replaced; any other block already in the
    file (the runner's `settings:`) is written back unchanged.
    """
    path = Path(config_dir) / SITE_FILENAME
    _reject_sentinels(submission, path)
    try:
        mount = parse_mount(submission, path)
    except (TypeError, ValueError) as exc:
        msg = f"{path}: invalid site file ({exc})"
        raise ConfigWriteError(msg) from exc
    _write_yaml(path, {**_other_blocks(path), "mount": asdict(mount)})
    return path


def _other_blocks(path: Path) -> dict[str, Any]:
    """The site file's blocks besides `mount:`, kept across a save.

    The runner's `settings:` block (SN-388) lives in the same file; the
    calibration page only ever edits the mount, so it must not drop them.
    """
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _reject_sentinels(submission: dict[str, Any], path: Path) -> None:
    """Refuse a submission that still carries a `REPLACE_ME_*` sentinel.

    The template ships sentinels so an unconfigured unit fails visibly
    (SN-469); writing one back through the editor would re-create exactly
    the state the sentinel exists to make impossible to miss.
    """
    found = _SENTINEL.search(str(submission))
    if found:
        msg = (
            f"{path}: submission still contains the placeholder "
            f"{found.group()!r} — replace it with the real value"
        )
        raise ConfigWriteError(msg)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write `data` to `path` atomically: temp file, fsync, rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.new")
    try:
        with tmp.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        # A successful rename already moved it; this clears the partial
        # file a crashed serialisation leaves behind.
        tmp.unlink(missing_ok=True)
