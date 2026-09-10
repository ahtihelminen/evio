"""SN-486: the site file + rig config split of calibration.

A site file (location/heading) and a rig config (camera serials/poses),
loaded together, must produce exactly the `Calibration` object today's
single-file `load_calibration` produces from the same data -- every
existing consumer reads that same in-memory shape either way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evio.geometry.calibration_io import (
    load_calibration,
    load_calibration_from_files,
)
from evio.geometry.rig_config import RigConfigError, load_rig
from evio.geometry.site_config import SiteConfigError, load_site

_SITE_YAML = """
mount:
  latitude_deg: 60.1699
  longitude_deg: 24.9384
  altitude_m: 20.0
  world_yaw_deg: 12.5
"""

_RIG_YAML = """
cameras:
  evt-cam-0:
    intrinsics:
      width_px: 1280
      height_px: 720
      h_fov_deg: 65.0
      v_fov_deg: 40.0
    extrinsics:
      yaw_deg: 0.0
      pitch_deg: 0.0
  evt-cam-1:
    intrinsics:
      width_px: 1280
      height_px: 720
      h_fov_deg: 65.0
      v_fov_deg: 40.0
      cx: 635.0
      cy: 362.5
    extrinsics:
      yaw_deg: 60.0
      pitch_deg: 0.0
      roll_deg: 0.5
    max_range_m: 550.0
"""

# The single-file equivalent of the two fixtures above -- same data, today's
# combined schema.
_COMBINED_YAML = _SITE_YAML + _RIG_YAML


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_two_file_load_equals_single_file_load(tmp_path: Path) -> None:
    site_path = _write(tmp_path, "site.yaml", _SITE_YAML)
    rig_path = _write(tmp_path, "rig.yaml", _RIG_YAML)
    combined_path = _write(tmp_path, "calibration.yaml", _COMBINED_YAML)

    from_two_files = load_calibration_from_files(site_path, rig_path)
    from_one_file = load_calibration(combined_path)

    assert from_two_files == from_one_file


def test_site_file_names_file_and_field_on_missing_mount_block(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "site.yaml", "not_mount:\n  x: 1\n")
    with pytest.raises(SiteConfigError, match=r"site\.yaml") as exc:
        load_site(path)
    assert "mount" in str(exc.value)


def test_site_file_names_file_and_field_on_bad_mount_type(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "site.yaml",
        "mount:\n  latitude_deg: 60.0\n  longitude_deg: 24.0\n"
        "  altitude_m: 1.0\n  world_yaw_deg: 0.0\n  bogus_field: 1\n",
    )
    with pytest.raises(SiteConfigError, match=r"site\.yaml") as exc:
        load_site(path)
    assert "bogus_field" in str(exc.value)


def test_rig_file_names_file_and_field_on_missing_cameras_block(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "rig.yaml", "not_cameras:\n  x: 1\n")
    with pytest.raises(RigConfigError, match=r"rig\.yaml") as exc:
        load_rig(path)
    assert "cameras" in str(exc.value)


def test_rig_file_names_file_serial_and_field_on_null_extrinsic(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "rig.yaml",
        "cameras:\n  evt-cam-0:\n"
        "    intrinsics:\n      width_px: 1280\n      height_px: 720\n"
        "      h_fov_deg: 65.0\n      v_fov_deg: 40.0\n"
        "    extrinsics:\n      yaw_deg: null\n      pitch_deg: 0.0\n",
    )
    with pytest.raises(RigConfigError, match=r"rig\.yaml") as exc:
        load_rig(path)
    assert "evt-cam-0" in str(exc.value)
    assert "yaw_deg" in str(exc.value)
