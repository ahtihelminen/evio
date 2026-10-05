"""SN-489: the calibration page's rig and site writers (SN-607 coverage)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evio.geometry.calibration_io import camera_to_dict
from evio.geometry.config_write import (
    ConfigWriteError,
    rig_entry,
    write_rig,
    write_site,
)
from evio.geometry.extrinsics import CameraExtrinsics
from evio.geometry.intrinsics import CameraIntrinsics
from evio.geometry.rig_config import CameraCalibration, load_rig
from evio.geometry.site_config import load_site

pytestmark = pytest.mark.t1

_INTRINSICS = {
    "width_px": 1280, "height_px": 720, "h_fov_deg": 65.0, "v_fov_deg": 40.0,
}
_SUBMISSION = {
    "cameras": {
        "evt-cam-0": {
            "intrinsics": _INTRINSICS,
            "extrinsics": {"yaw_deg": -30.0, "pitch_deg": 2.0},
            "position": 1,
        },
        "evt-cam-1": {
            "intrinsics": {**_INTRINSICS, "cx": 635.0, "cy": 362.5},
            "extrinsics": {"yaw_deg": 30.0, "pitch_deg": 1.5, "roll_deg": 0.5},
            "max_range_m": 800.0,
            "position": 0,
        },
    },
}
_MOUNT = {
    "latitude_deg": 60.1699, "longitude_deg": 24.9384,
    "altitude_m": 20.0, "world_yaw_deg": 12.5,
}


def test_write_rig_roundtrips_a_valid_submission(tmp_path: Path) -> None:
    path = write_rig(tmp_path, _SUBMISSION)

    assert path == tmp_path / "rig.yaml"
    assert load_rig(path) == {
        "evt-cam-0": CameraCalibration(
            intrinsics=CameraIntrinsics(**_INTRINSICS),
            extrinsics=CameraExtrinsics(yaw_deg=-30.0, pitch_deg=2.0),
            position=1,
        ),
        "evt-cam-1": CameraCalibration(
            intrinsics=CameraIntrinsics(**_INTRINSICS, cx=635.0, cy=362.5),
            extrinsics=CameraExtrinsics(
                yaw_deg=30.0, pitch_deg=1.5, roll_deg=0.5,
            ),
            max_range_m=800.0,
            position=0,
        ),
    }


def test_write_rig_refuses_invalid_submission_and_leaves_file_untouched(
    tmp_path: Path,
) -> None:
    before = write_rig(tmp_path, _SUBMISSION).read_bytes()
    camera = _SUBMISSION["cameras"]["evt-cam-0"]
    invalid = {"cameras": {"evt-cam-0": {
        **camera, "extrinsics": {"yaw_deg": None, "pitch_deg": 2.0},
    }}}

    with pytest.raises(ConfigWriteError, match="yaw_deg"):
        write_rig(tmp_path, invalid)

    assert (tmp_path / "rig.yaml").read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["rig.yaml"]


def test_rig_entry_adds_explicit_panorama_slot() -> None:
    cam = CameraCalibration(
        intrinsics=CameraIntrinsics(**_INTRINSICS),
        extrinsics=CameraExtrinsics(yaw_deg=10.0, pitch_deg=0.0),
        max_range_m=500.0,
        position=2,
    )

    assert rig_entry(cam) == {**camera_to_dict(cam), "position": 2}


def test_write_site_preserves_settings_block(tmp_path: Path) -> None:
    settings = {"fps": 30, "detector": {"threshold": 0.4, "tags": ["a", "b"]}}
    site = tmp_path / "site.yaml"
    site.write_text(yaml.safe_dump({
        "settings": settings,
        "mount": {"latitude_deg": 0.0, "longitude_deg": 0.0},
    }))

    write_site(tmp_path, {"mount": _MOUNT})

    written = yaml.safe_load(site.read_text())
    assert written == {"settings": settings, "mount": _MOUNT}
    assert load_site(site).world_yaw_deg == 12.5


def test_write_site_refuses_invalid_submission(tmp_path: Path) -> None:
    invalid = {"mount": {"longitude_deg": 24.9384, "elevation": 20.0}}

    with pytest.raises(ConfigWriteError, match="elevation"):
        write_site(tmp_path, invalid)

    assert list(tmp_path.iterdir()) == []
