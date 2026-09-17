"""Tests for the fov_publish cone reduction."""

from __future__ import annotations

import pytest

from evio.geometry.calibration_io import Calibration, CameraCalibration
from evio.geometry.extrinsics import CameraExtrinsics
from evio.geometry.fov import DEFAULT_MAX_RANGE_M, build_fov_cones, fov_response
from evio.geometry.intrinsics import CameraIntrinsics
from evio.geometry.mount import MountPose

pytestmark = pytest.mark.t1


def _calibration(*, world_yaw_deg: float = 0.0) -> Calibration:
    return Calibration(
        mount=MountPose(
            latitude_deg=60.18, longitude_deg=24.83,
            altitude_m=3.2, world_yaw_deg=world_yaw_deg,
        ),
        cameras={
            "ev-1": CameraCalibration(
                intrinsics=CameraIntrinsics(
                    width_px=1280, height_px=720,
                    h_fov_deg=65.0, v_fov_deg=35.0,
                ),
                extrinsics=CameraExtrinsics(yaw_deg=60.0, pitch_deg=5.0),
            ),
            "mono-zenith": CameraCalibration(
                intrinsics=CameraIntrinsics(
                    width_px=4504, height_px=4504,
                    h_fov_deg=95.0, v_fov_deg=95.0,
                ),
                extrinsics=CameraExtrinsics(yaw_deg=0.0, pitch_deg=90.0),
            ),
        },
    )


def test_cone_per_camera_with_shared_position() -> None:
    cones = build_fov_cones(_calibration())
    assert [c.id for c in cones] == ["ev-1", "mono-zenith"]
    for cone in cones:
        assert cone.lat_deg == 60.18
        assert cone.lng_deg == 24.83
        assert cone.alt_m == 3.2


def test_yaw_is_world_frame() -> None:
    cones = build_fov_cones(_calibration(world_yaw_deg=30.0))
    ev1 = next(c for c in cones if c.id == "ev-1")
    assert ev1.yaw_deg == 90.0  # 60 (mast) + 30 (world) = 90


def test_yaw_wraps_to_0_360() -> None:
    cones = build_fov_cones(_calibration(world_yaw_deg=320.0))
    ev1 = next(c for c in cones if c.id == "ev-1")
    assert ev1.yaw_deg == 20.0  # (60 + 320) % 360


def test_fov_and_pitch_from_calibration() -> None:
    cones = build_fov_cones(_calibration())
    ev1 = next(c for c in cones if c.id == "ev-1")
    assert ev1.fov_h_deg == 65.0
    assert ev1.fov_v_deg == 35.0
    assert ev1.pitch_deg == 5.0
    zenith = next(c for c in cones if c.id == "mono-zenith")
    assert zenith.pitch_deg == 90.0  # straight up, rendered as a disc


def test_max_range_default_and_override() -> None:
    cones = build_fov_cones(
        _calibration(), max_range_m={"ev-1": 800.0},
    )
    ev1 = next(c for c in cones if c.id == "ev-1")
    zenith = next(c for c in cones if c.id == "mono-zenith")
    assert ev1.max_range_m == 800.0  # overridden
    assert zenith.max_range_m == DEFAULT_MAX_RANGE_M  # default fallback


def test_max_range_from_calibration_yaml_key() -> None:
    # Precedence: caller override dict > per-camera calibration > default.
    cal = _calibration()
    cal.cameras["ev-1"] = CameraCalibration(
        intrinsics=cal.cameras["ev-1"].intrinsics,
        extrinsics=cal.cameras["ev-1"].extrinsics,
        max_range_m=550.0,
    )
    from_yaml = build_fov_cones(cal)
    assert next(c for c in from_yaml if c.id == "ev-1").max_range_m == 550.0
    overridden = build_fov_cones(cal, max_range_m={"ev-1": 900.0})
    assert next(c for c in overridden if c.id == "ev-1").max_range_m == 900.0


def test_fov_response_shape_is_static() -> None:
    body = fov_response(build_fov_cones(_calibration()))
    assert body["dynamic"] is False
    assert len(body["fovs"]) == 2
    first = body["fovs"][0]
    assert first["id"] == "ev-1"
    assert first["position"]["lat_deg"] == 60.18
    assert set(first) == {
        "id", "position", "yaw_deg", "pitch_deg",
        "fov_h_deg", "fov_v_deg", "max_range_m",
    }


def test_null_island_mount_serializes_position_none() -> None:
    cal = _calibration()
    cal = Calibration(
        mount=MountPose(latitude_deg=0.0, longitude_deg=0.0, altitude_m=0.0),
        cameras=cal.cameras,
    )
    cones = build_fov_cones(cal)
    assert all(not c.has_position for c in cones)
    body = fov_response(cones)
    # The HTTP surface must match the WS surface: no (0,0) origin claims.
    assert all(entry["position"] is None for entry in body["fovs"])
