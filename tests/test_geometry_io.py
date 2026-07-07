"""YAML round-trip tests for the calibration loader."""

from __future__ import annotations

from pathlib import Path

from evio.geometry import (
    Calibration,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    MountPose,
    load_calibration,
    save_calibration,
)


def _sample() -> Calibration:
    return Calibration(
        mount=MountPose(
            latitude_deg=60.1699,
            longitude_deg=24.9384,
            altitude_m=20.0,
            world_yaw_deg=12.5,
        ),
        cameras={
            "evt-cam-0": CameraCalibration(
                intrinsics=CameraIntrinsics(
                    width_px=1280, height_px=720,
                    h_fov_deg=65.0, v_fov_deg=40.0,
                ),
                extrinsics=CameraExtrinsics(yaw_deg=0.0, pitch_deg=0.0),
            ),
            "evt-cam-1": CameraCalibration(
                intrinsics=CameraIntrinsics(
                    width_px=1280, height_px=720,
                    h_fov_deg=65.0, v_fov_deg=40.0,
                    cx=635.0, cy=362.5,
                ),
                extrinsics=CameraExtrinsics(
                    yaw_deg=60.0, pitch_deg=0.0, roll_deg=0.5,
                ),
                max_range_m=550.0,
            ),
            "mono-cam-0": CameraCalibration(
                intrinsics=CameraIntrinsics(
                    width_px=2048, height_px=1536,
                    h_fov_deg=45.0, v_fov_deg=35.0,
                ),
                extrinsics=CameraExtrinsics(yaw_deg=0.0, pitch_deg=60.0),
            ),
        },
    )


def test_roundtrip_preserves_calibration(tmp_path: Path) -> None:
    original = _sample()
    path = tmp_path / "cal.yaml"
    save_calibration(original, path)
    loaded = load_calibration(path)
    assert loaded == original


def test_optional_intrinsic_centre_omitted_in_yaml(tmp_path: Path) -> None:
    """cx/cy=None should not appear in the saved YAML."""
    cal = _sample()
    path = tmp_path / "cal.yaml"
    save_calibration(cal, path)
    text = path.read_text()
    # evt-cam-0 has no explicit cx/cy, so they should be omitted
    evt0_block = text.split("evt-cam-0:")[1].split("evt-cam-1:")[0]
    assert "cx:" not in evt0_block
    assert "cy:" not in evt0_block


def test_optional_max_range_omitted_when_unset(tmp_path: Path) -> None:
    cal = _sample()
    path = tmp_path / "cal.yaml"
    save_calibration(cal, path)
    text = path.read_text()
    evt0_block = text.split("evt-cam-0:")[1].split("evt-cam-1:")[0]
    evt1_block = text.split("evt-cam-1:")[1].split("mono-cam-0:")[0]
    assert "max_range_m:" not in evt0_block
    assert "max_range_m: 550.0" in evt1_block
