"""YAML round-trip tests for the calibration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from evio.geometry import (
    Calibration,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    MountPose,
    load_calibration,
    require_cameras,
    save_calibration,
)
from evio.geometry.calibration_io import (
    check_live_cameras,
    is_placeholder_serial,
)

pytestmark = pytest.mark.t1


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


_MOUNT_BLOCK = """
mount:
  latitude_deg: 60.1699
  longitude_deg: 24.9384
  altitude_m: 20.0
  world_yaw_deg: 0.0
"""


def _write_cal(path: Path, cameras: str) -> Path:
    path.write_text(f"{_MOUNT_BLOCK}cameras:\n{cameras}")
    return path


_INTRINSICS = (
    "    intrinsics:\n"
    "      width_px: 1280\n"
    "      height_px: 720\n"
    "      h_fov_deg: 65.0\n"
    "      v_fov_deg: 40.0\n"
)


def _camera_block(serial: str, extrinsics: str) -> str:
    return f"  {serial}:\n{_INTRINSICS}    extrinsics:\n{extrinsics}"


@pytest.mark.parametrize("field", ["yaw_deg", "pitch_deg", "roll_deg"])
def test_null_extrinsic_names_file_serial_and_field(
    tmp_path: Path, field: str,
) -> None:
    """A null angle must not load — it kills the panorama sort much later."""
    angles = {"yaw_deg": "0.0", "pitch_deg": "0.0", "roll_deg": "0.0"}
    angles[field] = "null"
    path = _write_cal(
        tmp_path / "calibration.ops001.yaml",
        _camera_block(
            "evt-cam-0",
            "".join(f"      {k}: {v}\n" for k, v in angles.items()),
        ),
    )
    with pytest.raises(ValueError, match=r"calibration\.ops001\.yaml") as exc:
        load_calibration(path)
    assert "evt-cam-0" in str(exc.value)
    assert field in str(exc.value)


@pytest.mark.parametrize("field", ["yaw_deg", "pitch_deg"])
def test_absent_required_extrinsic_names_file_serial_and_field(
    tmp_path: Path, field: str,
) -> None:
    angles = {"yaw_deg": "0.0", "pitch_deg": "0.0"}
    del angles[field]
    path = _write_cal(
        tmp_path / "calibration.ops001.yaml",
        _camera_block(
            "evt-cam-0",
            "".join(f"      {k}: {v}\n" for k, v in angles.items()),
        ),
    )
    with pytest.raises(ValueError, match=r"calibration\.ops001\.yaml") as exc:
        load_calibration(path)
    assert "evt-cam-0" in str(exc.value)
    assert field in str(exc.value)


def test_absent_roll_keeps_its_documented_default(tmp_path: Path) -> None:
    """`roll_deg` is optional per the schema; only a null value is a fault."""
    path = _write_cal(
        tmp_path / "cal.yaml",
        _camera_block(
            "evt-cam-0", "      yaw_deg: 10.0\n      pitch_deg: 2.0\n",
        ),
    )
    cameras = load_calibration(path).cameras
    assert cameras["evt-cam-0"].extrinsics.roll_deg == 0.0


def test_require_cameras_rejects_disjoint_serials(tmp_path: Path) -> None:
    path = _write_cal(
        tmp_path / "calibration.ops001.yaml",
        _camera_block(
            "evt-cam-0", "      yaw_deg: 0.0\n      pitch_deg: 0.0\n",
        ),
    )
    calibration = load_calibration(path)
    with pytest.raises(ValueError, match=r"calibration\.ops001\.yaml") as exc:
        require_cameras(calibration, ["other-cam-1", "other-cam-2"], path)
    assert "other-cam-1" in str(exc.value)
    assert "evt-cam-0" in str(exc.value)


def test_require_cameras_accepts_a_known_subset(tmp_path: Path) -> None:
    path = _write_cal(
        tmp_path / "cal.yaml",
        _camera_block("a", "      yaw_deg: 0.0\n      pitch_deg: 0.0\n")
        + _camera_block("b", "      yaw_deg: 1.0\n      pitch_deg: 0.0\n"),
    )
    require_cameras(load_calibration(path), ["b"], path)


def test_check_live_cameras_names_missing_and_found_on_zero_match() -> None:
    """SN-469: no configured serial is live -- fail loudly, not silently."""
    with pytest.raises(ValueError) as exc:
        check_live_cameras(["cam-a", "cam-b"], {"cam-x", "cam-y"})
    message = str(exc.value)
    assert "cam-a" in message
    assert "cam-b" in message
    assert "cam-x" in message
    assert "cam-y" in message


def test_check_live_cameras_names_specific_missing_and_unconfigured() -> None:
    """SN-469: a partial match names specifics, not just a count mismatch."""
    with pytest.raises(ValueError) as exc:
        check_live_cameras(["cam-a", "cam-b"], {"cam-b", "cam-x"})
    message = str(exc.value)
    assert "cam-a" in message
    assert "cam-x" in message


def test_check_live_cameras_accepts_an_exact_match() -> None:
    """SN-469: every configured serial live -- no error, nothing to report."""
    unconfigured = check_live_cameras(["cam-a", "cam-b"], {"cam-a", "cam-b"})
    assert unconfigured == []


def test_check_live_cameras_reports_present_but_unconfigured_cameras() -> None:
    """A healthy connected camera outside the configured list is named, not
    silently ignored -- returned for the caller to report, without raising.
    """
    unconfigured = check_live_cameras(["cam-a"], {"cam-a", "cam-extra"})
    assert unconfigured == ["cam-extra"]


def test_is_placeholder_serial_flags_the_replace_me_sentinel() -> None:
    assert is_placeholder_serial("REPLACE_ME_0")
    assert not is_placeholder_serial("4110022520")
