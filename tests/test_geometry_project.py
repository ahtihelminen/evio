"""Golden-case tests for the pixel -> world az/el projection."""

from __future__ import annotations

import math

import pytest

from evio.geometry import (
    CameraExtrinsics,
    CameraIntrinsics,
    MountPose,
    pixel_to_world_az_el,
)

# A representative 1280x720 sensor with a ~65 deg horizontal FoV.
INTR = CameraIntrinsics(
    width_px=1280, height_px=720, h_fov_deg=65.0, v_fov_deg=40.0,
)
NORTH_MOUNT = MountPose(
    latitude_deg=60.0, longitude_deg=24.0, world_yaw_deg=0.0,
)


def _centre(intr: CameraIntrinsics) -> tuple[float, float]:
    return intr.principal_x, intr.principal_y


def _approx(actual: float, expected: float, tol: float = 1e-6) -> bool:
    """Compare azimuth in degrees with wrap-around tolerance."""
    diff = (actual - expected + 180.0) % 360.0 - 180.0
    return abs(diff) < tol


# ---- single-camera centre-pixel cases ----

@pytest.mark.parametrize(
    ("yaw", "pitch", "expected_az", "expected_el"),
    [
        (0.0, 0.0, 0.0, 0.0),
        (90.0, 0.0, 90.0, 0.0),
        (180.0, 0.0, 180.0, 0.0),
        (270.0, 0.0, 270.0, 0.0),
        (0.0, 45.0, 0.0, 45.0),
        (0.0, 90.0, 0.0, 90.0),  # zenith — az ill-defined
        (60.0, 30.0, 60.0, 30.0),
    ],
)
def test_centre_pixel_matches_extrinsics(
    yaw: float, pitch: float, expected_az: float, expected_el: float,
) -> None:
    extr = CameraExtrinsics(yaw_deg=yaw, pitch_deg=pitch)
    u, v = _centre(INTR)
    az, el = pixel_to_world_az_el(u, v, INTR, extr, NORTH_MOUNT)
    assert math.isclose(el, expected_el, abs_tol=1e-6)
    if expected_el < 89.999:
        assert _approx(az, expected_az)


# ---- off-centre pixel cases ----

def test_pixel_right_of_centre_yields_az_greater_than_zero() -> None:
    extr = CameraExtrinsics(yaw_deg=0.0, pitch_deg=0.0)
    u = INTR.principal_x + 100
    v = INTR.principal_y
    az, el = pixel_to_world_az_el(u, v, INTR, extr, NORTH_MOUNT)
    assert 0.0 < az < 30.0
    assert math.isclose(el, 0.0, abs_tol=1e-6)


def test_pixel_above_centre_yields_positive_elevation() -> None:
    extr = CameraExtrinsics(yaw_deg=0.0, pitch_deg=0.0)
    u = INTR.principal_x
    v = INTR.principal_y - 100
    az, el = pixel_to_world_az_el(u, v, INTR, extr, NORTH_MOUNT)
    assert el > 0.0
    assert math.isclose(az, 0.0, abs_tol=1e-6)


# ---- mount yaw applied uniformly ----

def test_mount_yaw_rotates_az_for_all_cameras() -> None:
    """Rotating the mount by N degrees adds N to every camera's az."""
    extr = CameraExtrinsics(yaw_deg=30.0, pitch_deg=0.0)
    u, v = _centre(INTR)
    base = pixel_to_world_az_el(u, v, INTR, extr, NORTH_MOUNT)
    rotated_mount = MountPose(
        latitude_deg=60.0, longitude_deg=24.0, world_yaw_deg=45.0,
    )
    rotated = pixel_to_world_az_el(u, v, INTR, extr, rotated_mount)
    assert math.isclose(rotated[1], base[1], abs_tol=1e-6)
    assert _approx(rotated[0], base[0] + 45.0)


# ---- FoV check via horizontal extent ----

def test_horizontal_fov_matches_intrinsics() -> None:
    """Leftmost to rightmost pixel spans ~h_fov_deg in azimuth."""
    extr = CameraExtrinsics(yaw_deg=0.0, pitch_deg=0.0)
    left = pixel_to_world_az_el(
        0.0, INTR.principal_y, INTR, extr, NORTH_MOUNT,
    )
    right = pixel_to_world_az_el(
        INTR.width_px - 1, INTR.principal_y, INTR, extr, NORTH_MOUNT,
    )
    span = (right[0] - left[0]) % 360.0
    assert math.isclose(span, INTR.h_fov_deg, abs_tol=0.5)
