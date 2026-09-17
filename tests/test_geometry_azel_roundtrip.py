"""Ground-truth geodesy + inverse-projection round-trips."""

from __future__ import annotations

import math

import pytest

from evio.geometry.extrinsics import CameraExtrinsics
from evio.geometry.geodesy import geodetic_to_azel
from evio.geometry.intrinsics import CameraIntrinsics
from evio.geometry.mount import MountPose
from evio.geometry.project import (
    apparent_half_extent_px,
    pixel_to_world_az_el,
    world_az_el_to_pixel,
)

pytestmark = pytest.mark.t1

_MOUNT = MountPose(
    latitude_deg=65.17738298, longitude_deg=26.04051037,
    altitude_m=0.0, world_yaw_deg=0.0,
)
_INTR = CameraIntrinsics(
    width_px=1280, height_px=720, h_fov_deg=29.925, v_fov_deg=16.6,
)
_EXTR = CameraExtrinsics(yaw_deg=15.0, pitch_deg=5.0, roll_deg=0.0)


@pytest.mark.parametrize(
    ("u", "v"),
    [(640.0, 360.0), (400.0, 200.0), (900.0, 500.0), (640.0, 100.0)],
)
def test_pixel_world_pixel_roundtrip(u: float, v: float) -> None:
    az, el = pixel_to_world_az_el(u, v, _INTR, _EXTR, _MOUNT)
    uv = world_az_el_to_pixel(az, el, _INTR, _EXTR, _MOUNT)
    assert uv is not None
    assert uv[0] == pytest.approx(u, abs=1e-6)
    assert uv[1] == pytest.approx(v, abs=1e-6)


def test_behind_camera_returns_none() -> None:
    # Directly opposite the camera's boresight (yaw 15 -> behind ~= 195).
    assert world_az_el_to_pixel(195.0, 0.0, _INTR, _EXTR, _MOUNT) is None


def _offset(
    lat: float, lon: float, dn_m: float, de_m: float,
) -> tuple[float, float]:
    """Crude north/east metre offset to (lat, lon) for test targets."""
    dlat = dn_m / 111_320.0
    dlon = de_m / (111_320.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def test_geodetic_azimuth_cardinals() -> None:
    lat, lon = _MOUNT.latitude_deg, _MOUNT.longitude_deg
    n_lat, n_lon = _offset(lat, lon, 1000.0, 0.0)
    az_n, _, rng_n = geodetic_to_azel(lat, lon, 0.0, n_lat, n_lon, 50.0)
    # North wraps: 360 ≡ 0, so compare the circular distance to north.
    assert min(az_n, 360.0 - az_n) == pytest.approx(0.0, abs=0.5)
    assert rng_n == pytest.approx(1000.0, rel=0.02)

    e_lat, e_lon = _offset(lat, lon, 0.0, 1000.0)
    az_e, _, _ = geodetic_to_azel(lat, lon, 0.0, e_lat, e_lon, 0.0)
    assert az_e == pytest.approx(90.0, abs=0.5)


def test_geodetic_elevation_positive_when_higher() -> None:
    lat, lon = _MOUNT.latitude_deg, _MOUNT.longitude_deg
    n_lat, n_lon = _offset(lat, lon, 1000.0, 0.0)
    _, el, _ = geodetic_to_azel(lat, lon, 0.0, n_lat, n_lon, 100.0)
    assert 0.0 < el < 20.0


def test_apparent_half_extent_px() -> None:
    # 0.43 m drone at 100 m through the OPS cam (fx = 640 / tan(14.96 deg)).
    hw, hh = apparent_half_extent_px(0.43, 100.0, _INTR)
    assert hw == pytest.approx(5.15, abs=0.1)
    assert hh == pytest.approx(5.31, abs=0.1)
    # Apparent size scales as 1 / range.
    hw_far, _ = apparent_half_extent_px(0.43, 200.0, _INTR)
    assert hw_far == pytest.approx(hw / 2, rel=1e-9)
    # Degenerate range / span -> no box.
    assert apparent_half_extent_px(0.43, 0.0, _INTR) == (0.0, 0.0)
    assert apparent_half_extent_px(0.0, 100.0, _INTR) == (0.0, 0.0)
