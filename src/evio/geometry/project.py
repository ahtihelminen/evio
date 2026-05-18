"""Projection pipeline: camera pixel -> world azimuth/elevation.

All functions are pure, side-effect-free, and return numpy arrays or floats.
The full pipeline composes as:

    pixel -> camera-frame ray (pinhole)
          -> mast-frame ray   (camera extrinsics)
          -> world-frame ray  (mount yaw)
          -> (azimuth, elevation)

Convention recap:
- World frame: +X east, +Y north, +Z up (ENU at mount lat/lng).
- Azimuth: 0 = north, increases clockwise (compass). Returned in [0, 360).
- Elevation: 0 = horizon, +90 = zenith. Returned in [-90, 90].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from evio.geometry.extrinsics import CameraExtrinsics
from evio.geometry.intrinsics import CameraIntrinsics
from evio.geometry.mount import MountPose

Vector3 = NDArray[np.float64]

# Base remap from camera (image) frame to an upright local frame:
#   local_x (east) = cam_x (right)
#   local_y (north) = cam_z (forward, boresight)
#   local_z (up) = -cam_y (cam_y is image-down)
_CAM_TO_UPRIGHT: Vector3 = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
)


def _compass_yaw_matrix(yaw_deg: float) -> Vector3:
    """Rotation around +Z. Compass-style: positive = clockwise from above."""
    a = np.deg2rad(yaw_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _pitch_up_matrix(pitch_deg: float) -> Vector3:
    """Rotation around +X. Positive = +Y rotates toward +Z (up)."""
    a = np.deg2rad(pitch_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _roll_matrix(roll_deg: float) -> Vector3:
    """Rotation around +Y (boresight at yaw=pitch=0). +ve drops right edge."""
    a = np.deg2rad(roll_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def camera_to_mast_rotation(extrinsics: CameraExtrinsics) -> Vector3:
    """3x3 rotation matrix taking a camera-frame ray to the mast frame."""
    return (
        _compass_yaw_matrix(extrinsics.yaw_deg)
        @ _pitch_up_matrix(extrinsics.pitch_deg)
        @ _roll_matrix(extrinsics.roll_deg)
        @ _CAM_TO_UPRIGHT
    )


def pixel_to_camera_ray(
    u: float, v: float, intrinsics: CameraIntrinsics,
) -> Vector3:
    """Unit ray in camera frame for pixel (u, v) under a pinhole model."""
    half_h = np.tan(np.deg2rad(intrinsics.h_fov_deg) / 2)
    half_v = np.tan(np.deg2rad(intrinsics.v_fov_deg) / 2)
    fx = (intrinsics.width_px / 2) / half_h
    fy = (intrinsics.height_px / 2) / half_v
    ray = np.array(
        [
            (u - intrinsics.principal_x) / fx,
            (v - intrinsics.principal_y) / fy,
            1.0,
        ],
    )
    return ray / np.linalg.norm(ray)


def ray_to_az_el(ray: Vector3) -> tuple[float, float]:
    """World ray (ENU) -> (az in [0, 360), el in [-90, 90]) compass-style."""
    x, y, z = ray / np.linalg.norm(ray)
    az = float(np.rad2deg(np.arctan2(x, y)) % 360.0)
    el = float(np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0))))
    return az, el


def pixel_to_world_az_el(
    u: float,
    v: float,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    mount: MountPose,
) -> tuple[float, float]:
    """Full pipeline: pixel (u, v) on this camera -> world (az, el)."""
    cam_ray = pixel_to_camera_ray(u, v, intrinsics)
    mast_ray = camera_to_mast_rotation(extrinsics) @ cam_ray
    world_ray = _compass_yaw_matrix(mount.world_yaw_deg) @ mast_ray
    return ray_to_az_el(world_ray)
