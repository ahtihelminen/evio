"""Geodesy: observer + target geodetic coords -> azimuth / elevation / range.

Pure, scalar, dependency-free. Places a drone's GPS ground-truth position in
the appliance's local ENU frame so it can be projected into a camera by
:func:`evio.geometry.project.world_az_el_to_pixel` (the inverse of the
pixel -> world az/el pipeline).

Convention matches :func:`evio.geometry.project.ray_to_az_el`:
- azimuth: 0 = north, increases clockwise (compass), in [0, 360);
- elevation: 0 = horizon, +90 = zenith, in [-90, 90].

Altitudes are whatever the inputs use, but they must AGREE: the FlytBase
``position.height`` is MSL, so ``mount.altitude_m`` must also be the rig's MSL
elevation for the elevation angle to be correct (a wrong mount altitude tilts
the whole overlay — which is exactly the kind of calibration error the live
overlay surfaces).
"""

from __future__ import annotations

import math

# WGS84 ellipsoid.
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)


def _geodetic_to_ecef(
    lat_deg: float, lon_deg: float, alt_m: float,
) -> tuple[float, float, float]:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    slat, clat = math.sin(lat), math.cos(lat)
    slon, clon = math.sin(lon), math.cos(lon)
    n = _A / math.sqrt(1.0 - _E2 * slat * slat)
    x = (n + alt_m) * clat * clon
    y = (n + alt_m) * clat * slon
    z = (n * (1.0 - _E2) + alt_m) * slat
    return x, y, z


def geodetic_to_azel(
    observer_lat_deg: float,
    observer_lon_deg: float,
    observer_alt_m: float,
    target_lat_deg: float,
    target_lon_deg: float,
    target_alt_m: float,
) -> tuple[float, float, float]:
    """(az_deg, el_deg, range_m) of the target seen from the observer (ENU)."""
    ox, oy, oz = _geodetic_to_ecef(
        observer_lat_deg, observer_lon_deg, observer_alt_m,
    )
    tx, ty, tz = _geodetic_to_ecef(
        target_lat_deg, target_lon_deg, target_alt_m,
    )
    dx, dy, dz = tx - ox, ty - oy, tz - oz
    lat, lon = math.radians(observer_lat_deg), math.radians(observer_lon_deg)
    slat, clat = math.sin(lat), math.cos(lat)
    slon, clon = math.sin(lon), math.cos(lon)
    east = -slon * dx + clon * dy
    north = -slat * clon * dx - slat * slon * dy + clat * dz
    up = clat * clon * dx + clat * slon * dy + slat * dz
    rng = math.sqrt(east * east + north * north + up * up)
    az = math.degrees(math.atan2(east, north)) % 360.0
    el = math.degrees(math.atan2(up, math.hypot(east, north)))
    return az, el, rng


def _ecef_to_geodetic(
    x: float, y: float, z: float,
) -> tuple[float, float, float]:
    """ECEF -> (lat_deg, lon_deg, alt_m) via Bowring's closed form."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    b = _A * (1.0 - _F)
    ep2 = (_A * _A - b * b) / (b * b)
    theta = math.atan2(_A * z, b * p)
    lat = math.atan2(
        z + ep2 * b * math.sin(theta) ** 3,
        p - _E2 * _A * math.cos(theta) ** 3,
    )
    n = _A / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), alt


def azel_range_to_geodetic(
    observer_lat_deg: float,
    observer_lon_deg: float,
    observer_alt_m: float,
    az_deg: float,
    el_deg: float,
    range_m: float,
) -> tuple[float, float, float]:
    """(lat, lon, alt) of a target at (az, el, range) from the observer.

    Exact inverse of :func:`geodetic_to_azel` (ENU -> ECEF -> geodetic), so a
    round trip reproduces the inputs. Use it to turn a desired camera-frame
    sightline (azimuth + elevation + range) into a drone GPS waypoint.
    """
    az, el = math.radians(az_deg), math.radians(el_deg)
    east = range_m * math.cos(el) * math.sin(az)
    north = range_m * math.cos(el) * math.cos(az)
    up = range_m * math.sin(el)
    lat, lon = math.radians(observer_lat_deg), math.radians(observer_lon_deg)
    slat, clat = math.sin(lat), math.cos(lat)
    slon, clon = math.sin(lon), math.cos(lon)
    ox, oy, oz = _geodetic_to_ecef(
        observer_lat_deg, observer_lon_deg, observer_alt_m,
    )
    dx = -slon * east - slat * clon * north + clat * clon * up
    dy = clon * east - slat * slon * north + clat * slon * up
    dz = clat * north + slat * up
    return _ecef_to_geodetic(ox + dx, oy + dy, oz + dz)
