# Geometry Module (`src/evio/geometry`)

This document describes the camera geometry code under `src/evio/geometry`.

Files covered:

- `src/evio/geometry/intrinsics.py`
- `src/evio/geometry/extrinsics.py`
- `src/evio/geometry/mount.py`
- `src/evio/geometry/project.py`
- `src/evio/geometry/geodesy.py`
- `src/evio/geometry/fov.py`
- `src/evio/geometry/calibration_io.py`
- `src/evio/geometry/rig_config.py`
- `src/evio/geometry/site_config.py`
- `src/evio/geometry/config_write.py`
- `src/evio/geometry/__init__.py`

## What this module owns

The geometry package owns how a camera pixel becomes a direction in the world,
and the files that carry the numbers for it:

1. Pinhole intrinsics, extrinsics in the mast frame and the mount pose.
2. The pure projection pipeline from pixel to world azimuth/elevation and back.
3. Geodesy: observer and target geodetic coordinates to azimuth, elevation and
   range.
4. Reducing a calibration to per-camera FoV cones.
5. The calibration schema: loading and saving the calibration YAML, the rig
   file (cameras) and the site file (mount), and validated writes of both.

It does not own:

- capturing the images or events that are projected (`src/evio/io/*`),
- detection or tracking in pixel space (`src/evio/detection/*`),
- the web pages that submit calibration, or publishing FoV cones to a backend,
- how the heading to true north is determined: `world_yaw_deg` is only a
  configurable slot.
