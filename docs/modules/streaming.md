# Streaming Module (`src/evio/streaming`)

This document describes the camera-stream utilities under
`src/evio/streaming`.

Files covered:

- `src/evio/streaming/stuck_pixel_scan.py`
- `src/evio/streaming/bias/constants.py`
- `src/evio/streaming/bias/types.py`
- `src/evio/streaming/bias/__init__.py`
- `src/evio/streaming/__init__.py`

## What this module owns

The streaming package owns the helpers that keep an IMX636 stream usable:

1. The stuck-pixel scan: counts per-pixel ON/OFF rates over a static-scene
   capture and nominates the fixed hot oscillators to mask in the DEM
   (`stuck_pixel_scan.py`).
2. The bias register constants (`bias/constants.py`).
3. The bias-control data types shared between bias modules (`bias/types.py`).

It does not own:

- opening the camera, streaming from it or programming the masks into the
  sensor (`src/evio/io/imx636_camera.py` calls into this package),
- the bias control loop itself: only its constants and types live here,
- the DEM mask manifest location or its per-serial policy.
