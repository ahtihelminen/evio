# evio

Toolkit for event-camera playback, live capture, panorama streaming, and
background/detection experiments.

## What this repo contains
- DT8 (`.dat`) and EVT3 (`.raw`) readers plus conversion tooling.
- Frame and panorama rendering utilities (OpenCV + Numba paths).
- Live local camera capture and RTSP panorama streaming tools.
- Detection/background pipeline components and diagnostics scripts.
- PySide6 desktop UIs for local and streamed visualization.

## Current repository structure

```text
.
├─ AGENTS.md / CLAUDE.md   # working conventions
├─ pyproject.toml, uv.lock, Makefile, SETUP.md
├─ config/            # calibration, site and LLR configs
├─ deploy/            # systemd units, netplan, dnsmasq for the mast NUC
├─ docs/              # module notes and design docs
├─ plans/             # design/work plans
├─ notebooks/         # analysis notebooks
├─ scripts/           # runnable entry points (_archive/ = retired)
├─ src/evio/
│  ├─ analysis/       # frame-metric instrumentation
│  ├─ aug/            # event-stream augmentation
│  ├─ bias_eval/      # flight scoring, range ladders, comb analysis
│  ├─ calibration/    # rig calibration capture, solver, Qt GUI
│  ├─ codec/          # DT8 / EVT3 codec logic
│  ├─ control/        # control plane, healthz, telemetry, web UI
│  ├─ core/           # events, windows, pacing, recording helpers
│  ├─ detection/      # pipelines, tracker, classifier, evaluation
│  ├─ edge/           # test-site edge-node data plane (WS egress)
│  ├─ focus/          # lens-focus metric and viewer
│  ├─ geometry/       # calibration IO, FOV, projection, geodesy
│  ├─ groundtruth/    # flight ground-truth ingest, labels, audit
│  ├─ io/             # file/camera readers and conversion
│  ├─ media/          # detection clip encode + HTTP serving
│  ├─ multiprocessing/# camera and worker process orchestration
│  ├─ ptz/            # Axis PTZ slew-to-cue follower
│  ├─ recording/      # hour-bucketed footage capture + recovery
│  ├─ rendering/      # frame, heatmap, panorama rendering
│  ├─ sapient/        # SAPIENT capability + detection mapper
│  ├─ streaming/      # RTSP, bias autotuner, ROI, camera health
│  ├─ time/           # clock discipline
│  └─ ui/             # PySide6 app, controllers, widgets, styles
└─ tests/
```

## Setup

See `SETUP.md` for the full sibling-repo bootstrap. Short version:

```bash
make install     # build the IMX636 driver, uv sync, link the .pth shim
```

## Script run examples

`scripts/README.md` documents each entry point; `--help` gives full
options. The ones you most likely want:

```bash
# The mast appliance: detect -> SAPIENT + panorama RTSP + web dashboard.
# systemd runs this in production (deploy/systemd/evio-mast.service).
uv run scripts/run_ops.py --calibration config/calibration.yaml \
    --footage-root /var/lib/evio/footage

# Live cameras with no detection and nothing drawn on top.
uv run scripts/stream_raw_cameras.py

# Play back one recording, no inference.
uv run scripts/play_dat.py /path/to/file.dat --fps 30 --window 50

# Same, through the live detection engine with tracker overlays.
uv run scripts/play_dat_with_detection.py /path/to/file.dat --fps 30 \
    --transformer-ckpt models/live_detect_v3.pt --device mps

# Qt viewer (recordings, RTSP panorama, or a remote mast).
uv run scripts/emovie.py

# Map physical cameras to serials during mast bring-up.
sudo uv run scripts/identify_event_cameras.py

# Talk to a deployed mast's control plane.
uv run scripts/evio_dev.py --host mast.lan:5006 status
```

Retired entry points live in `scripts/_archive/`.

## Optional runtime dependencies
- Live camera scripts need supported hardware plus camera runtime libraries.
- RTSP streaming scripts need `mediamtx` and `ffmpeg` on `PATH`.
