# evioF

The field-deployable event-camera detection appliance. One program runs a
mast: read the IMX636 ring live, detect → track → classify, emit SAPIENT
detection reports, publish a panorama over RTSP, serve a browser
dashboard, and record footage to disk.

Its research sibling is **evioR**, which depends on this repo for its base
components. Anything that exists to *develop* detection methods —
training, evaluation, flight scoring, ground-truth ingest, rig
calibration, the Qt viewer, notebooks, the test-site edge integration —
lives there, not here. See `plans/repo_split_evioF_evioR.md`.

## What this repo contains

- IMX636 event-camera and IDS Peak mono-camera capture.
- The live detection chain: GMM-online / simplified pipeline → Kalman
  tracker → temporal-transformer classifier → verdict gate.
- The bias autotuner (v2 controller + the legacy adaptive SNR loop).
- SAPIENT egress, detection media clips, and the control plane
  (JSON-over-TCP, healthz, telemetry, browser UI).
- Hour-bucketed footage recording with crash recovery.
- Panorama rendering and RTSP publishing.
- Mast bring-up tools: camera identification, lens focus, stuck-pixel scan.

## Repository structure

```text
.
├─ AGENTS.md / CLAUDE.md   # working conventions
├─ pyproject.toml, uv.lock, Makefile
├─ config/            # calibration, site configs, DEM masks, LLR tables
├─ docs/modules/      # per-package notes
├─ plans/             # the evioF/evioR split plan
├─ scripts/           # the six field entry points + sync_footage.sh
├─ src/evio/
│  ├─ codec/          # DT8 codec
│  ├─ control/        # control plane, healthz, telemetry, web UI
│  ├─ core/           # events, windows, pacing, recording helpers
│  ├─ detection/      # pipelines, tracker, classifier, verdict gate
│  ├─ focus/          # lens-focus metric and viewer
│  ├─ geometry/       # calibration IO, FOV, projection, geodesy
│  ├─ io/             # IMX636 + IDS Peak sources, .dat readers
│  ├─ media/          # detection clip encode + HTTP serving
│  ├─ multiprocessing/# per-camera worker processes
│  ├─ recording/      # hour-bucketed footage capture + recovery
│  ├─ rendering/      # frame, panorama, overlay, MJPEG
│  ├─ sapient/        # SAPIENT capability + detection mapper
│  ├─ streaming/      # RTSP, bias autotuner, camera health
│  └─ time/           # clock discipline
└─ tests/
```

## Setup

Setup notes are being rewritten. With `sapient/` and `IMX636/` cloned as
siblings:

```bash
make install     # build the IMX636 driver, uv sync, link the .pth shim
```

## Running it

In production OpenRC owns this. By hand:

Config ships as templates `config/site.yaml.example` +
`config/rig.yaml.example`; copy each to `config/site.yaml` /
`config/rig.yaml` (gitignored, per-unit) and fill in the real serials.
`--calibration`
still takes a single combined YAML until consumer wiring for the two-file
layout lands (separate issue).

```bash
# The appliance: detect -> SAPIENT + panorama RTSP + web dashboard.
uv run scripts/run_ops.py --calibration combined-calibration.yaml \
    --footage-root /var/lib/evio/footage

# Panorama + dashboard only, no fusion link.
uv run scripts/run_ops.py --calibration combined-calibration.yaml --no-sapient

# Live cameras with no detection and nothing drawn on top.
uv run scripts/stream_raw_cameras.py

# Mast bring-up.
sudo uv run scripts/identify_event_cameras.py
uv run scripts/focus_meter.py
```

`scripts/README.md` documents each entry point; `--help` gives full options.

## Optional runtime dependencies

- Live capture needs the IMX636 driver built (`make build-externals`) and,
  for the mono camera, either the IDS Peak GenTL producer on
  `GENICAM_GENTL64_PATH` (default backend) or, with
  `EVIO_MONO_BACKEND=u3v`, the in-house U3V driver — also built by `make
  build-externals` when `U3V_DIR` (siblings-shaped by default) points at
  a checkout of it.
- RTSP publishing needs `mediamtx` and `ffmpeg` on `PATH`.
