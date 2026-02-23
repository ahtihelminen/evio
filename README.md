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
├─ AGENTS.md
├─ README.md
├─ pyproject.toml
├─ uv.lock
├─ data/              # sample recordings (*.dat)
├─ diagnostics/       # generated CSV diagnostics
├─ docs/              # design and pipeline notes
├─ notebooks/         # analysis notebooks
├─ scripts/           # runnable entry points
├─ src/evio/
│  ├─ codec/          # DT8 / EVT3 codec logic
│  ├─ core/           # events, windows, pacing, recording helpers
│  ├─ detection/      # features, background model, detection pipeline
│  ├─ io/             # file/camera readers and conversion
│  ├─ multiprocessing/# camera and worker process orchestration
│  ├─ rendering/      # frame, heatmap, panorama rendering
│  ├─ streaming/      # RTSP and control-plane utilities
│  └─ ui/             # PySide6 app, controllers, widgets, styles
└─ tests/             # test assets (currently minimal)
```

## Setup

```bash
uv sync
```

## Script run examples

Use `--help` with any script for full options.

### File playback and conversion

```bash
# DT8 DAT playback (single process)
uv run scripts/play_dat.py data/fred1.dat --fps 60 --window 10 --speed 1

# DT8 DAT playback (multiprocessing source)
uv run scripts/play_dat_mp.py data/fred1.dat --fps 60 --window 10 --speed 1

# EVT3 RAW playback (single RAW file)
uv run scripts/play_raw_mp.py path/to/events.raw --fps 60 --window 10 --speed 1

# EVT3 RAW playback (folder of RAW files)
uv run scripts/play_raw_folder.py path/to/raw_folder --fps 60 --window 10

# EVT3 RAW -> DT8 DAT conversion
uv run scripts/convert_raw_to_dat.py \
  --input path/to/input.raw \
  --output data/converted.dat
```

### Stats, diagnostics, and detection

```bash
# Grid feature heatmap/stat view over DAT playback
uv run scripts/event_stats.py data/fred1.dat --fps 60 --stat entropy

# Background anomaly scoring + optional diagnostics CSV
uv run scripts/bg_stats.py data/fred1.dat --fps 60 --diagnostics

# End-to-end detection pipeline MVP visualizer
uv run scripts/run_detection_pipeline_mvp.py data/fred1.dat --display-fps 30

# Synthetic anomaly/background scenario generator and player
uv run scripts/synthetic_background_test.py --total-windows 400 --playback-fps 30
```

### Live camera and panorama tools

```bash
# Single mono camera view (or event view with --event)
uv run scripts/play_live_mono.py --fps 60

# Multi-camera local panorama UI
uv run scripts/play_live_panorama.py --window-ms 10 --fps 60 --view-width 1920

# Stream live panorama to RTSP (requires mediamtx + ffmpeg)
uv run scripts/stream_live_panorama.py \
  --rtsp-url rtsp://127.0.0.1:8554/panorama \
  --fps 60

# View/control RTSP panorama stream
uv run scripts/play_stream.py --host-ip 127.0.0.1 --path panorama

# eMovie launcher UI (source selector for recording/RTSP)
uv run scripts/emovie.py
```

### Benchmarks

```bash
# Event generation benchmark from mono camera frames
uv run scripts/bench_event_generation.py --seconds 3 --threshold 8

# Numba vs numpy live render benchmark
uv run scripts/bench_live_render.py --method both --frames 300 --warmup 30
```

## Optional runtime dependencies
- Live camera scripts need supported hardware plus camera runtime libraries.
- RTSP streaming scripts need `mediamtx` and `ffmpeg` on `PATH`.
