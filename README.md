# evio

Lightweight toolbox for reading, slicing, and replaying event-camera recordings. It handles DAT (DT8) and EVT3 RAW files, builds fixed-duration windows, paces playback to a target speed, and renders frames with an OpenCV HUD.

## What it can do
- Load DAT (DT8) files into a `Recording` with stable timestamp order.
- Decode EVT3 RAW streams (Numba) and optionally stream them via multiprocessing without blocking the main process.
- Build fixed-duration time windows (`BatchRange` / `StreamingBatch`) for downstream processing or playback.
- Pace offline or streaming batches to a requested speed, tracking drop rates and effective speed.
- Render event windows into RGB frames with overlays for wall/recording time and drop statistics.
- Ship with small scripts: play DAT, play EVT3 RAW (single-process), play EVT3 RAW with multiprocessing, and a RAW decoder helper.

## Repository Structure

```
.
├─ README.md
├─ pyproject.toml
├─ uv.lock
├─ data/
│  └─ fred1.dat
├─ scripts/
│  ├─ play_dat.py
│  ├─ play_raw.py
│  ├─ play_raw_mp.py
│  └─ read_raw_evt3.py
└─ src/
   └─ evio/
      ├─ __init__.py
      ├─ codec/
      │  ├─ __init__.py
      │  └─ evt3.py
      ├─ core/
      │  ├─ __init__.py
      │  ├─ index_scheduler.py
      │  ├─ pacer.py
      │  ├─ recording.py
      │  ├─ streaming_pacer.py
      │  ├─ utils.py
      │  └─ window.py
      ├─ rendering/
      │  ├─ __init__.py
      │  └─ frame.py
      └─ source/
         ├─ __init__.py
         ├─ dat_file.py
         ├─ raw_file.py
         └─ raw_file_mp.py
```

## Quick start

```bash
# install in editable mode (using uv or pip)
uv sync

# play back the sample DAT recording (real-time by default)
uv run scripts/play_dat.py path/to/dat --window 10 --speed 1

# play an EVT3 RAW file (single-process or multiprocessing streamer)
uv run scripts/play_raw.py path/to/file.raw --window 10 --speed 1
uv run scripts/play_raw_mp.py path/to/file.raw --window 10 --speed 1
```
