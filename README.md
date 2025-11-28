# evio

Lightweight toolbox for reading, slicing, and replaying event-camera recordings. Supports Prophesee DAT (DT8) and EVT3 RAW streams, can window them on the fly, pace playback, and render frames with an OpenCV HUD.

## What it does
- Decode DT8 (`codec/dt8.py`) and EVT3 (`codec/evt3.py`) event streams with Numba kernels.
- Stream files via a generic multiprocessing reader (`source/mp_file.py`) that pairs a codec with a batch processor.
- Build fixed-duration windows for offline (`core/pacer.py`) and streaming (`core/streaming_pacer.py`) playback.
- Render batches to RGB frames with timestamp/drop overlays (`rendering/frame.py`).
- Ship runnable viewers: single-process DAT player and multiprocessing DAT/RAW players.

## CLI scripts
- `scripts/play_dat.py`: single-process DAT player using precomputed windows.
- `scripts/play_dat_mp.py`: multiprocessing DAT (DT8) streamer with windowing.
- `scripts/play_raw_mp.py`: multiprocessing EVT3 RAW streamer with windowing.

Run any script with `--help` for flags such as `--window`, `--speed`, and `--force-speed`.

## Project layout

```
.
├─ README.md
├─ pyproject.toml
├─ uv.lock
├─ data/
│  └─ fred1.dat
├─ scripts/
│  ├─ play_dat.py
│  ├─ play_dat_mp.py
│  └─ play_raw_mp.py
└─ src/
   └─ evio/
      ├─ __init__.py
      ├─ codec/
      │  ├─ __init__.py
      │  ├─ dt8.py
      │  ├─ evt3.py
      │  └─ protocol.py
      ├─ core/
      │  ├─ __init__.py
      │  ├─ events.py
      │  ├─ index_scheduler.py
      │  ├─ pacer.py
      │  ├─ recording.py
      │  ├─ streaming_pacer.py
      │  └─ window.py
      ├─ rendering/
      │  ├─ __init__.py
      │  └─ frame.py
      └─ source/
         ├─ __init__.py
         ├─ dat_file.py
         ├─ dat_file_mp.py
         ├─ mp_file.py
         └─ raw_file_mp.py
```

## Quick start

```bash
# install dependencies (uv or pip)
uv sync

# play a DAT (DT8) file
uv run scripts/play_dat.py path/to/events.dat --window 10 --speed 1

# play a DAT file with multiprocessing decode + windowing
uv run scripts/play_dat_mp.py path/to/events.dat --window 10 --speed 1

# play an EVT3 RAW file with multiprocessing decode + windowing
uv run scripts/play_raw_mp.py path/to/events.raw --window 10 --speed 1
```

The DAT players assume 1280x720 geometry by default; override width/height in the sources if needed.
