# evio

Minimal library for standardized handling of event camera data, with file
playback, live capture, and panorama streaming tools.

## What it does
- Decode DT8 (.dat) and EVT3 (.raw) streams with Numba codecs
  (`src/evio/codec/*`).
- Read files with single-process and multiprocessing sources, then build
  fixed-duration windows (`src/evio/io/*`, `src/evio/core/*`).
- Render event frames, HUD overlays, and panoramas (`src/evio/rendering/*`).
- Capture live cameras via Metavision HAL and multiplex them into a panorama
  (`src/evio/io/camera.py`, `src/evio/streaming/camera_utils.py`).
- Stream panoramas over RTSP and control them via the PySide6 UI
  (`src/evio/streaming/stream.py`, `src/evio/ui/*`).
- Convert EVT3 RAW to DT8 DAT or NPZ with `EventDataConverter`
  (`src/evio/io/converter.py`).

## CLI scripts
- `play_dat.py` / `play_dat_mp.py`: DAT playback (single-process vs
  multiprocessing).
- `play_raw_mp.py` / `play_raw_folder.py`: EVT3 RAW playback (single file or
  folder).
- `play_live_panorama.py`: local multi-camera panorama UI.
- `stream_live_panorama.py`: capture local cameras and stream a panorama over
  RTSP.
- `play_stream.py`: PySide6 viewer/controller for an RTSP panorama stream.
- `convert_raw_to_dat.py` / `bench_live_render.py`: format conversion and live
  render benchmarks.

Run any script with `--help` for flags such as `--window`, `--speed`, and
`--force-speed`.

## Optional system dependencies
- Live camera capture requires `metavision_hal` and a supported camera.
- RTSP panorama streaming requires `mediamtx` and `ffmpeg` on your PATH.

## Project layout

```
.
├─ README.md
├─ pyproject.toml
├─ uv.lock
├─ data/
│  └─ fred1.dat
├─ scripts/
│  ├─ bench_live_render.py
│  ├─ convert_raw_to_dat.py
│  ├─ play_dat.py
│  ├─ play_dat_mp.py
│  ├─ play_live_panorama.py
│  ├─ play_raw_folder.py
│  ├─ play_raw_mp.py
│  ├─ play_stream.py
│  └─ stream_live_panorama.py
└─ src/
   └─ evio/
      ├─ codec/
      ├─ core/
      ├─ io/
      ├─ rendering/
      ├─ streaming/
      └─ ui/
```

## Quick start

```bash
# install dependencies (uv or pip)
uv sync

# play a DAT (DT8) file
uv run scripts/play_dat.py data/fred1.dat --window 10 --speed 1

# play DAT/RAW with multiprocessing
uv run scripts/play_dat_mp.py path/to/events.dat --window 10 --speed 1
uv run scripts/play_raw_mp.py path/to/events.raw --window 10 --speed 1

# play a folder of EVT3 RAW files
uv run scripts/play_raw_folder.py path/to/raws --window 10 --speed 1

# local panorama UI (requires Metavision HAL)
uv run scripts/play_live_panorama.py --window-ms 10 --view-width 1920

# stream a panorama over RTSP (requires mediamtx + ffmpeg)
uv run scripts/stream_live_panorama.py \
  --rtsp-url rtsp://127.0.0.1:8554/panorama
uv run scripts/play_stream.py --host-ip <host-ip> --path panorama
```

DAT/RAW playback defaults to 1280x720 geometry; adjust the source defaults in
`src/evio/io/*` (or the scripts) if your recordings differ.
