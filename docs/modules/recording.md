# Recording Module (`src/evio/recording`)

This document describes the footage capture code under `src/evio/recording`.
The labelled recording file format is specified separately in
[`labelled-recording.md`](labelled-recording.md).

Files covered:

- `src/evio/recording/footage_recorder.py`
- `src/evio/recording/mono_footage_recorder.py`
- `src/evio/recording/mono_capture.py`
- `src/evio/recording/manifest.py`
- `src/evio/recording/recovery.py`
- `src/evio/recording/footage_admin.py`
- `src/evio/recording/labelled_recording.py`
- `src/evio/recording/__init__.py`

## What this module owns

The recording package owns what is written to disk and how it is found again:

1. `FootageRecorder`: tees `EventsBatch` into hour-bucketed DT8 `.dat` files.
2. `MonoFootageRecorder`: tees mono-camera frames into hour-bucketed `.mraw`
   files, and the capture thread that feeds it (`mono_capture.py`).
3. The per-day JSONL manifest of closed files (`manifest.py`).
4. Boot-time recovery of files that never got a manifest entry
   (`recovery.py`).
5. Listing and deleting footage a day at a time (`footage_admin.py`).
6. The writer and reader of the labelled recording HDF5 file
   (`labelled_recording.py`).

It does not own:

- the cameras and their sources (`src/evio/io/*`),
- the DT8 codec (`src/evio/codec/*`),
- retention policy, uploading or training-data fetching: those read the
  manifest but live elsewhere,
- the labelled recording format specification (`labelled-recording.md`).
