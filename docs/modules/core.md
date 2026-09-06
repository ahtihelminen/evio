# Core Module (`src/evio/core`)

This document describes the foundational data structures and stream-processing
primitives under `src/evio/core`.

Files covered:

- `src/evio/core/events.py`
- `src/evio/core/window.py`
- `src/evio/core/pacer.py`
- `src/evio/core/streaming_pacer.py`
- `src/evio/core/recording.py`
- `src/evio/core/generate_events.py`
- `src/evio/core/__init__.py`

## What this module owns

`core` provides shared primitives used by IO, rendering, detection, and UI:

1. Event container types (`EventsBatch`, `BatchRange`).
2. Rolling time-window assembly over decoded event chunks.
3. Playback pacing against wall clock (with optional dropping).
4. Low-level recording loaders for DAT/RAW.
5. Frame-to-event synthesis helpers for mono-camera workflows and tests.

## Data model and time semantics

Most `core` APIs operate with:

- event time in microseconds (`int64` timestamps in batches),
- wall time from `time.perf_counter()` for pacing/HUD timing.

`core` keeps these domains separate and derives playback speed as:

- `event_elapsed / wall_elapsed`.

## File-by-file reference

### `events.py`

Defines the two canonical event-stream containers.

#### `EventsBatch`

Dataclass with parallel arrays:

- `timestamps: NDArray[int64]`
- `x_coords: NDArray[int16]`
- `y_coords: NDArray[int16]`
- `polarities: NDArray[int8]`

This is the standard batch payload across codecs, IO, detection, rendering,
and UI.

#### `BatchRange`

Named tuple for index ranges into a time-ordered stream:

- `start_idx`, `end_idx` (slice into sorted event order)
- `start_ts_us`, `end_ts_us` (best-effort bounds)

Used by DAT index-based playback (`DatFileSource` + `Pacer` path).

### `window.py`

Contains `RollingWindowBatchProcessor`, a stateful post-processor that converts
incoming decoded event batches into fixed-duration rolling windows.

Window behavior:

- left-aligned windows `[start_ts, start_ts + window_length)`,
- next window start advances by `step_us`,
- empty windows are skipped.

Internal strategy:

- keeps a `deque` of incoming chunks,
- drops old chunks/events before current window start,
- collects slices from one or more chunks into one output window,
- supports final flush (`process(..., final=True)`).

Current call sites:

- `src/evio/io/dat_file_mp.py`
- `src/evio/io/raw_file_mp.py`
- `src/evio/io/event_camera.py`

### `pacer.py`

Legacy/simple pacer for `BatchRange` iterables (index-window DAT playback).

Class: `Pacer`

- input: iterable of `BatchRange`
- target timing: based on each range `end_ts_us`
- optional behavior: if `force_speed=True` and stream is too far behind,
  drops overdue ranges.

Tracks:

- dropped/emitted counts,
- instantaneous and running average drop rates,
- anchor times (`_t_start` wall, `_e_start` event).

Primary call path:

- `scripts/play_dat.py`
- HUD rendering via `draw_hud(...)` in `src/evio/rendering/frame.py`.

### `streaming_pacer.py`

Main pacer for `EventsBatch` iterables used by most modern playback scripts.

Classes:

- `PacerConfig`: immutable config (`speed`, `force_speed`, tolerance).
- `PacerState`: mutable runtime state and stats.
- `StreamingPacer`: pacing engine.

Key differences vs `Pacer`:

- operates directly on `EventsBatch` instead of `BatchRange`.
- stores explicit HUD anchors (`hud_wall_start`, `hud_rec_start`).
- tracks instantaneous playback speed (`inst_speed`) and drop stats.
- includes bounded drop bursts and re-anchoring logic when badly behind.

HUD helpers:

- `wall_elapsed()`
- `recording_elapsed(end_time_us)`
- `actual_playback_speed(end_time_us)`

Primary call paths:

- `scripts/play_dat_mp.py`
- `scripts/play_raw_mp.py`
- `scripts/play_raw_folder.py`
- `scripts/event_stats.py`
- `scripts/bg_stats.py`
- HUD rendering via `draw_hud_stream(...)`.

### `recording.py`

Low-level file loaders returning a `Recording` object:

#### `Recording`

- `width`, `height`
- `timestamps` (sorted `int64`)
- `event_words` (packed `uint32` words)
- `order` (int32 stable argsort permutation)

#### `open_dat_dt8(...)`

- scans/uses DAT payload offset,
- memory-maps DT8 payload (`[(t32, w32)]`),
- returns sorted timestamps and corresponding order permutation.

Current direct use:

- `src/evio/io/dat_file.py`

#### `open_raw_evt3(...)`

- standalone EVT3 RAW decoder to `Recording` (CD events),
- reconstructs timestamps from `TIME_HIGH`/`TIME_LOW`,
- handles high-time wraparound,
- packs decoded x/y/polarity into `event_words`.

This path is currently not the main RAW playback path (RAW playback mainly uses
codec-based multiprocessing decoders).

### `generate_events.py`

Frame-difference-to-events synthesis (Numba-accelerated).

Pipeline:

1. `get_mean_change(...)`: global frame mean delta.
2. `get_event_count(...)`: count above/below threshold after mean removal.
3. `fill_event_arrays(...)`: materialize x/y/t/p arrays.
4. `get_events_from_frames(...)`: returns `EventsBatch`.

Used by:

- `src/evio/io/mono_camera.py` (`iter_events`).

This is useful for emulating event output from grayscale frame cameras.

### `__init__.py`

Currently empty (no package-level re-exports).

## How `core` interacts with other modules

### IO stack

- Codecs decode raw words to `EventsBatch`.
- `RollingWindowBatchProcessor` converts decoded chunks to rolling windows.
- `StreamingPacer`/`Pacer` pace those windows for playback.

### Rendering and HUD

- render helpers consume `BatchRange` or `EventsBatch` outputs,
- HUD text pulls drop/speed stats from pacer objects.

### Detection stack

- detection features/stages accept `EventsBatch`,
- no detection-specific logic is inside `core`.

## Practical notes

- `EventsBatch` assumes parallel arrays with matching lengths; upstream readers
  must preserve this invariant.
- Windowing assumes timestamp-monotonic chunks in normal operation.
- Pacer timing is sensitive to clock granularity and batching; instant speed is
  meaningful over multiple emitted windows, not single tiny batches.
- There are two pacing APIs (`Pacer` and `StreamingPacer`) because the project
  still supports both index-range and decoded-batch playback paths.



