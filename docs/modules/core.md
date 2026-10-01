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
- `src/evio/core/ids.py`
- `src/evio/core/__init__.py`

## What this module owns

`core` provides shared primitives used by IO, rendering, detection, focus and
recording:

1. Event container types (`EventsBatch`, `BatchRange`).
2. Rolling time-window assembly over decoded event chunks.
3. Playback pacing against wall clock (with optional dropping).
4. A low-level DAT (DT8) recording loader.
5. Frame-to-event synthesis for mono-camera workflows and tests.
6. ULID generation for track identifiers.

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

This is the standard batch payload across codecs, IO, detection, rendering
and recording.

#### `BatchRange`

Named tuple for index ranges into a time-ordered stream:

- `start_idx`, `end_idx` (slice into sorted event order)
- `start_ts_us`, `end_ts_us` (best-effort bounds)

Used by DAT index-based playback (`DatFileSource` + `Pacer` path).

### `window.py`

#### `RollingWindowBatchProcessor`

A stateful post-processor that converts incoming decoded event batches into
fixed-duration rolling windows.

Window behavior:

- left-aligned windows `[start_ts, start_ts + window_length)`,
- next window start advances by `step_us`,
- empty windows are skipped.

Internal strategy:

- keeps a `deque` of incoming chunks,
- drops old chunks/events before current window start,
- collects slices from one or more chunks into one output window,
- supports final flush (`process(..., final=True)`).

Call sites in this repo:

- `src/evio/io/dat_file_mp.py`
- `src/evio/io/imx636_camera.py`

#### `drain_window_backlog(window_queue, current, max_backlog)`

Collapses a backed-up window queue to its freshest window when a consumer
cannot keep up. Returns the newest window, the stale windows the caller must
still record (so footage stays lossless) and their summed event count. It is
a no-op until the queue exceeds `max_backlog`; `max_backlog <= 0` disables it.

Used by `src/evio/focus/sources.py`; in the ops repo the per-camera worker
processes use it on the live camera queue.

### `pacer.py`

Simple pacer for `BatchRange` iterables (index-window DAT playback).

Class: `Pacer`

- input: iterable of `BatchRange`
- target timing: based on each range `end_ts_us`
- optional behavior: if `force_speed=True` and the stream is further behind
  than `drop_tolerance_s`, drops overdue ranges.

Tracks:

- dropped/emitted counts,
- instantaneous and running average drop rates,
- anchor times (`_t_start` wall, `_e_start` event).

HUD rendering for it is `draw_hud(...)` in `src/evio/rendering/frame.py`.

### `streaming_pacer.py`

Pacer for `EventsBatch` iterables.

Classes:

- `PacerConfig`: immutable config (`speed`, `force_speed`, `drop_tolerance`).
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

HUD rendering for it is `draw_hud_stream(...)` in
`src/evio/rendering/frame.py`.

### `recording.py`

Low-level DAT (DT8) file access.

#### `Recording`

- `width`, `height`
- `timestamps` (sorted `int64`)
- `event_words` (packed `uint32` words)
- `order` (int32 stable argsort permutation)

#### `memmap_dat_dt8(path, *, data_offset=None)`

- skips the `%` header lines and the 2-byte type/size marker (unless
  `data_offset` is given),
- returns a lazy `[(t32, w32)]` memmap; event data pages in only as indexed.

#### `open_dat_dt8(path, *, width, height, data_offset=None)`

- memmaps the payload via `memmap_dat_dt8`,
- unwraps the 32-bit `t32` counter (wraps every 2**32 µs, ~71.6 min) into a
  monotonic timeline,
- returns sorted timestamps and the corresponding order permutation.

Direct use: `src/evio/io/dat_file.py`.

### `generate_events.py`

Frame-difference-to-events synthesis (Numba-accelerated).

Pipeline:

1. `get_mean_change(...)`: global frame mean delta.
2. `get_event_count(...)`: count above/below threshold after mean removal.
3. `fill_event_arrays(...)`: materialize x/y/t/p arrays.
4. `get_events_from_frames(...)`: returns `EventsBatch`.

Used by the mono-camera backends (`src/evio/io/mono_backends/ids_peak.py`,
`src/evio/io/mono_backends/u3v.py`) in `iter_events`.

### `ids.py`

`new_ulid()` returns a ULID (48-bit ms timestamp + 80 random bits, Crockford
base32), byte-for-byte the SAPIENT library's generator so the detection chain
needs no SAPIENT dependency. `src/evio/detection/tracker.py` uses it for
`track_id`s, which become SAPIENT `object_id`s downstream.

### `__init__.py`

Empty (no package-level re-exports).

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
- There are two pacing APIs (`Pacer` and `StreamingPacer`) because both
  index-range and decoded-batch playback paths are supported.
