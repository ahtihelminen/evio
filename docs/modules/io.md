# IO Module (`src/evio/io`)

This document describes ingestion and export components under
`src/evio/io`: file readers, multiprocessing decode infrastructure, live
camera sources, and conversion utilities.

Files covered:

- `src/evio/io/dat_file.py`
- `src/evio/io/dat_file_mp.py`
- `src/evio/io/raw_file_mp.py`
- `src/evio/io/raw_folder_mp.py`
- `src/evio/io/mp_file.py`
- `src/evio/io/event_camera.py`
- `src/evio/io/mono_camera.py`
- `src/evio/io/converter.py`
- `src/evio/io/__init__.py`

## What this module owns

The IO package owns how bytes or hardware streams become `EventsBatch`:

1. DAT/RAW file decoding sources (single-process and multiprocessing variants).
2. Generic reader/decoder worker orchestration for chunked files.
3. Live camera adapters for event cameras and mono frame cameras.
4. Export/conversion APIs (`.dat`, `.npz`, range/ROI subsets).

It does not own:

- codec internals (`src/evio/codec/*`),
- event container definitions/pacing/window logic (`src/evio/core/*`),
- rendering or detection logic.

## Architecture overview

There are three main IO paths:

### 1) DAT indexed playback path

`DatFileSource` in `dat_file.py`:

- uses `open_dat_dt8` from `core.recording` to load memmapped words plus sorted
  timestamp order,
- precomputes `BatchRange` windows by timestamp search,
- leaves event-word unpacking to downstream rendering code.

Used by: `scripts/play_dat.py`.

### 2) Multiprocessing decode path (DAT/RAW)

`MultiprocessingFileSource` in `mp_file.py` is the generic engine:

- reader process:
  - seeks to payload offset,
  - reads fixed-size chunks of words,
  - pushes raw word arrays to input queue.
- worker process:
  - instantiates codec,
  - decodes chunks to `EventsBatch`,
  - optionally post-processes via a `BatchProcessor`,
  - pushes batches to output queue.

`MultiprocessingDatFileSource` and `MultiprocessingRawFileSource` are thin
wrappers that configure:

- payload offset scanner (`find_dat_data_offset` / `find_data_offset`),
- codec (`Dt8Codec` / `Evt3Codec`),
- rolling windowing (`RollingWindowBatchProcessor`).

Used by:

- `scripts/play_dat_mp.py`
- `scripts/play_raw_mp.py`
- `scripts/event_stats.py`
- `scripts/bg_stats.py`
- `scripts/run_detection_pipeline_mvp.py`
- UI playback source (`src/evio/ui/sources/dat_playback.py`).

### 3) Live camera path

`EventCameraSource` in `event_camera.py`:

- opens Metavision HAL device,
- receives decoded CD events via callback,
- converts callbacks to `EventsBatch`,
- windowizes with `RollingWindowBatchProcessor`.

`MonoCameraSource` in `mono_camera.py`:

- opens IDS Peak frame camera,
- yields frames (`read(...)`) or synthesized events via frame differencing
  (`iter_events(...)` -> `core.generate_events.get_events_from_frames`).

Used by:

- `scripts/run_ops.py` (mono footage capture)
- `scripts/stream_raw_cameras.py`
- multicamera worker (`src/evio/multiprocessing/multicamera.py`).

## File-by-file reference

### `dat_file.py`

Class: `DatFileSource`

Purpose:

- lightweight DAT source for index-range playback.

Behavior:

- precomputes `[start, stop)` index ranges from sorted timestamps,
- exposes:
  - `event_words` (packed `uint32`),
  - `order` (sorted permutation),
  - `width`, `height`,
  - `ranges()` iterator over `BatchRange`.

No decoding to x/y/p arrays happens here.

### `dat_file_mp.py`

Class: `MultiprocessingDatFileSource`

Purpose:

- DT8 DAT decode + rolling windows using `MultiprocessingFileSource`.

Behavior:

- validates path exists,
- configures codec as `Dt8Codec(width, height)`,
- applies `RollingWindowBatchProcessor(window_length, step_us)`,
- delegates iteration/polling/context-manager to inner source.

### `raw_file_mp.py`

Class: `MultiprocessingRawFileSource`

Purpose:

- EVT3 RAW decode + rolling windows via same multiprocessing engine.

Behavior:

- computes `chunk_words` from `chunk_bytes`,
- uses `Evt3Codec` and RAW payload offset scanner,
- applies rolling window processor,
- delegates to inner `MultiprocessingFileSource`.

Note:

- defines `__len__` returning `0` (placeholder; no precomputed length).

### `raw_folder_mp.py`

Class: `MultiprocessingRawFolderSource`

Purpose:

- sequence multiple RAW files as one continuous timestamp stream.

Behavior:

- opens one `MultiprocessingRawFileSource` at a time,
- on each file switch, computes timestamp shift:
  - first batch of new file is shifted so it starts at `last_emitted + 1`,
- exposes non-blocking polling:
  - `get_batch_nowait() -> (batch | None, ended)`.

Current use:

- UI playback source for RAW folders (`dat_playback.py`).

Note:

- this class is poll-oriented; it does not provide `__iter__`.

### `mp_file.py`

Generic multiprocessing reader/decoder implementation.

Key types:

- `BatchProcessor` protocol: optional post-decoder stateful processor.
- `_IdentityBatchProcessor`: pass-through default.
- `MultiprocessingFileSource`: public source class.

Sentinel handling:

- uses a custom pickle-safe module sentinel for EOS signaling across queues.

Public API:

- iteration (`for batch in source`)
- non-blocking polling (`get_batch_nowait`)
- `close()`
- context manager support.

Shutdown strategy:

- sets stop event,
- drains output queue best effort,
- terminates/joins child processes if alive,
- closes queues.

### `event_camera.py`

Live event-camera source using Metavision HAL.

Utilities:

- `get_camera_serials()` parses local device IDs and returns sorted serials.

Class: `EventCameraSource`

Behavior:

- optional `initial_biases` applied before stream start,
- callback captures CD event buffers into a deque,
- `__next__` pumps HAL buffers, converts structured arrays to `EventsBatch`,
  and yields rolling windows.

Important details:

- window processor is rebuilt if `window_length_us` / `step_us` change.
- output geometry defaults to `1280x720` but is read from camera geometry.

### `mono_camera.py`

Frame-camera source using IDS Peak APIs.

Class: `MonoCameraSource`

Behavior:

- device selection by index or serial,
- configurable acquisition FPS/exposure/gain (+ auto modes),
- `read(...)` returns Mono8 frame (and optional timestamp),
- `iter_events(...)` synthesizes events from consecutive frames using
  thresholded differencing.

Used for:

- local mono camera viewer,
- event-generation benchmarking/prototyping.

### `converter.py`

Class: `EventDataConverter`

Purpose:

- stream-decode DAT/RAW and export full or partial data.

Core methods:

- streaming:
  - `__iter__()` yields decoded batches.
  - `iter_range(start, stop, roi_bounds=None)` yields bounded events.
  - `count_range_events(...)`.
- materialization:
  - `to_arrays()`
  - `to_npz(output_path)`
  - `to_dat(output_path, header=None)`
- range export:
  - `to_dat_range(...)`
  - `to_npz_range(...)`.

Input format selection:

- `.dat` -> DT8 codec + DAT offset parser
- `.raw` -> EVT3 codec + RAW offset parser

Header handling:

- DAT output headers are normalized to include DT8 marker bytes.

Range/ROI behavior:

- timestamp interval is `[start, stop)`,
- ROI bounds are normalized to ordered corners,
- per-batch masks are applied before writing.

Memory behavior:

- full-stream `to_arrays()` buffers in RAM,
- `to_npz_range()` uses temporary disk-backed memmaps to avoid full in-memory
  concatenation for large range exports.

### `__init__.py`

Currently empty (no package-level re-exports).

## Integration with UI playback/export

`src/evio/ui/sources/dat_playback.py` is the primary integration hub:

- loads DAT/RAW/RAW-folder sources,
- computes recording bounds (format-specific),
- supports seek/reload behavior using non-blocking source polling,
- exports selected segments via `EventDataConverter`,
- has dedicated RAW-folder export path that stitches multiple files with
  timestamp continuity and optional ROI filtering.

## Practical notes and limitations

- Many sources default to `1280x720`; callers should override for non-default
  sensors or recorded geometry assumptions.
- Multiprocessing sources are best suited to continuous playback or polling
  loops; they do process/queue setup on first use.
- `iter_range` in converter assumes source timestamps are time-ordered for
  efficient early skip/stop behavior.
- Live camera paths depend on external SDK/runtime availability:
  - Metavision HAL (`event_camera.py`)
  - IDS Peak (`mono_camera.py`).


