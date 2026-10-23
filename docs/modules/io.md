# IO Module (`src/evio/io`)

This document describes ingestion components under `src/evio/io`: DAT file
readers, multiprocessing decode infrastructure, and live camera sources.

Files covered:

- `src/evio/io/dat_file.py`
- `src/evio/io/dat_file_mp.py`
- `src/evio/io/mp_file.py`
- `src/evio/io/imx636_camera.py`
- `src/evio/io/camera_serial_report.py`
- `src/evio/io/mono_camera.py`
- `src/evio/io/mono_backends/ids_peak.py`
- `src/evio/io/mono_backends/u3v.py`
- `src/evio/io/__init__.py`

## What this module owns

The IO package owns how bytes or hardware streams become `EventsBatch`:

1. DAT file decoding sources (single-process and multiprocessing variants).
2. Generic reader/decoder worker orchestration for chunked files.
3. The live IMX636 event-camera source.
4. The mono frame-camera facade and its backends.

It does not own:

- codec internals (`src/evio/codec/*`),
- event container definitions/pacing/window logic (`src/evio/core/*`),
- rendering or detection logic.

## Architecture overview

There are three main IO paths.

### 1) DAT indexed playback path

`DatFileSource` in `dat_file.py`:

- uses `open_dat_dt8` from `core.recording` to load memmapped words plus sorted
  timestamp order,
- precomputes `BatchRange` windows by timestamp search,
- leaves event-word unpacking to downstream rendering code.

Used by `src/evio/focus/sources.py` (DAT replay panes).

### 2) Multiprocessing decode path (DAT)

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

`MultiprocessingDatFileSource` is a thin wrapper that configures:

- payload offset scanner (`find_dat_data_offset`),
- codec (`Dt8Codec`),
- rolling windowing (`RollingWindowBatchProcessor`).

Used by `tests/test_footage_recorder_replay_roundtrip.py`; in the ops repo, by
the appliance loop's replay mode.

### 3) Live camera path

`Imx636CameraSource` in `imx636_camera.py`:

- opens an IMX636 camera through the in-house `imx636` driver binding
  (libusb, no Metavision/openeb dependency),
- receives decoded CD events via a push-style callback,
- converts callback buffers to `EventsBatch`,
- windowizes with `RollingWindowBatchProcessor`.

`MonoCameraSource` (via the `mono_camera.py` facade):

- opens a mono frame camera through the selected backend,
- yields frames (`read(...)`) or synthesized events via frame differencing
  (`iter_events(...)` -> `core.generate_events.get_events_from_frames`).

Consumers in this repo: `src/evio/focus/sources.py` (IMX636) and
`src/evio/recording/mono_capture.py` (mono). Consumers in the ops repo: the
per-camera worker processes and the appliance loop.

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
  - `ranges()` iterator over `BatchRange`,
  - `__len__` (number of precomputed ranges).

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
- non-blocking polling (`get_batch_nowait() -> (batch | None, ended)`)
- `close()`
- context manager support.

Shutdown strategy:

- sets stop event,
- drains output queue best effort,
- terminates/joins child processes if alive,
- closes queues.

### `imx636_camera.py`

Live event-camera source using the in-house `imx636` driver.

Utilities:

- `get_camera_serials()` returns the sorted real (Treuzell) serial of each
  connected camera and prints the `camera_serial_report` warning when a
  USB-matched device returned no readable serial.

Class: `Imx636CameraSource`

Configuration:

- `serial`, `window_length_us` (default 10 ms), `step_us`,
- optional `anti_flicker` band-stop filter and `erc_event_rate` (Event Rate
  Controller cap), both off by default,
- `dem_mask_dir` (default `config/dem_masks/`): per-serial DEM mask manifest
  re-programmed at every open, since the on-sensor DEM is volatile; the
  manifest's software tier is exposed as `soft_mask_coords`.

Behavior:

- `__enter__` claims the device and runs sensor init; streaming starts lazily
  on the first `__next__()` so a multi-camera open lock is not held through
  `cam.start()`,
- `__next__` drains the driver pipe completely each call, converts every
  pending callback buffer, and returns the next rolling window (or an empty
  batch after one idle window so a static scene still ticks),
- `boot_scan_stuck_pixels(duration_s)` releases the DEM masks, counts the raw
  stream, re-masks current flooders and rewrites the manifest (helpers in
  `src/evio/streaming/stuck_pixel_scan.py`),
- `read_env()` samples lux, die temperature, pixel dead time and in-sensor ERC
  counters, skipping failed reads,
- `__exit__` stops streaming best-effort and drops the driver handle so the
  libusb claim is released.

### `camera_serial_report.py`

`describe_serial_read(serials)` returns a warning string when fewer devices
reported a serial than were USB-matched (an empty entry means present but not
fully initialized), or `None`. Kept separate from `imx636_camera.py` so the
decision is testable without the driver binding.

### `mono_camera.py`

Thin facade selecting the mono-camera backend from `EVIO_MONO_BACKEND`
(default `ids_peak`). `MonoCameraSource` and `get_mono_camera_serials` are
resolved lazily from `evio.io.mono_backends.<backend>`, so the facade imports
with no camera SDK installed.

### `mono_backends/ids_peak.py`

IDS Peak backend.

- `get_mono_camera_serials()`: sorted serials from the IDS device manager.
- `MonoCameraSource`: device selection by index or serial, configurable
  acquisition FPS/exposure/gain (+ auto modes), `read(...)` returns a Mono8
  frame (and optional timestamp), `iter_events(...)` synthesizes events from
  consecutive frames.

### `mono_backends/u3v.py`

Backend for the in-house `u3v` USB3 Vision driver (built by
`make build-externals`, not a pip dependency).

- `get_mono_camera_serials()`: filters `u3v.enumerate()` by
  `MONO_CAMERA_PID` (`0x8000`), since the same vendor id also matches event
  cameras.
- `MonoCameraSource`: same surface as the IDS backend (`read`, `iter_events`);
  `read` raises `TimeoutError` when no frame arrives in time.

### `__init__.py`

Empty (no package-level re-exports).

## Practical notes and limitations

- DAT sources default to `1280x720`; callers should override for other
  recorded geometries.
- Multiprocessing sources are best suited to continuous playback or polling
  loops; they do process/queue setup on first use.
- Live camera paths depend on external drivers/SDKs at runtime:
  - the `imx636` binding (`imx636_camera.py`),
  - IDS Peak or the `u3v` driver (`mono_backends/*`).

## Hardware tests

The `hardware` tier (`tests/hardware/`) runs the live camera paths against
real cameras. It covers the two supported camera types, IMX636 (event camera,
USB `1409:8e00`) and U3V (mono camera, USB `1409:8000`). IDS Peak is not
covered: it is deprecated and has no hardware test.

What it checks, for each camera type:

- recording: real events (IMX636) or frames (U3V) go through the footage
  recorder and read back intact (`test_imx636_recording.py`,
  `test_u3v_mono_recording.py`);
- crash recovery: a recorder SIGKILLed mid-write leaves footage that recovers
  (`test_crash_recovery.py`);
- unplug recovery: a camera unplugged mid-recording is given up within a
  timeout, keeps what it recorded, and records again after the reattach
  (`test_camera_unplug.py`);
- unplug helper: a software unplug hides one camera from enumeration and the
  reattach restores it (`test_usb_unplug.py`).

The crash and the unplug are done in software, so nobody has to touch the
rig; `docs/hardware-tests.md` describes both helpers.

### Prerequisites

- The drivers, built and linked into the venv (see the README's Setup):

  ```bash
  make drivers
  ```

- Write access to the camera's sysfs `authorized` attribute, which the
  unplug helper sets to `0` and back to `1`. It is root-owned, so install
  the udev rule in `docs/hardware-tests.md` ("Granting write access to
  `authorized`"): it makes the attribute group-writable by `plugdev` for
  vendor `1409`, which both camera types use. Add yourself to `plugdev`,
  log in again, and reload udev as that page shows. Without the access the
  unplug tests skip and name what is missing.

### Running

```bash
make test-hardware
```

On a unit with several identical IMX636 cameras, name the one to unplug by
its port (find it with `lsusb -t`); without it the IMX636 unplug tests skip
and list the matching ports:

```bash
EVIO_UNPLUG_PORT=2-1.1 make test-hardware
```

### Skipped tests

A skip means the machine lacks something the test needs, not that the test
passed. The camera fixtures in `tests/hardware/conftest.py` skip with the
reason: `imx636 driver not importable` or `u3v driver not importable` (run
`make drivers`), `no IMX636 camera enumerated` or `no U3V mono camera
enumerated` (no camera attached). With no camera attached every hardware test
skips, and `make test-hardware` still exits 0.

The laptop run never executes this tier: `make test` and `make check` select
`-m "not hardware"`, and `pyproject.toml` excludes the marker from a plain
pytest run too. Only `make test-hardware` selects it.
