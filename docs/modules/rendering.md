# Rendering Module (`src/evio/rendering`)

This document describes the pixel-level rendering utilities in this repo.

Files covered:

- `src/evio/rendering/frame.py`
- `src/evio/rendering/numba.py`
- `src/evio/rendering/overlay.py`
- `src/evio/rendering/__init__.py`

Panorama assembly, loading screens and the mono preview are OPS-unit display
concerns and live in the ops repo (the OPS-unit appliance), not here.

## What this module owns

1. Event-window to BGR frame rendering.
2. HUD overlays for playback pacing stats.
3. Verdict-coloured detection-box overlays.

It does not own:

- event decoding/windowing/pacing logic (core/io),
- detection scoring logic,
- stream transport/networking logic.

## Shared conventions

- Frames are BGR `np.uint8` arrays (`height x width x 3`).
- Event windows are commonly represented as:
  - `(x_coords, y_coords, polarities_on)` tuple of NumPy arrays.
- Event palette (pinned by `tests/test_rendering_palette.py`): green = ON
  (brighter), red = OFF (dimmer), black = still.

## File-by-file reference

### `frame.py`

Core event-frame helpers and HUD text overlays.

#### Packed-word window extractors

- `get_window_dt8(...)`
  - extracts x/y/polarity from DT8 packed event words using time-sorted index
    slices.
- `get_window_evt3(...)`
  - extracts x/y/polarity from EVT3-style packed words.

Both return `(x_coords, y_coords, polarity_bool_array)`.

#### `get_frame(window, width=1280, height=720, ...)`

- builds base frame color,
- drops coordinates outside frame bounds,
- paints ON/OFF polarity pixels with configured colors.

Used in this repo by `src/evio/detection/reid.py` (event patches for
re-identification).

#### HUD overlays

- `draw_hud(frame, pacer, batch_range, ...)` for `Pacer`.
- `draw_hud_stream(frame, pacer, end_ts_us, ...)` for `StreamingPacer`.

These write wall/recording time and speed/drop metrics directly on the frame.

### `numba.py`

Numba-accelerated event renderer for repeated high-rate frame generation.

Components:

- `fill_frame(...)` (`@njit`) clears frame to base intensity.
- `paint_events(...)` (`@njit`) applies ON/OFF event colors.
- `NumbaEventRenderer` class:
  - owns reusable frame buffer,
  - compiles kernels on init with tiny warmup arrays,
  - `render(...)` fills and paints each call.

Consumers in the ops repo: the per-camera worker processes and the panorama
renderer.

### `overlay.py`

Detection-box overlay drawing: one palette and one box-drawing routine for
the verdict-coloured boxes an operator sees. The label is read from
`Detection.classification`, which the verdict gate has already committed
upstream.

- Colours (BGR): `DRONE_COLOR` (yellow), `CLUTTER_COLOR` (amber),
  `UNKNOWN_COLOR` (grey, tracked but not yet classified). Pure green/red are
  left to ON/OFF event pixels.
- `box_style(detection)` returns the colour and label tag for one detection.
- `draw_detection_boxes(frame, detections)` draws bboxes + labels in place.
- `make_box_draw_fn(latest_getter)` builds a per-tile
  `draw_fn(serial, frame)` (`DrawFn`) from a latest-snapshot getter.

Consumers in the ops repo: the appliance loop, the web UI and the panorama
renderer.

### `__init__.py`

Empty (no package-level re-exports).

## Practical notes and limitations

- Rendering functions assume BGR ordering (OpenCV convention).
- `NumbaEventRenderer` uses the first `base_color` channel as scalar fill
  intensity; base is effectively grayscale even if tuple values differ.
