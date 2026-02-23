# Rendering Module (`src/evio/rendering`)

This document describes rendering utilities used across playback, streaming,
detection visualization, and UI controllers.

Files covered:

- `src/evio/rendering/frame.py`
- `src/evio/rendering/numba.py`
- `src/evio/rendering/panorama.py`
- `src/evio/rendering/heatmap.py`
- `src/evio/rendering/grid_overlay.py`
- `src/evio/rendering/liquid_logo.py`
- `src/evio/rendering/__init__.py`
- asset: `src/evio/rendering/senso_logo.png`

## What this module owns

The rendering package owns pixel-level visualization primitives:

1. Event-window to BGR frame rendering.
2. HUD overlays for playback pacing stats.
3. Heatmap construction and color mapping.
4. Grid and detection rectangle overlays.
5. Panorama strip assembly and circular viewport extraction.
6. Loading-screen visuals with animated liquid fill logo.

It does not own:

- event decoding/windowing/pacing logic (core/io),
- detection scoring logic,
- stream transport/networking logic.

## Shared conventions

- Frames are BGR `np.uint8` arrays (`height x width x 3`).
- Event windows are commonly represented as:
  - `(x_coords, y_coords, polarities_on)` tuple of NumPy arrays.
- Sensor defaults used in panorama/heatmap helpers:
  - `SENSOR_WIDTH_PX = 1280`
  - `SENSOR_HEIGHT_PX = 720`

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

#### `get_frame(window, width=..., height=..., ...)`

- builds base frame color,
- validates coordinates are inside frame bounds,
- paints ON/OFF polarity pixels with configured colors.

Used heavily by playback scripts, detection debug scripts, and UI playback
sources.

#### HUD overlays

- `draw_hud(frame, pacer, batch_range, ...)` for legacy `Pacer`.
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

Primary use:

- multicamera worker rendering (`src/evio/multiprocessing/multicamera.py`)
- benchmark script (`scripts/bench_live_render.py`).

### `panorama.py`

Panorama strip rendering and loading-screen visuals.

#### Constants

- `SENSOR_WIDTH_PX`, `SENSOR_HEIGHT_PX`
- `FRAME_CHANNELS`, `FRAME_SHAPE`
- `BASE_INTENSITY`

#### Panorama helpers

- `draw_zero_marker(strip, ...)` draws origin marker.
- `extract_circular_view(strip, start_x, view_width)` returns wrapped viewport.
- `render_panorama_view(frame_views, pan_offset, view_width, ...)`:
  - concatenates per-camera frames into one strip,
  - applies origin marker/label,
  - returns circularly wrapped viewport view.

#### Loading screen

- `render_loading_screen(...)`:
  - builds neutral background panel,
  - renders animated logo fill layer from `liquid_logo.py`,
  - overlays camera opening progress text.

#### `get_pan_offset(...)`

- small UDP-message helper for pan control updates.
- currently defined here but not used by current streaming/UI paths.

### `heatmap.py`

Heatmap conversion and range tracking utilities.

Key APIs:

- `HEATMAP_COLORMAPS` registry and `DEFAULT_HEATMAP_COLORMAP`.
- `resolve_colormap(...)` accepts string or OpenCV colormap integer.
- `get_feature_heatmap(values, min_value=None, max_value=None)`:
  - scales numeric values to 8-bit grayscale.
- `build_heatmap_color(...)`:
  - rescales to target frame size (default panorama sensor size),
  - applies OpenCV colormap to produce BGR heatmap.
- `update_global_range(values, prev_min, prev_max, alpha)`:
  - EMA-like range tracking across frames.
- `clip_percentiles(values, lower, upper)`:
  - robust clipping helper for noisy features.

Used by detection overlays, event/grid stats scripts, and stream-side grid
overlay rendering.

### `grid_overlay.py`

Grid and detection rectangle overlays on top of frames.

- `grid_line_positions(cfg)` computes cell boundary pixel lines.
- `draw_grid_overlay(frame, cfg)` draws region grid lines.
- `draw_detection_overlay(frame, cfg, z_scores, threshold=..., ...)` draws
  per-cell rectangles where score exceeds threshold.

Used by background/detection visualization paths and UI playback overlays.

### `liquid_logo.py`

Logo-mask loading and animated “liquid fill” effect used by loading screens.

#### `get_logo_mask(size)`

- loads `senso_logo.png` once and caches resized mask globally,
- prefers alpha channel when informative,
- otherwise computes binary mask from luminance + Otsu threshold.

#### `liquid_fill_layer(mask, frac, t)`

- builds animated wave surface,
- composites liquid + foam colors inside logo mask,
- returns BGR layer for overlay onto loading frame.

### `__init__.py`

Currently empty (no package-level re-exports).

## Integration points

### Playback scripts

- `scripts/play_dat.py`, `scripts/play_dat_mp.py`, `scripts/play_raw_mp.py`,
  `scripts/play_raw_folder.py`:
  - `get_frame`, `draw_hud`/`draw_hud_stream`.

### Detection visualizations

- `scripts/bg_stats.py`, `scripts/event_stats.py`,
  `src/evio/detection/pipeline/visualization.py`:
  - heatmaps and grid/detection overlays.

### Panorama streaming and local UI

- `scripts/stream_live_panorama.py` and `src/evio/ui/controllers/local_panorama.py`:
  - `render_panorama_view`, `render_loading_screen`.
- `src/evio/ui/controllers/rtsp_panorama.py`:
  - loading-screen rendering while stream is not ready.

### Worker-side rendering

- `src/evio/multiprocessing/multicamera.py`:
  - per-camera frame rendering via `NumbaEventRenderer`.

## Practical notes and limitations

- Rendering functions assume BGR ordering (OpenCV convention).
- `NumbaEventRenderer` uses the first `base_color` channel as scalar fill
  intensity; base is effectively grayscale even if tuple values differ.
- `build_heatmap_color` defaults to panorama sensor dimensions when width/height
  are omitted; callers rendering smaller/larger grids should pass explicit size.
- `liquid_logo.py` relies on a repo-relative logo path; missing asset raises
  runtime error during loading-screen rendering.

