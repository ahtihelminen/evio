# Detection Module (`src/evio/detection`)

This document describes the detection package as a whole: grid abstractions,
feature extraction, filtering transforms, background modeling primitives, and
the multi-stage detection/tracking pipeline.

Files and packages covered:

- `src/evio/detection/grid.py`
- `src/evio/detection/feature_functions.py`
- `src/evio/detection/transforms.py`
- `src/evio/detection/features/*`
- `src/evio/detection/background/*`
- `src/evio/detection/pipeline/*`
- `src/evio/detection/__init__.py`

## What this package owns

The detection package owns algorithms on top of `EventsBatch`:

1. Spatial/temporal discretization (`GridConfig`).
2. Per-cell feature extraction and registries.
3. Event-level denoising transforms (hot-pixel suppression).
4. Background/anomaly scoring models and utilities.
5. Multi-phase candidate generation, TBD scoring, and track attribution.

It does not own:

- file/camera ingestion (`src/evio/io/*`),
- playback timing (`src/evio/core/*`),
- frame rendering primitives (`src/evio/rendering/*`).

## Package structure

### `grid.py`

Core grid primitives used across almost every detection component.

- `GridConfig`: immutable geometry/time binning config.
  - validates positive dimensions and time parameters.
  - derives `cell_w`, `cell_h`, `dt_us`, `cells`, `cell_area`.
  - maps coordinates to flattened cell ids (`cell_id(...)`).
- `FeatureFn` protocol: standard callable signature for per-cell features.
- `GridState`: evaluates a dict of feature functions and stores results as
  attributes on each `update(events)` call.

`GridConfig` is the central contract between feature extraction, background
stages, pipeline stages, overlays, and UI/streaming control.

### `features/*`

Per-cell feature implementations.

`features/basic_features.py`:

- `cell_event_count`
- `cell_active_pixels`
- `cell_fano_factor`
- `cell_8_coactivity`
- `cell_polarity_skewness`

`features/entropy.py`:

- `cell_event_probabilities_axis`
- `cell_shannon_entropy_axis`
- `cell_shannon_entropy` (combined entropy norm)
- `cell_bernoulli_entropy`

Feature outputs are typically shape `(grid.cells,)` flattened arrays.

### `feature_functions.py`

Feature-name registry used by UI/streaming controls and multicamera workers.

- `FEATURE_FUNCTIONS: dict[str, Callable]`
- `DEFAULT_FEATURE_NAME`
- `get_feature_fn(name)` with fallback to default.

This enables selecting features by string through control messages.

### `transforms.py`

Batch-level filtering utilities.

- `filter_hot_pixels_fixed(events, max_events=...)`
- `filter_hot_pixels_percentile(events, percentile=...)`

Implementation strategy:

- builds unique pixel counts within a batch,
- removes all events from pixels above threshold.

Used in playback/detection scripts before scoring stages.

### `background/*`

Background model primitives used by phase-1 scoring and overlays.

- microstate construction (`microstate.py`)
- score/stat helpers (`metrics.py`)
- 2-class background clustering (`classes.py`)

Key consumers:

- `pipeline/dual_background.py` (the live sky/not-sky scorer)
- `pipeline/simplified_pipeline.py` (the default pipeline's class map)

### `pipeline/*`

Phase-oriented detection pipeline with explicit stage configs/results.

`pipeline/dual_background.py` (Phase 1):

- region-wise background scoring, one model per background class
- z-maps, update mask, calibration handling

`pipeline/candidate_stage.py` (Phase 2):

- thresholding and persistence history on z-map
- connected-components grouping on region grid
- ranking/selection of candidate ROIs

`pipeline/tbd_stage.py` (Phase 3):

- ROI-limited track-before-detect scoring
- motion hypothesis warping over buffered integration windows
- PSR baseline and coherence score (`m_coh`)

`pipeline/track_stage.py` (Phase 4):

- association of object-like candidates to tracks
- M-of-N confirmation and miss-based pruning
- candidate attribution:
  - `background_fluctuation`
  - `tentative_object`
  - `confirmed_object`

`pipeline/visualization.py`:

- 2x2 diagnostic canvas combining event frame, z-map, candidate/TBD, and
  attribution overlays.

`pipeline/__init__.py` re-exports stage types and visualization helpers.

## Pipeline data flow (current implementation)

`scripts/run_detection_pipeline_mvp.py` orchestrates:

1. optional hot-pixel filtering (`transforms.py`)
2. `BackgroundStage.update(batch)` -> `BackgroundStageResult`
3. `CandidateStage.update(result.z_map)` -> `CandidateStageResult`
4. `TbdStage.update(batch, candidate_result)` -> `TbdStageResult`
5. `TrackStage.update(tbd_result)` -> `TrackStageResult`
6. `render_background_phase_canvas(...)` for visualization

This represents the package’s primary end-to-end reference flow today.

## Other important integration points

### Statistics overlays in playback and streaming

- `scripts/event_stats.py` uses `GridConfig` + `GridState` + feature functions.
- `src/evio/multiprocessing/multicamera.py` builds grid stats per camera by
  feature name (via `get_feature_fn(...)`).
- `scripts/run_ops.py` manages feature-grid runtime state and selection in
  control-plane messages.

### Background-only diagnostics

- `scripts/bg_stats.py` runs `BackgroundStage` only and writes optional per-cell
  diagnostics CSV rows.

## Public API shape and stability notes

- `src/evio/detection/__init__.py` is currently empty.
- Stable entry points for pipeline users are effectively
  `src/evio/detection/pipeline/__init__.py` exports plus `GridConfig` and
  selected transforms/features.
- `features/__init__.py` is currently empty; call sites import feature modules
  directly.

## Practical notes

- Grid dimensions/time bins directly control feature vector sizes and model
  state footprint.
- Many stage configs enforce strict validation in `__post_init__`; invalid
  runtime tuning values fail fast.
- Most computations assume timestamp-ordered batches.
- The pipeline is CPU-first and designed for incremental updates over rolling
  windows.

## Suggested next docs to write

After this package-level overview, high-value focused docs are:

1. `docs/modules/detection-pipeline.md` for deeper stage internals.
2. `docs/modules/detection-grid.md` for grid contracts and shape math.
3. `docs/modules/detection-features.md` for exact feature definitions and
   normalization choices.

