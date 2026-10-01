# Detection Module (`src/evio/detection`)

This document describes the detection package as a whole: grid abstractions,
feature extraction, background modeling, the detection pipeline, tracking and
the track classifier/verdict stack.

Files and packages covered:

- `src/evio/detection/grid.py`, `feature_functions.py`, `features/*`
- `src/evio/detection/background/*`
- `src/evio/detection/pipeline/*`
- `src/evio/detection/types.py`, `emitter.py`, `dedupe.py`
- `src/evio/detection/persistence.py`, `spawn_density.py`
- `src/evio/detection/tracker.py`, `reid.py`
- `src/evio/detection/live_camera.py`
- `src/evio/detection/track_classifier.py`, `temporal_transformer*.py`,
  `openvino_transformer.py`, `event_tokens.py`
- `src/evio/detection/verdict_gate.py`, `kinematic_llr.py`, `rotor_llr.py`,
  `rotor_periodicity.py`, `irstd.py`
- `src/evio/detection/__init__.py`

## What this package owns

The detection package owns algorithms on top of `EventsBatch`:

1. Spatial/temporal discretization (`GridConfig`).
2. Per-cell feature extraction and a feature-name registry.
3. Background classification and per-class anomaly scoring.
4. Candidate detection via track-before-detect (TBD) scoring.
5. Clutter suppression, tracking and re-identification.
6. Per-track drone classification and verdict gating.

It does not own:

- file/camera ingestion (`src/evio/io/*`),
- playback timing (`src/evio/core/*`),
- frame rendering primitives (`src/evio/rendering/*`).

## Package structure

### Grid and features

`grid.py`:

- `GridConfig`: immutable geometry/time binning config.
  - validates positive dimensions and time parameters.
  - derives `cell_w`, `cell_h`, `dt_us`, `cells`, `cell_area`.
  - maps coordinates to flattened cell ids (`cell_id(...)`).
- `FeatureFn` protocol: standard callable signature for per-cell features.
- `GridState`: evaluates a dict of feature functions and stores results as
  attributes on each `update(events)` call.

`features/basic_features.py`: `cell_event_count`, `cell_active_pixels`,
`cell_fano_factor`, `cell_8_coactivity`, `cell_polarity_skewness`.

`features/entropy.py`: `cell_shannon_entropy`,
`cell_event_probabilities_axis`, `cell_shannon_entropy_axis`,
`cell_bernoulli_entropy`.

Feature outputs are typically flattened `(grid.cells,)` arrays.
`features/__init__.py` is empty; call sites import feature modules directly.

`feature_functions.py` is the feature-name registry:

- `FEATURE_FUNCTIONS: dict[str, Callable]`
- `DEFAULT_FEATURE_NAME` (`"event_count"`)
- `get_feature_fn(name)` with fallback to the default.

This lets a feature be selected by string. Consumers in the ops repo: the
per-camera worker processes, which build grid stats per camera by name.

### `background/*`

Background model primitives:

- microstate construction (`microstate.py`, re-exported by
  `background/__init__.py`)
- score/stat helpers (`metrics.py`)
- 2-class (sky / not-sky) background clustering (`classes.py`,
  `BackgroundClassesClassifier`)

Key consumers: `pipeline/dual_background.py` and
`pipeline/simplified_pipeline.py`.

### `pipeline/*`

- `registry.py`: the `DetectionPipeline` protocol (anything with
  `.update(batch).detections`) and a name-keyed registry (`register`,
  `build`, `list_pipelines`, `PIPELINES`) so the active pipeline can be
  swapped by name. Registration happens at import of
  `evio.detection.pipeline`.
- `simplified_pipeline.py`: `SimplifiedDetectionPipeline`, the only
  registered pipeline (`"simplified"`, factory in `pipeline/__init__.py`).
- `dual_background.py`: `DualBackgroundScorer`, per-class background
  abnormality scoring with separate sky/not-sky models (z-maps, calibration).
- `simplified_tbd.py`: `SimplifiedTbdStage`, track-before-detect with
  integrated candidate detection from the z-map (motion-hypothesis velocity
  sweep, PSR baseline, coherence score `m_coh`).

`pipeline/__init__.py` re-exports the stage types, the registry API and
`Detection`/`Classification`.

### Detection types and post-processing

- `types.py`: `Detection` (the single-window primitive: bbox, velocity,
  `m_coh`, `best_psr`, `z_score`, `camera_id`, `track_id`, optional
  `classification`, patch/embedding and score fields) and `Classification`
  (`label`, `confidence`).
- `emitter.py`: `DetectionEmitter` pub/sub hook the pipeline emits each
  detection into; subscriber exceptions are logged and swallowed.
  `relay_with_camera_id` tags detections with their camera.
- `dedupe.py`: `deduplicate_by_iou` (greedy NMS by `m_coh`) and
  `top_k_by_mcoh`.
- `persistence.py`: `PersistenceMap`, per-cell hit-rate map that drops
  detections in persistent slow-moving clutter cells before the tracker.
- `spawn_density.py`: `SpawnDensityMap`, suppresses regions spawning too many
  new tracks in a rolling time window.

### Tracking

- `tracker.py`: `TrackManager`, a constant-velocity Kalman tracker. Run once
  per window after dedupe; returns detections annotated with stable
  `track_id`s (ULIDs from `core.ids.new_ulid`) and a fit-scaled `m_coh`.
  Configured via `TrackManagerConfig`.
- `reid.py`: signature/embedding re-identification so a track that times out
  (occlusion, sun) can be recognised and its id reused; renders event patches
  with `rendering.frame.get_frame`.

### Per-camera detection

`live_camera.py`: `CameraDetector`, torch-free detect -> track ->
patch/tokenize stage that can run inside a camera worker process. `update`
returns a `CameraDetectionResult` (boxes, recorder object-ids, and stacked
patches/events for tracks whose 16-frame ring just filled). `safe_update`
runs one tick and returns `None` on failure, so detection errors never crash
the worker.

### Classification

- `temporal_transformer.py`, `temporal_transformer_v2.py`,
  `temporal_transformer_v3.py`: per-track temporal transformer models (v2 adds
  auxiliary features and a bbox-mask head; v3 adds a raw-event branch).
- `event_tokens.py`: pure-numpy raw-event tokeniser for v3.
- `openvino_transformer.py`: OpenVINO FP16 backends and exporters for v2/v3.
- `track_classifier.py`: shared load + inference (`load_track_transformer`,
  `predict_drone_probs`, `TrackTransformerClassifier`). The transformer is
  optional at runtime: without torch or a checkpoint, loading returns a
  `None` bundle plus a `load_warning` and detection continues unclassified.
  An OpenVINO IR next to the `.pt` is preferred when present.

### Verdict gating

- `verdict_gate.py`: `PersistenceGate` (M-of-K, the default),
  `SequentialGate` (SPRT) and `FusionGate` (additive LLR arms), built by
  `build_verdict_gate(...)`.
- `kinematic_llr.py`: `KinematicLLR`, logistic `P(drone | kinematics)` arm.
- `rotor_llr.py`: `RotorLLR`, one-sided arm on rotor periodicity.
- `rotor_periodicity.py`: `RotorPeriodicityEstimator`, per-track
  blade-pass-frequency estimator producing `periodicity_score`.
- `irstd.py`: MPCM local-contrast re-scorer on the event patch
  (`irstd_features`).

## Pipeline data flow

`SimplifiedDetectionPipeline.update(batch)` runs:

1. `BackgroundClassesClassifier.update(batch)` -> sky/not-sky class map
   (calibrating until the class map exists).
2. `DualBackgroundScorer.update(batch, class_map)` -> `DualBackgroundResult`
   with the z-map.
3. `SimplifiedTbdStage.update(batch, z_map, class_map, ...)` with per-class
   thresholds -> detections.
4. Each detection is emitted through `DetectionEmitter`.

It returns a `SimplifiedPipelineResult` (`window_index`, `calibrating`,
`class_map`, `z_map`, `detections`, calibration progress, `bg_result`).

Downstream of the pipeline, consumers dedupe, apply the persistence and
spawn-density gates, track with `TrackManager`, classify with
`TrackTransformerClassifier` and commit verdicts through a verdict gate.
Consumers in the ops repo: the per-camera worker processes and the appliance
loop.

## Public API shape and stability notes

- `src/evio/detection/__init__.py` is empty.
- Stable entry points are `evio.detection.pipeline` exports plus `GridConfig`,
  `types`, `tracker`, `track_classifier` and `verdict_gate`.

## Practical notes

- Grid dimensions/time bins directly control feature vector sizes and model
  state footprint.
- Many stage configs enforce strict validation in `__post_init__`; invalid
  runtime tuning values fail fast.
- Most computations assume timestamp-ordered batches.
- The pipeline is CPU-first and designed for incremental updates over rolling
  windows; swapping pipelines resets calibration.
