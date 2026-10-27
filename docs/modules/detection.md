# Detection Module (`src/evio/detection`)

This document describes the detection package as a whole: grid abstractions,
feature extraction, tracking and the track classifier/verdict stack. Detection
methods are composed into chains by `src/evio/dtc` (`docs/modules/dtc.md`).

Files and packages covered:

- `src/evio/detection/grid.py`, `feature_functions.py`, `features/*`
- `src/evio/detection/types.py`, `emitter.py`
- `src/evio/detection/persistence.py`, `spawn_density.py`
- `src/evio/detection/tracker.py`, `reid.py`
- `src/evio/detection/track_classifier.py`, `temporal_transformer*.py`,
  `openvino_transformer.py`, `event_tokens.py`
- `src/evio/detection/verdict_gate.py`, `kinematic_llr.py`, `rotor_llr.py`,
  `rotor_periodicity.py`, `irstd.py`
- `src/evio/detection/__init__.py`

## What this package owns

The detection package owns algorithms on top of `EventsBatch`:

1. Spatial/temporal discretization (`GridConfig`).
2. Per-cell feature extraction and a feature-name registry.
3. Clutter suppression, tracking and re-identification.
4. Per-track drone classification and verdict gating.

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

### Detection types and post-processing

- `types.py`: `Detection` (the single-window primitive: bbox, velocity,
  `m_coh`, `best_psr`, `z_score`, `camera_id`, `track_id`, optional
  `classification`, patch/embedding and score fields) and `Classification`
  (`label`, `confidence`).
- `emitter.py`: `DetectionEmitter` pub/sub hook each detection is emitted
  into; subscriber exceptions are logged and swallowed.
  `relay_with_camera_id` tags detections with their camera.
- `persistence.py`: `PersistenceMap`, per-cell hit-rate map that drops
  detections in persistent slow-moving clutter cells before the tracker.
- `spawn_density.py`: `SpawnDensityMap`, suppresses regions spawning too many
  new tracks in a rolling time window.

### Tracking

- `tracker.py`: `TrackManager`, a constant-velocity Kalman tracker. Run once
  per window; returns detections annotated with stable
  `track_id`s (ULIDs from `core.ids.new_ulid`) and a fit-scaled `m_coh`.
  Configured via `TrackManagerConfig`.
- `reid.py`: signature/embedding re-identification so a track that times out
  (occlusion, sun) can be recognised and its id reused; renders event patches
  with `rendering.frame.get_frame`.

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

## Public API shape and stability notes

- `src/evio/detection/__init__.py` is empty.
- Stable entry points are `GridConfig`, `types`, `tracker`,
  `track_classifier` and `verdict_gate`.

## Practical notes

- Grid dimensions/time bins directly control feature vector sizes and model
  state footprint.
- Many stage configs enforce strict validation in `__post_init__`; invalid
  runtime tuning values fail fast.
- Most computations assume timestamp-ordered batches.
