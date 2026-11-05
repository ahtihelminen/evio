# Detection method index

One place to find every detection method in evio: what it is, and where it
lives. Longer descriptions are in `docs/modules/detection.md` and
`docs/modules/dtc.md`.

## How this index is maintained

The tables are written by hand, one row per method, in the form
`| name | kind | where | description |`. `tests/test_method_index.py` checks
them against the code and fails when:

- a method registered in `evio.dtc.registry` has no `dtc` row;
- a module in `src/evio/detection` has no `detection` row (the plumbing modules
  `__init__`, `emitter`, `feature_functions`, `grid` and `types` are listed in
  the test's `NOT_METHODS` and are not methods);
- a row names a method that is not registered, or a path that does not exist.

When you add, rename or remove a method, edit its row here in the same change.

## DTC chain methods

Registered with `@method(...)` and composed into chains by `src/evio/dtc`.
The stage (detect, track, classify or decide) is in each method's manifest.

| Method | Kind | Where | Description |
| --- | --- | --- | --- |
| `rate-excess` | dtc | `src/evio/dtc/methods/rate_excess.py` | Detect: blobs of events beyond the learned per-cell background. |
| `oracle-detect` | dtc | `src/evio/dtc/methods/oracle.py` | Detect: ground-truth detector, the step's labelled boxes at score 1.0. |
| `replay-detect` | dtc | `src/evio/dtc/methods/oracle.py` | Detect: re-emits recorded candidates at score 1.0. |
| `blob-tracker` | dtc | `src/evio/dtc/methods/blob_tracker.py` | Track: image-plane tracks of blobs, ranked by recent z, with re-acquisition. |
| `trackmanager` | dtc | `src/evio/dtc/methods/trackmanager.py` | Track: Kalman tracks with confirmation, coasting and optional re-id. |
| `oracle-track` | dtc | `src/evio/dtc/methods/oracle.py` | Track: ground-truth tracks, the id is the label. |
| `kinematic` | dtc | `src/evio/dtc/methods/kinematic.py` | Classify: drone or not from steady, straight, sustained motion. |
| `transformer` | dtc | `src/evio/dtc/methods/transformer.py` | Classify: temporal transformer drone classifier, batched across streams. |
| `verdict-gate` | dtc | `src/evio/dtc/methods/verdict_gate.py` | Decide: alarm on tracks whose verdicts pass the persistence or SPRT gate. |

## Detection package methods

Algorithm modules under `src/evio/detection`, used directly or wrapped by the
DTC methods above.

| Method | Kind | Where | Description |
| --- | --- | --- | --- |
| `persistence` | detection | `src/evio/detection/persistence.py` | Per-cell hit-rate map that drops detections in persistent slow-moving clutter. |
| `spawn_density` | detection | `src/evio/detection/spawn_density.py` | Suppresses regions that spawn too many new tracks in a rolling window. |
| `tracker` | detection | `src/evio/detection/tracker.py` | Constant-velocity Kalman tracker giving stable track ids. |
| `reid` | detection | `src/evio/detection/reid.py` | Signature re-identification so a timed-out track can reuse its id. |
| `temporal_transformer` | detection | `src/evio/detection/temporal_transformer.py` | Per-track temporal transformer with drone, re-id and validity heads. |
| `temporal_transformer_v2` | detection | `src/evio/detection/temporal_transformer_v2.py` | V2 transformer with auxiliary features and a bbox-mask head. |
| `temporal_transformer_v3` | detection | `src/evio/detection/temporal_transformer_v3.py` | V3 transformer with spatial and raw-event branches. |
| `event_tokens` | detection | `src/evio/detection/event_tokens.py` | Raw-event tokeniser feeding the v3 transformer. |
| `openvino_transformer` | detection | `src/evio/detection/openvino_transformer.py` | OpenVINO FP16 backends and exporters for the v2 and v3 transformers. |
| `track_classifier` | detection | `src/evio/detection/track_classifier.py` | Shared transformer load and inference, with a sticky per-track drone probability. |
| `verdict_gate` | detection | `src/evio/detection/verdict_gate.py` | Persistence (M-of-K), sequential (SPRT) and fusion verdict gates. |
| `kinematic_llr` | detection | `src/evio/detection/kinematic_llr.py` | Logistic P(drone given kinematics) arm for the fusion gate. |
| `rotor_llr` | detection | `src/evio/detection/rotor_llr.py` | One-sided calibrated rotor-periodicity arm for the fusion gate. |
| `rotor_periodicity` | detection | `src/evio/detection/rotor_periodicity.py` | Per-track blade-pass-frequency estimator giving a periodicity score. |
| `irstd` | detection | `src/evio/detection/irstd.py` | MPCM local-contrast re-scorer on the event patch. |
