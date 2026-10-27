# DTC pipeline architecture

**Status:** evio side implemented (rollout steps 0 and 1) on
`feat/dtc-architecture`; ops and evioR ports (steps 2 and 3) and the release
(step 4) are not done. Written 2026-10-14. See `docs/modules/dtc.md`.
**Scope:** how evio hosts detect, track, classify (DTC) pipelines so that
methods can be swapped, run live or offline, and compared. Methods themselves
(beyond a small fallback set) live in evioR.

## 1. Principle

A pipeline is a deterministic, causal state machine over time windows.
Placement, timing, errors and scoring are done to that machine from outside,
so a method stays small and every cross-cutting concern is written once.

## 2. Clean-slate design

### 2.1 Separate typed outputs, joined by track id

- A detector emits `Candidates`, a tracker emits `Tracks`, a classifier emits
  `Verdicts`, and a decision policy turns verdicts into alarms. Each stage
  sees the previous stage's output, not a type that has grown the fields of
  every later stage.
- A method that does all three stages in one is a single object with the same
  pipeline interface. A detector-only or tracker-only chain is the same shape
  with fewer outputs.
- The tracker can feed priors (such as predicted regions) back to the detector
  on the next step, because real chains use them. More generally any stage
  may declare that it reads the chain's outputs of the last step (feedback),
  and any stage may read cues from outside the chain, both on the `Step`;
  none has to. Within a step the chain stays linear; feedback only crosses
  step boundaries, so it is no general DAG.

### 2.2 Causal, driven by sensor time

- A step takes a `Step`: one or more aligned `Window`s (events, time span,
  coordinate-space extent, an opaque stream key; see 2.9) and never reads the
  wall clock. Replay is the same code path as live.
- Same inputs give the same outputs; seeds live in the config.
- State is explicit and resettable. "Calibrating" is a status in the output,
  not a hidden mode.

### 2.3 Methods declare a manifest

- Stages implemented, requirements on the source's facts (see 2.9), compute
  class (CPU, GPU, NPU), dependencies, per-window cost class and cadence.
- The builder checks it before any data flows: stage types line up, the device
  and dependencies exist, the source's facts cover the method's requirements. A mismatch fails at build, not at
  window 10,000.
- The manifest also drives discovery and selective install.

### 2.4 Evidence is requested, not baked in

Classifiers need different views of a track: a 16x32x32 patch ring, raw event
tokens, events binned at 0.5 ms, kinematic features. A classifier declares what
it needs; the runtime computes each view once and shares it. This replaces
patches and embeddings riding on the detection object.

### 2.5 Logic is separate from placement

- A method is a pure per-stream state machine. The runtime decides where it
  runs: in process, in a per-stream worker, or in a shared service.
- Stage outputs are plain serializable data, so any stage boundary can cross a
  process.
- Classifiers also get a batch form, so one model serves all streams and scores
  many tracks at once.
- Classifier cadence (every N windows) belongs to the runtime. A verdict states
  its age instead of silently sticking.

### 2.6 Visibility is built in, not per method

- **Timing.** The runtime wraps every stage and records wall and CPU time per
  window, with percentiles and the real-time factor against the window length.
- **Taps.** Every stage output is a named tap. Overlays, telemetry, recorders
  and test assertions subscribe to it, at no cost when nobody listens.
- **Faults.** Errors are typed: `ConfigError` at build, `ResourceError` when a
  device or dependency is missing, `StageFault` at runtime. A fault appears in
  the result stream with a degraded status. The field policy isolates the
  stage, counts and surfaces the fault, and never swallows it. The research
  policy fails fast.
- **Provenance.** Results and scores carry a hash of the pipeline spec and the
  method versions.

### 2.7 Benchmarking falls out of that

- Scoring sits outside the pipeline and consumes the result stream plus labels.
  A scorer declares the level it needs (boxes, ids, verdicts, alarms) and is
  matched against what the pipeline emits.
- Oracle stages (ground-truth detector, ground-truth tracks) isolate one stage's
  error and give a funnel: where labelled seconds are lost (threshold, cap,
  tracking, classification).
- Cost is a first-class metric, measured on the target hardware through the
  same runner. A tap recorder lets a downstream stage be re-scored without
  re-running upstream.

### 2.8 Usability

- A toy method is about 30 lines: a config dataclass, `step`, and a decorator
  that builds the manifest.
- Config is data, validated against the dataclass; unknown keys are rejected.
- The core contract, chain and registry are torch-free. A zero-config fallback
  chain (blob tracking plus drone / no-drone) ships with evio.
- A conformance test kit that every registered method must pass: deterministic,
  `reset()` works, no cross-camera state leaks, inputs not mutated, empty
  windows handled, timestamps monotonic.

### 2.9 Neutral about topology and method class

The contract constrains only what the runtime itself must do generically:
time it, tap it, fault it, key its state, score it. Every element must be used
by the runtime or a scorer; if nothing uses it, it stays out. Camera id fails
that test (its only runtime use is keying state), so it is replaced by an
opaque **stream key**. Where a stance could creep in, and how it is avoided:

- **Camera topology.** The pipeline's input is a `Step` holding one or more
  aligned windows, each under a stream key. Per-camera, mosaic and joint
  multi-camera setups are then configuration:
  - a per-stream method gets one instance per stream from the runtime;
  - a mosaic is a source (a stitcher) that turns several windows into one;
  - a joint method declares that it takes the whole set.
  Whether the OPS arc runs as six streams or one 6x1280 by 720 mosaic is a
  deployment choice, not a property of any method.
- **Spatial representation.** A window is a set of events (x, y, t, polarity)
  in a declared coordinate space, with the extent optional. A raster is not
  required. A method that wants a voxel grid, patches or a graph builds it
  itself; the evidence providers in 2.4 are opt-in helpers, not part of the
  contract. A spiking or point-cloud method that never forms a frame fits the
  same interface.
- **Facts and requirements instead of a sensor struct.** A source advertises
  whatever facts it has: extent, degrees per pixel, seam positions, time
  synchronisation, polarity available. All are optional and plugins can add new
  kinds. A method declares what it needs, for example degrees per pixel or
  seam awareness. The builder checks that the facts cover the requirements
  (feature detection, not inheritance). A method that declares nothing runs
  anywhere. Sensor model and dataset are metadata for scoring and diagnostics,
  not inputs to detection logic. Where a method depends on something like
  event rate, it reads a measured quantity from the stream.
- **Output geometry.** Tracks are regions in the window's coordinate space plus
  an id. A box is the common minimum; a point, mask or direction can accompany
  it. A scorer declares what it needs (box IoU, centre distance) and the builder
  checks it, so a method that emits only centroids is still scoreable.
- **Mapping back to the world.** Turning a pipeline-space region into a
  physical camera and sky direction is done by a resolver the runtime owns.
  SAPIENT, recording, bias control and health need it. It is device
  configuration and lives outside the pipeline and outside evioR.
- **Time.** This is the one stance that stays: the contract is
  step-synchronous, with a step duration the runtime sets. Cadence, per-step
  timing and replay equality depend on it. An event-driven method can still
  iterate events inside a step.

**Neutrality gate.** Before the contract is frozen, write deliberately
different reference methods against it: the classic raster blob tracker, a
no-raster point or spiking method, a GPU voxel CNN (stubbed, to exercise the
manifest and dependencies), a joint multi-stream method, and a stitcher plus
single-frame detector. If any needs special-casing in the contract, the
contract has a stance. The conformance kit runs over all of them.

**Cost.** The facts and requirements machinery is the one new mechanism. Keep
the fact kinds few at first, and let the decorator infer "no requirements" by
default, so the simple case (one stream, known extent, plain raster method)
costs a method author nothing.

### 2.10 Deliberately left out

A general DAG of stages (the linear chain, a fused option, feedback across
steps and cues cover what exists today), and async everywhere (async stays
inside the shared classifier service).

## 3. How the current modules fit

| Current piece | Fit |
|---|---|
| `simplified` pipeline, `background/`, `grid` | A **Detector**. It already has a registry and a calibrating state. |
| `TrackManager` (Kalman, persistence, re-id, dormant tracks) | A **Tracker**. `track_features` and `live_track_ids` are side channels that should be part of the `Tracks` output. |
| `TrackTransformerClassifier` | A **Classifier**. It mixes generic ring and cadence handling (runtime and evidence) with the model; its sticky verdicts need explicit freshness. |
| `verdict_gate`, `FusionGate`, the `LLRArm` protocol | The **decision stage**. `LLRArm` is the best-shaped existing plugin interface; copy its pattern. |
| `rotor_periodicity`, `irstd`, `kinematic_llr`, `rotor_llr` | Evidence providers and fusion arms. Today they are wired inline in ops. |
| `Detection` | The main problem: a god object that every stage patches with `dataclasses.replace` (`patch`, `embedding`, `rotor_hz`, `classification`, `track_id`). It splits into Candidate, Track and Verdict. |
| `dedupe`, `top_k_by_mcoh` | Candidate filtering that three places re-implement: ops `process_batch`, evio `live_camera`, evioR `CurrentPipeline`. The chain should own it once. |
| `live_camera.CameraDetector` | Proof the stage boundary must be serializable: its result (boxes, ids, patches, tokens) is the worker-to-main shipping format. |
| `DetectionEmitter` | A narrow version of taps. |
| `RollingWindowBatchProcessor` | The `Window` source for one stream. |
| ops per-stage timing for the frozen replay | The runtime timing, generalised. |
| evioR `Pipeline.step` / `Output`, scoring, runner | The scoring side, essentially as designed. Blob detector plus tracker is a Detector and Tracker pair; `tracks.py` is a tap recorder; the 0.5 ms event series is an evidence provider; `system_eval` is the decision-level scorer. |

**New work:** the `Step` and `Window` with stream key and source facts, the result with status and timing,
taps, manifests, evidence providers, verdict freshness, oracle stages, the fault
policy and the conformance kit.

## 4. Rollout (no release until all three repos work)

0. Neutrality gate (2.9): sketch the contract and write the reference
   methods against it before building the runtime.
1. Build additively on an evio feature branch: new protocols, registries and
   chain; the existing `DetectionPipeline` and `build` stay untouched, so ops
   pinned to v3.0.0 keeps working.
2. Try it in ops on its own branch with `make check EVIO_DIR=../evio` (layers a
   local evio over the pin for that run). The frozen replay shows whether output
   and per-stage cost changed.
3. Port evioR's methods to the contract against the local evio, with an
   uncommitted path override.
4. When all three work, merge to evio `dev`, tag a release (additive, so a minor
   version), then bump the pins in ops and evioR.

A published tag is never moved, so a flawed contract is cheapest to find in
steps 2 and 3.
