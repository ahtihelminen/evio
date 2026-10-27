# DTC Module (`src/evio/dtc`)

Hosts detect, track, classify (DTC) pipelines so methods can be swapped, run
live or offline, and compared. The design is `docs/plans/dtc-architecture.md`.

## Responsibilities

- The contract a method implements (`types`, `protocols`, `manifest`,
  `facts`, `registry`, `spec`).
- The chain that builds and runs methods (`chain`, `filtering`, `faults`,
  `evidence`, `evidence_providers`, `taps`, `timing`, `result`).
- A source side for tests and replay (`runner`, `sources`, `recording`).
- The conformance kit every registered method must pass (`conformance`).
- Reference, adapter and oracle methods (`methods/`) and the zero-config
  `fallback_spec()`.

It does not own models or research methods (those are promoted from evior
one at a time), device configuration, the pixel-to-sky resolver or
appliance concerns.

## The idea

A pipeline is a deterministic, causal state machine over sensor-time steps.
Placement, timing, errors and scoring are done to it from outside.

- Stages emit separate plain types (`Candidates`, `Tracks`, `Verdicts`,
  `Alarms`), joined by track id. Every one is frozen data and can cross a
  process.
- A `Step` holds one or more aligned `Window`s under opaque stream keys.
  Methods never read a clock; replay is the live code path.
- A method declares a `Manifest` (stages, required source facts, compute
  class, dependencies, evidence, cadence, geometry). `build()` checks it
  before data flows: `ConfigError` for a bad spec, config or missing fact,
  `ResourceError` for a missing dependency.
- Evidence views (patch ring, ROI events, event series, kinematics) are
  requested in the manifest and computed once per stream per step.
- Any stage may take priors, and none has to (see *Priors* below).
- The chain owns candidate filtering, classifier cadence and verdict age,
  timing (injectable `Timer`), taps, faults and provenance.
- Faults: `FaultPolicy.FIELD` isolates the faulting stage, surfaces a
  `Fault` and a degraded status; `RESEARCH` raises `StageFault`.

## Priors

Every stage can be given priors through the `Step` it already receives; no
stage signature carries them, so a method that ignores them is unchanged.

- **Cues** (`Step.cues`, `Cue`) come from outside the chain: another
  camera's hand-off, a radar cue, an operator-confirmed target, a site map.
  The source sets them; `Step.only` hands a stream its own cues and the
  global ones (`stream` None), and a joint method sees all. A `Cue` names a
  `kind` and its `source`, a region, a track id or both. A method ignores
  kinds it does not know (the conformance kit checks this), so new kinds
  need no method changes. The chain publishes each step's cues on the
  `cues` tap and `runner.with_cues` puts recorded cues back for a replay.
  Static priors that do not change per step stay source facts.
- **Feedback** (`Step.feedback`, an `Output`) is the chain's own outputs of
  the last step it ran. A method names the stage kinds it reads in its
  manifest, `feedback=(Stage.CLASSIFY,)`, its own kind included; the chain
  fills only those parts, each stream's share for a per-stream method and
  every stream merged for a joint one. Everything else stays empty
  (`NO_FEEDBACK`), and so does a declared kind no stage in the chain
  produces: feedback is never required. A source may not set it.

Within a step the stages still run in order, each seeing only earlier
stages; feedback only crosses step boundaries, so a chain stays causal and
deterministic and replay is still the live path. `reset()` forgets the last
step. "The last step it ran" is not always `index - 1`: a source may skip
indices across a gap, so a method that ages what it holds uses `step.index`
or the verdicts' `age_steps`.

```python
@method("hold-tracker", stages=(Stage.TRACK,), feedback=(Stage.CLASSIFY,))
class HoldTracker:
    def step(self, step: Step, candidates: Candidates) -> Tracks:
        drones = {v.track_id for v in step.feedback.verdicts.items
                  if v.probability >= 0.9}
        ...  # keep `drones` alive longer while candidates are missing
```

The tracker-to-detector `priors` argument is unchanged and is still the
tracker's predicted regions.

## A toy method

```python
@dataclass(frozen=True)
class ThresholdConfig:
    min_events: int = 50

@method("threshold", stages=(Stage.DETECT,), config=ThresholdConfig)
class Threshold:
    """Emit a point where a window holds enough events."""

    def __init__(self, config: ThresholdConfig, context: Context) -> None: ...
    def step(self, step: Step, priors: Sequence[Region]) -> Candidates: ...
    def reset(self) -> None: ...
```

Run `check_method("threshold", facts={...})` from `evio.dtc.conformance`:
determinism, `reset()`, no cross-stream leak, inputs not mutated, empty
windows, no clock, no heavy import at module level, well-formed and
deterministic output with cues and feedback on the steps, and unknown cue
kinds ignored.

## Running

```python
import evio.dtc.methods  # registers every method
from evio.dtc.chain import build
from evio.dtc.fallback import fallback_spec

pipeline = build(fallback_spec(), {"extent": (1280, 720)},
                 streams=["cam0"], step_us=10_000)
for result in run(pipeline, steps_from_batches(batches, "cam0", 10_000)):
    ...
```

## Neutrality

The contract constrains only what the runtime must do generically (time it,
tap it, fault it, key its state, score it). It does not assume a camera, a
raster, a topology or a sensor. The neutrality gate checked that with five deliberately different fixtures
under `tests/fakes/` (`dtc_*`): a raster blob method, a no-raster spiking
method, a stubbed GPU voxel CNN, a joint multi-stream method and a stitcher
plus single-frame detector (`sources.stitch`, `dtc_frame`). They are test
fixtures, not shipped methods; the conformance suite runs the kit over them
and over every shipped method.

The shipped fallback (`fallback_spec()`) is the rate-excess detector and the
blob tracker, ported from evioR, then the kinematic classifier and the
verdict gate. The rate-excess detector exposes its learned per-cell background
as read-only `mean` and `spread` arrays of shape (rows, cols); callers must not
mutate them.

## Known limits

- `TrackManager` mints ULIDs from the wall clock; the `trackmanager` adapter
  defaults to counter ids so replay is reproducible.
- A tracker's predicted regions (`Tracks.priors`) and live ids carry no
  stream key; a joint tracker's reach every stream's detector, and its
  `live_ids` and `priors` reach every stream in feedback too.
- A classifier's cadence is fixed by its manifest at decoration time.
