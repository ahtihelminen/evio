# DTC Module (`src/evio/dtc`)

Hosts detect, track, classify (DTC) pipelines so methods can be swapped, run
live or offline, and compared. The design is `docs/plans/dtc-architecture.md`.
The existing `detection.pipeline` registry is untouched; this package is
additive.

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
- The chain owns candidate filtering, classifier cadence and verdict age,
  timing (injectable `Timer`), taps, faults and provenance.
- Faults: `FaultPolicy.FIELD` isolates the faulting stage, surfaces a
  `Fault` and a degraded status; `RESEARCH` raises `StageFault`.

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
windows, no clock, no heavy import at module level.

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
verdict gate.

## Known limits

- `TrackManager` mints ULIDs from the wall clock; the `trackmanager` adapter
  defaults to counter ids so replay is reproducible.
- Priors and live ids carry no stream key; a joint tracker's priors reach
  every stream's detector.
- A classifier's cadence is fixed by its manifest at decoration time.
