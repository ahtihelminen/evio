"""Conformance kit: what any registered DTC method must satisfy.

`check_method` builds a method through the registry and drives it directly
with synthetic input matching its first stage (steps for a detector or a
fused method, candidates for a tracker, tracks for a classifier, tracks and
verdicts for a decision). It returns human-readable failures, each prefixed
with the check that found it; an empty list means conformant. Each check is
its own function of a `Rig`, so a caller can run just one.

Numpy only: no torch, no camera, no clock.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass
from inspect import ismodule
from typing import Any
from unittest import mock

import numpy as np

from evio.core.events import EventsBatch
from evio.dtc.facts import Facts, missing_facts
from evio.dtc.manifest import Manifest, Scope, Stage, check_dependencies
from evio.dtc.protocols import Context
from evio.dtc.registry import METHODS, create
from evio.dtc.types import (
    OK,
    Alarm,
    Alarms,
    Candidate,
    Candidates,
    Output,
    Region,
    State,
    Status,
    Step,
    Track,
    Tracks,
    Verdict,
    Verdicts,
    Window,
)

STEP_COUNT = 8
MAX_REPORTED = 3
BLOB_RADIUS = 3
STAGE_OUTPUT: dict[Stage, type] = {
    Stage.DETECT: Candidates,
    Stage.TRACK: Tracks,
    Stage.CLASSIFY: Verdicts,
    Stage.DECIDE: Alarms,
}
OUTPUT_FIELD = {
    Stage.DETECT: "candidates",
    Stage.TRACK: "tracks",
    Stage.CLASSIFY: "verdicts",
    Stage.DECIDE: "alarms",
}
ITEM_TYPE: dict[type, type] = {
    Candidates: Candidate,
    Tracks: Track,
    Verdicts: Verdict,
    Alarms: Alarm,
}
CLOCKS = (
    "time",
    "time_ns",
    "monotonic",
    "monotonic_ns",
    "perf_counter",
    "perf_counter_ns",
    "process_time",
    "process_time_ns",
)
HEAVY_MODULES = frozenset({"torch", "torchvision", "onnxruntime", "openvino"})
OK_REGION = Region(point=(0.0, 0.0))
Instance = Any
Key = str | None
Runs = dict[Key, list[object]]


def synthetic_steps(
    streams: tuple[str, ...],
    step_us: int,
    extent: tuple[int, int],
    *,
    count: int = STEP_COUNT,
    seed: int = 0,
    empty: bool = False,
) -> list[Step]:
    """Steps of one blob per stream, drifting right; empty windows if asked."""
    width, height = extent
    steps = []
    for index in range(count):
        t0, t1 = index * step_us, (index + 1) * step_us
        windows = {}
        for number, stream in enumerate(streams):
            rng = np.random.default_rng([seed, number, index])
            n = 0 if empty else 50
            cx = (10 + 5 * index + 17 * number) % width
            cy = (height // 2 + 11 * number) % height
            batch = EventsBatch(
                np.sort(rng.integers(t0, t1, n)).astype(np.int64),
                (cx + rng.integers(-BLOB_RADIUS, BLOB_RADIUS + 1, n)).astype(
                    np.int16
                ),
                (cy + rng.integers(-BLOB_RADIUS, BLOB_RADIUS + 1, n)).astype(
                    np.int16
                ),
                rng.integers(0, 2, n).astype(np.int8),
            )
            windows[stream] = Window(stream, batch, t0, t1, extent)
        steps.append(Step(index, t0, t1, windows))
    return steps


def _regions(step: Step) -> list[tuple[str, Region]]:
    """One box per non-empty window, around its events' centroid."""
    found = []
    for stream, window in step.windows.items():
        if len(window.events.timestamps):
            x = float(window.events.x_coords.mean())
            y = float(window.events.y_coords.mean())
            found.append((stream, Region(box=(x - 3, y - 3, x + 3, y + 3))))
    return found


def _tracks(step: Step) -> Tracks:
    items = tuple(
        Track(stream, f"{stream}:0", region, 0.9)
        for stream, region in _regions(step)
    )
    return Tracks(items, frozenset(t.id for t in items))


def _call(instance: Instance, manifest: Manifest, step: Step) -> object:
    """Call `instance.step` with the arguments its first stage takes."""
    first = manifest.stages[0]
    if len(manifest.stages) > 1 or first is Stage.DETECT:
        return instance.step(step, ())
    if first is Stage.TRACK:
        found = tuple(Candidate(s, r, 1.0) for s, r in _regions(step))
        return instance.step(step, Candidates(found))
    tracks = _tracks(step)
    if first is Stage.CLASSIFY:
        return instance.step(step, tracks, {})
    verdicts = Verdicts(
        tuple(Verdict(t.stream, t.id, 0.5) for t in tracks.items)
    )
    return instance.step(step, tracks, verdicts, {})


@dataclass(frozen=True, slots=True)
class Rig:
    """One method under test and the synthetic world it is driven in."""

    name: str
    facts: Facts
    streams: tuple[str, ...]
    step_us: int
    extent: tuple[int, int]
    config: Mapping[str, Any]

    @property
    def manifest(self) -> Manifest:
        return METHODS[self.name].manifest

    def steps(self, *, empty: bool = False) -> list[Step]:
        return synthetic_steps(
            self.streams, self.step_us, self.extent, empty=empty
        )

    def build(self, stream: Key) -> Instance:
        context = Context(stream, self.streams, self.facts, self.step_us)
        return create(self.name, self.config, context)

    def instances(self) -> dict[Key, Instance]:
        """One per stream, or a single one (key None) for a joint method."""
        if self.manifest.scope is Scope.JOINT:
            return {None: self.build(None)}
        return {stream: self.build(stream) for stream in self.streams}

    def drive(
        self, instances: Mapping[Key, Instance], steps: list[Step]
    ) -> Runs:
        """Step every instance through `steps` in order; outputs by key."""
        runs: Runs = {key: [] for key in instances}
        for step in steps:
            for key, instance in instances.items():
                view = step if key is None else step.only(key)
                runs[key].append(_call(instance, self.manifest, view))
        return runs


def _parts(manifest: Manifest, output: object) -> Iterator[object]:
    """The per-stage outputs inside what a step returned, as declared."""
    if len(manifest.stages) == 1:
        yield output
    else:
        for stage in manifest.stages:
            yield getattr(output, OUTPUT_FIELD[stage], None)


@contextmanager
def _clock_reads() -> Iterator[list[str]]:
    """Record the `time` clocks read inside the block (they still work)."""
    reads: list[str] = []

    def watch(name: str) -> Callable[[], float]:
        real = getattr(time, name)

        def read() -> float:
            reads.append(name)
            return real()

        return read

    with mock.patch.multiple(time, **{name: watch(name) for name in CLOCKS}):
        yield reads


def _output_problems(rig: Rig, output: object) -> Iterator[str]:
    manifest = rig.manifest
    if len(manifest.stages) > 1:
        if not isinstance(output, Output):
            yield f"returned {type(output).__name__}, not Output"
            return
        for stage in Stage:
            if (getattr(output, OUTPUT_FIELD[stage]) is None) != (
                stage not in manifest.stages
            ):
                yield f"Output.{OUTPUT_FIELD[stage]} disagrees with stages"
    parts = zip(manifest.stages, _parts(manifest, output), strict=True)
    for stage, part in parts:
        want = STAGE_OUTPUT[stage]
        if not isinstance(part, want):
            yield f"{stage.name} gave {type(part).__name__}"
            continue
        if not (
            isinstance(part.status, Status)
            and isinstance(part.status.state, State)
        ):
            yield f"{want.__name__}.status is not a Status"
        for item in part.items:
            if not isinstance(item, ITEM_TYPE[want]):
                yield f"{want.__name__} holds a {type(item).__name__}"
            elif item.stream not in rig.streams:
                yield f"item stream {item.stream!r} is not a stream given"
            elif not isinstance(getattr(item, "region", OK_REGION), Region):
                yield f"{type(item).__name__}.region is not a Region"
        for prior in getattr(part, "priors", ()):
            if not isinstance(prior, Region):
                yield "Tracks.priors holds a non-Region"


def _shape_failures(rig: Rig, runs: Runs) -> list[str]:
    problems = [
        f"stream {key!r} step {index}: {problem}"
        for key, outputs in runs.items()
        for index, output in enumerate(outputs)
        for problem in _output_problems(rig, output)
    ]
    return problems[:MAX_REPORTED]


def _differences(what: str, first: Runs, second: Runs) -> list[str]:
    return [
        f"{what}: stream {key!r} step {index} differs"
        for key, outputs in first.items()
        for index, (a, b) in enumerate(zip(outputs, second[key], strict=True))
        if a != b
    ][:1]


def _fingerprint(steps: list[Step]) -> list[object]:
    return [
        (
            step.index,
            tuple(step.windows),
            [
                array.tobytes()
                for window in step.windows.values()
                for array in (
                    window.events.timestamps,
                    window.events.x_coords,
                    window.events.y_coords,
                    window.events.polarities,
                )
            ],
        )
        for step in steps
    ]


def manifest_sane(rig: Rig) -> list[str]:
    """Stages contiguous, facts covered, defaults enough to build."""
    manifest, failures = rig.manifest, []
    first = manifest.stages[0]
    wanted = tuple(Stage(first + i) for i in range(len(manifest.stages)))
    if manifest.stages != wanted:
        failures.append("stages are not contiguous")
    if manifest.name != rig.name:
        failures.append(f"manifest name {manifest.name!r} is not {rig.name!r}")
    if manifest.cadence < 1:
        failures.append("cadence is below 1")
    if not is_dataclass(METHODS[rig.name].config_type):
        failures.append("config is not a dataclass")
    if missing := missing_facts(manifest.requires, rig.facts):
        failures.append(f"requires facts the source lacks: {list(missing)}")
    check_dependencies(manifest)
    rig.instances()
    return failures


def deterministic(rig: Rig) -> list[str]:
    """Two fresh instances given the same steps give equal outputs."""
    first = rig.drive(rig.instances(), rig.steps())
    second = rig.drive(rig.instances(), rig.steps())
    return _differences("two fresh instances", first, second)


def reset_restores_fresh_state(rig: Rig) -> list[str]:
    """Run, reset, rerun: the rerun equals the first run."""
    instances = rig.instances()
    first = rig.drive(instances, rig.steps())
    for instance in instances.values():
        instance.reset()
    again = rig.drive(instances, rig.steps())
    return _differences("rerun after reset", first, again)


def no_cross_stream_leak(rig: Rig) -> list[str]:
    """A stream's instance is unaffected by the other streams' instances."""
    if rig.manifest.scope is Scope.JOINT or len(rig.streams) < 2:
        return []
    stream = rig.streams[0]
    alone = rig.drive({stream: rig.build(stream)}, rig.steps())
    together = rig.drive(rig.instances(), rig.steps())
    return _differences(
        "stepping other streams", alone, {stream: together[stream]}
    )


def inputs_not_mutated(rig: Rig) -> list[str]:
    """The window arrays are as the method found them."""
    steps = rig.steps()
    before = _fingerprint(steps)
    rig.drive(rig.instances(), steps)
    return [] if _fingerprint(steps) == before else ["window events changed"]


def empty_windows_handled(rig: Rig) -> list[str]:
    """Empty windows do not raise and still give well-formed output."""
    runs = rig.drive(rig.instances(), rig.steps(empty=True))
    return _shape_failures(rig, runs)


def timestamps_monotonic(rig: Rig) -> list[str]:
    """Steps arrive in increasing order; nothing in the output runs back.

    A verdict is never older than the stream, and calibration progress
    never decreases.
    """
    failures = []
    for key, outputs in rig.drive(rig.instances(), rig.steps()).items():
        progress = 0.0
        for index, output in enumerate(outputs):
            for part in _parts(rig.manifest, output):
                status = getattr(part, "status", OK)
                where = f"stream {key!r} step {index}"
                if status.state is State.CALIBRATING and status.progress:
                    if status.progress < progress:
                        failures.append(f"{where}: progress fell")
                    progress = status.progress
                failures.extend(
                    f"{where}: age_steps {v.age_steps}"
                    for v in getattr(part, "items", ())
                    if isinstance(v, Verdict) and not 0 <= v.age_steps <= index
                )
    return failures


def reads_no_clock(rig: Rig) -> list[str]:
    """Stepping never calls a `time` clock; time comes from the windows."""
    with _clock_reads() as reads:
        rig.drive(rig.instances(), rig.steps())
    return [f"called time.{name}" for name in sorted(set(reads))]


def imports_nothing_heavy(rig: Rig) -> list[str]:
    """The method's module holds no torch-class import at module level."""
    namespace = vars(sys.modules[METHODS[rig.name].__module__])
    held = {
        (value.__name__ if ismodule(value) else str(value.__module__)).split(
            "."
        )[0]
        for value in namespace.values()
        if ismodule(value) or hasattr(value, "__module__")
    }
    return [f"module imports {name!r}" for name in sorted(HEAVY_MODULES & held)]


def output_well_typed(rig: Rig) -> list[str]:
    """The stage's output type, a Status, Regions and known streams."""
    return _shape_failures(rig, rig.drive(rig.instances(), rig.steps()))


CHECKS: tuple[Callable[[Rig], list[str]], ...] = (
    deterministic,
    reset_restores_fresh_state,
    no_cross_stream_leak,
    inputs_not_mutated,
    empty_windows_handled,
    timestamps_monotonic,
    reads_no_clock,
    imports_nothing_heavy,
    output_well_typed,
)


def _attempt(check: Callable[[Rig], list[str]], rig: Rig) -> list[str]:
    """Run one check, reporting an error the method raises as a failure."""
    try:
        return [f"{check.__name__}: {message}" for message in check(rig)]
    except Exception as error:
        return [f"{check.__name__}: raised {error!r}"]


def check_method(
    name: str,
    *,
    facts: Facts,
    streams: tuple[str, ...] = ("a", "b"),
    step_us: int = 10_000,
    extent: tuple[int, int] = (128, 96),
    config: Mapping[str, Any] | None = None,
) -> list[str]:
    """Failures of registered method `name`; empty means conformant."""
    if name not in METHODS:
        return [f"unknown method {name!r}"]
    rig = Rig(name, facts, streams, step_us, extent, config or {})
    if failures := _attempt(manifest_sane, rig):
        return failures
    return [f for check in CHECKS for f in _attempt(check, rig)]
