"""The chain builder and runtime: `build` turns a spec into a `Pipeline`.

Build-time checks run before any data flows: unknown method, config, stage
kinds that do not line up, missing source facts or dependencies, unknown
evidence. A built chain keeps one instance per stream for a STREAM method and
one for a JOINT method. Per step it runs the stages in order, hands each the
previous stage's output, and owns what no method should: candidate filtering,
the tracker-to-detector priors feedback, evidence (computed once per step and
shared), classifier cadence and verdict age, timing, taps and faults.

Outputs are held per stream. A joint stage sees all streams merged and its
output is split back by stream key; its `live_ids` and `priors`, which carry
no stream, go to every stream.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from itertools import chain
from typing import Any

from evio.dtc.errors import ConfigError
from evio.dtc.evidence import PROVIDERS
from evio.dtc.facts import Facts, missing_facts
from evio.dtc.faults import guarded
from evio.dtc.filtering import filter_candidates
from evio.dtc.manifest import Manifest, Scope, Stage, check_dependencies
from evio.dtc.protocols import ClassifierCall, Context
from evio.dtc.registry import METHODS, create
from evio.dtc.result import FaultPolicy, Pipeline, StepResult
from evio.dtc.spec import PipelineSpec
from evio.dtc.taps import TapBus
from evio.dtc.timing import StageTiming, SystemTimer, Timer, TimingStats
from evio.dtc.types import (
    OK,
    Alarms,
    Candidates,
    Fault,
    Output,
    Region,
    State,
    Status,
    Step,
    StreamKey,
    Tracks,
    Verdicts,
)

# Stage kind -> (name of its field in `Output`, its output type).
_OUTPUTS: dict[Stage, tuple[str, type]] = {
    Stage.DETECT: ("candidates", Candidates),
    Stage.TRACK: ("tracks", Tracks),
    Stage.CLASSIFY: ("verdicts", Verdicts),
    Stage.DECIDE: ("alarms", Alarms),
}
StageOutput = Candidates | Tracks | Verdicts | Alarms | Output
_SEVERITY = {State.OK: 0, State.CALIBRATING: 1, State.DEGRADED: 2}


def worst(statuses: Sequence[Status]) -> Status:
    return max(statuses, key=lambda s: _SEVERITY[s.state], default=OK)


def merge[T](cls: type[T], parts: Sequence[Any]) -> T:
    """One output of type `cls` from per-stream parts."""
    extra = (
        {
            "live_ids": frozenset().union(*(p.live_ids for p in parts)),
            "priors": tuple(chain.from_iterable(p.priors for p in parts)),
        }
        if cls is Tracks
        else {}
    )
    items = tuple(chain.from_iterable(p.items for p in parts))
    return cls(items, status=worst([p.status for p in parts]), **extra)


def split[T](whole: T, streams: Sequence[StreamKey]) -> dict[StreamKey, T]:
    """Per-stream outputs of a joint stage, by each item's stream key."""
    return {
        s: replace(whole, items=tuple(i for i in whole.items if i.stream == s))
        for s in streams
    }


@dataclass(frozen=True, slots=True)
class _Stage:
    manifest: Manifest
    instances: dict[StreamKey | None, Any]

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def kind(self) -> Stage:
        return self.manifest.stages[0]

    @property
    def fused(self) -> bool:
        return len(self.manifest.stages) > 1

    @property
    def batch(self) -> bool:
        return hasattr(next(iter(self.instances.values())), "step_many")

    def empty(self, status: Status) -> StageOutput:
        """A well-typed output holding nothing, for a faulted stage."""
        parts = {
            field_name: cls(status=status)
            for field_name, cls in map(_OUTPUTS.get, self.manifest.stages)
        }
        return Output(**parts) if self.fused else next(iter(parts.values()))


@dataclass
class _Run:
    """What one step has produced so far."""

    step: Step
    layers: dict[type, dict[StreamKey, Any]] = field(
        default_factory=lambda: {cls: {} for _, cls in _OUTPUTS.values()}
    )
    status: defaultdict[StreamKey, list[Status]] = field(
        default_factory=lambda: defaultdict(list)
    )
    faults: list[Fault] = field(default_factory=list)
    timings: dict[str, list[int]] = field(default_factory=dict)
    evidence: dict[tuple[str, StreamKey], Mapping[str, object]] = field(
        default_factory=dict
    )
    priors: dict[StreamKey | None, tuple[Region, ...]] = field(
        default_factory=dict
    )

    @property
    def streams(self) -> tuple[StreamKey, ...]:
        return tuple(self.step.windows)


class Chain:
    """A built pipeline; satisfies `evio.dtc.result.Pipeline`."""

    def __init__(
        self,
        spec: PipelineSpec,
        stages: list[_Stage],
        providers: dict[tuple[str, StreamKey], Any],
        *,
        streams: Sequence[StreamKey],
        step_us: int,
        policy: FaultPolicy,
        timer: Timer,
        bus: TapBus,
        provenance: str,
    ) -> None:
        self._filter = spec.candidate_filter
        self._stages = stages
        self._providers = providers
        self._streams = tuple(streams)
        self._step_us = step_us
        self._policy = policy
        self._timer = timer
        self.bus = bus
        self.provenance = provenance
        self.stats: dict[str, TimingStats] = {}
        self.fault_counts: Counter[str] = Counter()
        self.reset()

    def reset(self) -> None:
        for stage in self._stages:
            for instance in stage.instances.values():
                instance.reset()
        for provider in self._providers.values():
            provider.reset()
        self._last_index: int | None = None
        self._priors: dict[StreamKey | None, tuple[Region, ...]] = {}
        self._last_verdicts: dict[
            tuple[str, StreamKey | None], tuple[int, Verdicts]
        ] = {}
        names = [stage.name for stage in self._stages]
        names += [f"evidence:{name}" for name, _ in self._providers]
        self.stats.clear()
        self.stats.update({n: TimingStats(self._step_us) for n in names})
        self.fault_counts.clear()

    def step(self, step: Step) -> StepResult:
        if self._last_index is not None and step.index <= self._last_index:
            msg = f"step {step.index} does not follow {self._last_index}"
            raise ConfigError(msg)
        unknown = set(step.windows) - set(self._streams)
        if unknown:
            msg = f"windows for streams not built for: {sorted(unknown)}"
            raise ConfigError(msg)
        cued = {c.stream for c in step.cues} - {None, *self._streams}
        if cued:
            msg = f"cues for streams not built for: {sorted(cued)}"
            raise ConfigError(msg)
        self._last_index = step.index
        run = _Run(step)
        for stage in self._stages:
            self._run_stage(run, stage)
        return self._finish(run)

    def _run_stage(self, run: _Run, stage: _Stage) -> None:
        units = (
            [(None, run.step)]
            if stage.manifest.scope is Scope.JOINT
            else [(s, run.step.only(s)) for s in run.streams]
        )
        if stage.kind is Stage.CLASSIFY and not stage.fused:
            self._run_classifier(run, stage, units)
            return
        for stream, sub in units:
            evidence = self._evidence(run, stage, stream)
            out, _ = self._call(
                run,
                stage.name,
                stream,
                partial(self._invoke, run, stage, stream, sub, evidence),
                stage.empty,
            )
            self._publish(run, f"stage:{stage.name}", out)
            if stage.kind is Stage.DETECT and not stage.fused:
                items = filter_candidates(out.items, self._filter)
                out = replace(out, items=items)
            self._store(run, out, stream)

    def _invoke(
        self,
        run: _Run,
        stage: _Stage,
        stream: StreamKey | None,
        sub: Step,
        evidence: Mapping[str, Mapping[str, object]],
    ) -> StageOutput:
        instance = stage.instances[stream]
        match stage.kind:
            case Stage.DETECT:
                return instance.step(sub, self._priors_for(run, stream))
            case Stage.TRACK:
                return instance.step(sub, self._view(run, Candidates, stream))
            case Stage.CLASSIFY:
                tracks = self._view(run, Tracks, stream)
                return instance.step(sub, tracks, evidence)
            case Stage.DECIDE:
                tracks = self._view(run, Tracks, stream)
                verdicts = self._view(run, Verdicts, stream)
                return instance.step(sub, tracks, verdicts, evidence)

    def _run_classifier(
        self,
        run: _Run,
        stage: _Stage,
        units: list[tuple[StreamKey | None, Step]],
    ) -> None:
        """Run on due steps only; otherwise re-emit the last, aged."""
        ready = [(s, sub, self._evidence(run, stage, s)) for s, sub in units]
        due = run.step.index % stage.manifest.cadence == 0
        scored = self._score(run, stage, ready) if due else {}
        for stream, _, _ in ready:
            if due:
                verdicts, fault = scored[stream]
                self._publish(run, f"stage:{stage.name}", verdicts)
                if fault is None:
                    key = (stage.name, stream)
                    self._last_verdicts[key] = (run.step.index, verdicts)
            else:
                verdicts = self._aged(run, stage, stream)
            self._store(run, verdicts, stream)

    def _score(
        self,
        run: _Run,
        stage: _Stage,
        ready: list[tuple[StreamKey | None, Step, Mapping]],
    ) -> dict[StreamKey | None, tuple[Verdicts, Fault | None]]:
        if stage.batch:
            calls = [
                ClassifierCall(sub, self._view(run, Tracks, s), evidence)
                for s, sub, evidence in ready
            ]
            instance = next(iter(stage.instances.values()))
            results, fault = self._call(
                run,
                stage.name,
                None,
                partial(instance.step_many, calls),
                lambda status: [Verdicts(status=status)] * len(calls),
            )
            return {
                stream: (verdicts, fault)
                for (stream, _, _), verdicts in zip(ready, results, strict=True)
            }
        return {
            stream: self._call(
                run,
                stage.name,
                stream,
                partial(
                    stage.instances[stream].step,
                    sub,
                    self._view(run, Tracks, stream),
                    evidence,
                ),
                stage.empty,
            )
            for stream, sub, evidence in ready
        }

    def _aged(
        self, run: _Run, stage: _Stage, stream: StreamKey | None
    ) -> Verdicts:
        """The last verdicts, older by the steps since, minus dead tracks."""
        scored_at, last = self._last_verdicts.get(
            (stage.name, stream), (0, Verdicts())
        )
        live = self._view(run, Tracks, stream).live_ids
        age = run.step.index - scored_at
        return Verdicts(
            tuple(
                replace(v, age_steps=age)
                for v in last.items
                if v.track_id in live
            )
        )

    def _evidence(
        self, run: _Run, stage: _Stage, stream: StreamKey | None
    ) -> dict[str, dict[str, object]]:
        """The views a classifier or decision asked for, each computed once."""
        if stage.fused or stage.kind < Stage.CLASSIFY:
            return {}
        streams = run.streams if stream is None else (stream,)
        return {
            name: {
                track_id: value
                for s in streams
                for track_id, value in self._provide(run, name, s).items()
            }
            for name in stage.manifest.evidence
        }

    def _provide(
        self, run: _Run, name: str, stream: StreamKey
    ) -> Mapping[str, object]:
        key = (name, stream)
        if key not in run.evidence:
            provider = self._providers[key]
            run.evidence[key], _ = self._call(
                run,
                f"evidence:{name}",
                stream,
                partial(
                    provider.compute,
                    run.step.only(stream),
                    self._view(run, Tracks, stream),
                ),
                lambda status: {},
            )
        return run.evidence[key]

    def _call[T](
        self,
        run: _Run,
        name: str,
        stream: StreamKey | None,
        call: Callable[[], T],
        fallback: Callable[[Status], T],
    ) -> tuple[T, Fault | None]:
        """Time `call` under `name` and apply the fault policy to it."""
        wall, cpu = self._timer.wall_ns(), self._timer.cpu_ns()
        out, fault = guarded(
            self._policy, name, stream, run.step.index, call, fallback
        )
        spent = run.timings.setdefault(name, [0, 0])
        spent[0] += self._timer.wall_ns() - wall
        spent[1] += self._timer.cpu_ns() - cpu
        if fault:
            self.fault_counts[name] += 1
            run.faults.append(fault)
            degraded = Status(State.DEGRADED, detail=fault.message)
            for s in run.streams if stream is None else (stream,):
                run.status[s].append(degraded)
        return out, fault

    def _store(
        self, run: _Run, out: StageOutput, stream: StreamKey | None
    ) -> None:
        typed = (
            (out.candidates, out.tracks, out.verdicts, out.alarms)
            if isinstance(out, Output)
            else (out,)
        )
        for part in filter(None, typed):
            per_stream = (
                {stream: part}
                if stream is not None
                else split(part, run.streams)
            )
            run.layers[type(part)].update(per_stream)
            if isinstance(part, Tracks):
                run.priors[stream] = part.priors
            for s, one in per_stream.items():
                run.status[s].append(one.status)

    def _view[T](self, run: _Run, cls: type[T], stream: StreamKey | None) -> T:
        """The latest `cls` output of one stream, or of all merged."""
        layer = run.layers[cls]
        if stream is not None:
            return layer.get(stream, cls())
        return merge(cls, [layer.get(s, cls()) for s in run.streams])

    def _priors_for(
        self, run: _Run, stream: StreamKey | None
    ) -> tuple[Region, ...]:
        """Priors of one stream (or all), plus a joint tracker's, once."""
        streams = run.streams if stream is None else (stream,)
        keys = (*streams, None)
        return tuple(chain.from_iterable(self._priors.get(k, ()) for k in keys))

    def _finish(self, run: _Run) -> StepResult:
        merged = {cls: self._view(run, cls, None) for cls in run.layers}
        tracks = merged[Tracks]
        self._priors.update(run.priors)
        timings = {
            name: StageTiming(wall, cpu)
            for name, (wall, cpu) in run.timings.items()
        }
        for name, timing in timings.items():
            self.stats[name].record(timing)
        step = run.step
        self._publish(run, "cues", step.cues)
        self._publish(run, "candidates", merged[Candidates].items)
        self._publish(run, "tracks", tracks.items)
        self._publish(run, "verdicts", merged[Verdicts].items)
        self._publish(run, "alarms", merged[Alarms].items)
        result = StepResult(
            step.index,
            step.t_start_us,
            step.t_end_us,
            merged[Candidates].items,
            tracks.items,
            tracks.live_ids,
            merged[Verdicts].items,
            merged[Alarms].items,
            {s: worst(run.status[s]) for s in run.streams},
            tuple(run.faults),
            timings,
            self.provenance,
        )
        self._publish(run, "result", result)
        for fault in result.faults:
            self._publish(run, "fault", fault)
        return replace(result, faults=tuple(run.faults))

    def _publish(self, run: _Run, name: str, value: object) -> None:
        """Publish to a tap; a raising subscriber is a fault, not a crash."""
        if not self.bus.listening(name):
            return
        _, fault = guarded(
            self._policy,
            f"tap:{name}",
            None,
            run.step.index,
            partial(self.bus.publish, name, run.step.index, value),
            lambda status: None,
        )
        if fault:
            self.fault_counts[fault.stage] += 1
            run.faults.append(fault)


def build(
    spec: PipelineSpec,
    facts: Facts,
    *,
    streams: Sequence[StreamKey],
    step_us: int,
    policy: FaultPolicy = FaultPolicy.FIELD,
    timer: Timer | None = None,
    bus: TapBus | None = None,
) -> Pipeline:
    """Check the spec against the facts and build a ready chain.

    Raises `ConfigError` or `ResourceError` before any data flows.
    """
    if not streams:
        msg = "a chain needs at least one stream"
        raise ConfigError(msg)
    manifests = _checked_manifests(spec, facts)
    stages = []
    for stage_spec, manifest in zip(spec.stages, manifests, strict=True):
        keys = [None] if manifest.scope is Scope.JOINT else list(streams)
        contexts = {
            key: Context(key, tuple(streams), facts, step_us) for key in keys
        }
        instances = {
            key: create(stage_spec.method, stage_spec.config, context)
            for key, context in contexts.items()
        }
        stages.append(_Stage(manifest, instances))
    providers = {
        (name, stream): PROVIDERS[name](
            Context(stream, tuple(streams), facts, step_us)
        )
        for name in dict.fromkeys(n for m in manifests for n in m.evidence)
        for stream in streams
    }
    return Chain(
        spec,
        stages,
        providers,
        streams=streams,
        step_us=step_us,
        policy=policy,
        timer=timer or SystemTimer(),
        bus=bus or TapBus(),
        provenance=_provenance(spec, manifests),
    )


def _checked_manifests(spec: PipelineSpec, facts: Facts) -> list[Manifest]:
    manifests = []
    for stage_spec in spec.stages:
        if stage_spec.method not in METHODS:
            known = sorted(METHODS)
            msg = f"unknown method {stage_spec.method!r}; known: {known}"
            raise ConfigError(msg)
        manifest = METHODS[stage_spec.method].manifest
        missing = missing_facts(manifest.requires, facts)
        if missing:
            msg = f"method {manifest.name!r} needs source facts {missing}"
            raise ConfigError(msg)
        check_dependencies(manifest)
        manifests.append(manifest)
    kinds = tuple(chain.from_iterable(m.stages for m in manifests))
    if not kinds or kinds != tuple(Stage)[: len(kinds)]:
        msg = f"stages must run DETECT, TRACK, CLASSIFY, DECIDE; got {kinds}"
        raise ConfigError(msg)
    wanted = {name for m in manifests for name in m.evidence}
    if wanted - PROVIDERS.keys():
        importlib.import_module("evio.dtc.evidence_providers")
    unknown = wanted - PROVIDERS.keys()
    if unknown:
        msg = f"unknown evidence providers: {sorted(unknown)}"
        raise ConfigError(msg)
    return manifests


def _provenance(spec: PipelineSpec, manifests: list[Manifest]) -> str:
    """Hash of the spec and each method's name and version."""
    payload = {
        "spec": asdict(spec),
        "methods": [[m.name, m.version] for m in manifests],
    }
    text = json.dumps(_canonical(payload, "provenance"), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _canonical(value: object, path: str) -> object:
    """JSON-stable form: str keys, lists for sequences, no `repr` fallback."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(k): _canonical(v, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_canonical(v, f"{path}[{i}]") for i, v in enumerate(value)]
    msg = f"{path}: {type(value).__name__} is not hashable into provenance"
    raise ConfigError(msg)
