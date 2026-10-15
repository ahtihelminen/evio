"""The stage interfaces a method implements.

A method is a deterministic, causal state machine over steps. It never reads
a clock and is built once per stream by the runtime (or once, for a joint
method). State is explicit and `reset()` returns it to the freshly built
state. Methods are structural: no base class to inherit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

from evio.dtc.facts import Facts
from evio.dtc.manifest import Manifest
from evio.dtc.types import (
    Alarms,
    Candidates,
    Evidence,
    Output,
    Region,
    Step,
    StreamKey,
    Tracks,
    Verdicts,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class Context:
    """What the builder tells a method about where it runs.

    `stream` is None for a joint method, which sees every stream in `streams`.
    """

    stream: StreamKey | None
    streams: tuple[StreamKey, ...]
    facts: Facts
    step_us: int


class Method(Protocol):
    """Every method: built from its config and a context, resettable."""

    manifest: ClassVar[Manifest]

    def reset(self) -> None: ...


class Detector(Method, Protocol):
    """`priors` are regions (boxes or points) the tracker predicts for this
    step; a detector may ignore them. A joint method may receive a partial
    step, with fewer windows than `Context.streams`, and says so with a
    DEGRADED status."""

    def step(self, step: Step, priors: Sequence[Region]) -> Candidates: ...


class Tracker(Method, Protocol):
    def step(self, step: Step, candidates: Candidates) -> Tracks: ...


class Classifier(Method, Protocol):
    """Scores tracks. The runtime owns cadence and verdict age."""

    def step(
        self, step: Step, tracks: Tracks, evidence: Evidence
    ) -> Verdicts: ...


@dataclass(frozen=True, slots=True)
class ClassifierCall:
    step: Step
    tracks: Tracks
    evidence: Evidence


class BatchClassifier(Classifier, Protocol):
    """Optional: one model scores many streams' tracks in one call."""

    def step_many(self, calls: Sequence[ClassifierCall]) -> list[Verdicts]: ...


class Decision(Method, Protocol):
    def step(
        self,
        step: Step,
        tracks: Tracks,
        verdicts: Verdicts,
        evidence: Evidence,
    ) -> Alarms: ...


class FusedMethod(Method, Protocol):
    """A method implementing several stages at once, starting at DETECT.

    It owns its priors, so the `priors` argument may be ignored."""

    def step(self, step: Step, priors: Sequence[Region]) -> Output: ...
