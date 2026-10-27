"""Toy DTC methods and an evidence provider for the chain tests.

`registered()` registers every toy under a unique `toy-*` name (and the
`toy-count` provider) for the length of a `with` block, then removes them,
so no other test sees them. What each toy does is logged in `LOG`, keyed by
role, so a test can see which calls the chain made.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from evio.dtc.evidence import PROVIDERS, evidence_provider
from evio.dtc.manifest import Scope, Stage
from evio.dtc.protocols import ClassifierCall, Context
from evio.dtc.registry import METHODS, NoConfig, method
from evio.dtc.types import (
    Alarm,
    Alarms,
    Candidate,
    Candidates,
    Evidence,
    Output,
    Region,
    Step,
    Track,
    Tracks,
    Verdict,
    Verdicts,
)

LOG: defaultdict[str, list[Any]] = defaultdict(list)
EVIDENCE = "toy-count"
HALF_PX = 3


@dataclass(frozen=True)
class DetectorConfig:
    """`boxes` candidates per window, `spread_px` apart, best score first."""

    boxes: int = 1
    spread_px: int = 20


@dataclass(frozen=True)
class ExplodeConfig:
    """Raise in the detector of `stream` at step `at`."""

    stream: str = "cam0"
    at: int = 1


class Detector:
    """One box per non-empty window around its mean, `boxes` of them."""

    def __init__(self, config: DetectorConfig, context: Context) -> None:
        self.config = config
        self.stream = context.stream
        LOG["built"].append(("detector", context.stream))

    def reset(self) -> None:
        """Stateless."""

    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        LOG["priors"].append((self.stream, tuple(priors)))
        LOG["cues"].append((self.stream, step.cues))
        events = step.windows[self.stream].events
        if not len(events.timestamps):
            return Candidates()
        cx, cy = float(events.x_coords.mean()), float(events.y_coords.mean())
        count = float(len(events.timestamps))
        return Candidates(
            tuple(
                Candidate(
                    self.stream,
                    Region(
                        (
                            cx - HALF_PX + i * self.config.spread_px,
                            cy - HALF_PX,
                            cx + HALF_PX + i * self.config.spread_px,
                            cy + HALF_PX,
                        )
                    ),
                    count - i,
                )
                for i in range(self.config.boxes)
            )
        )


class JointDetector:
    """One box per non-empty window, from one instance over every stream."""

    def __init__(self, config: NoConfig, context: Context) -> None:
        LOG["built"].append(("joint-detector", context.stream))

    def reset(self) -> None:
        """Stateless."""

    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        LOG["joint-priors"].append(tuple(priors))
        LOG["joint-cues"].append(step.cues)
        return Candidates(
            tuple(
                Candidate(stream, Region((0, 0, 2 * HALF_PX, 2 * HALF_PX)), 1)
                for stream, window in step.windows.items()
                if len(window.events.timestamps)
            )
        )


class Exploder(Detector):
    def __init__(self, config: ExplodeConfig, context: Context) -> None:
        super().__init__(DetectorConfig(), context)
        self.explode = config

    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        if (
            self.stream == self.explode.stream
            and step.index == self.explode.at
        ):
            msg = f"boom at {step.index}"
            raise ValueError(msg)
        return super().step(step, priors)


class Tracker:
    """A track per candidate, id by its rank; every track is live."""

    def __init__(self, config: NoConfig, context: Context) -> None:
        self.stream = context.stream
        LOG["built"].append(("tracker", context.stream))

    def reset(self) -> None:
        """Stateless."""

    def step(self, step: Step, candidates: Candidates) -> Tracks:
        LOG["feedback"].append((self.stream, step.index, step.feedback))
        items = tuple(
            Track(c.stream, f"{c.stream}:{i}", c.region, 0.9)
            for i, c in enumerate(candidates.items)
        )
        return Tracks(
            items,
            frozenset(t.id for t in items),
            tuple(t.region for t in items),
        )


class DetectTrack:
    """Fused detector and tracker of one stream."""

    def __init__(self, config: DetectorConfig, context: Context) -> None:
        self.detector = Detector(config, context)
        self.tracker = Tracker(NoConfig(), context)

    def reset(self) -> None:
        """Stateless."""

    def step(self, step: Step, priors: Sequence[Region]) -> Output:
        found = self.detector.step(step, priors)
        return Output(found, self.tracker.step(step, found))


class Classifier:
    """Probability from the `toy-count` view: events in the window / 100."""

    def __init__(self, config: NoConfig, context: Context) -> None:
        self.stream = context.stream

    def reset(self) -> None:
        """Stateless."""

    def step(
        self, step: Step, tracks: Tracks, evidence: Evidence
    ) -> Verdicts:
        LOG["classifier"].append((self.stream, step.index))
        LOG["classifier-feedback"].append(step.feedback.verdicts)
        counts = evidence.get(EVIDENCE, {})
        return Verdicts(
            tuple(
                Verdict(t.stream, t.id, counts.get(t.id, 0) / 100)
                for t in tracks.items
            )
        )


class BatchClassifier(Classifier):
    """Scores every due stream's tracks at 0.5 in one `step_many`."""

    def step_many(self, calls: Sequence[ClassifierCall]) -> list[Verdicts]:
        LOG["batch"].append(len(calls))
        LOG["batch-feedback"] += [c.step.feedback.verdicts for c in calls]
        return [
            Verdicts(
                tuple(Verdict(t.stream, t.id, 0.5) for t in c.tracks.items)
            )
            for c in calls
        ]


class Decision:
    """An alarm for every verdict of at least 0.5, scored by it."""

    def __init__(self, config: NoConfig, context: Context) -> None:
        self.stream = context.stream

    def reset(self) -> None:
        """Stateless."""

    def step(
        self,
        step: Step,
        tracks: Tracks,
        verdicts: Verdicts,
        evidence: Evidence,
    ) -> Alarms:
        return Alarms(
            tuple(
                Alarm(v.stream, v.track_id, v.probability)
                for v in verdicts.items
                if v.probability >= 0.5
            )
        )


class Count:
    """Evidence: the number of events in the window, for every track."""

    def __init__(self, context: Context) -> None:
        self.stream = context.stream
        LOG["provider-built"].append(context.stream)

    def compute(self, step: Step, tracks: Tracks) -> dict[str, object]:
        LOG["provider"].append((self.stream, step.index))
        events = step.windows[self.stream].events
        return {t.id: len(events.timestamps) for t in tracks.items}

    def reset(self) -> None:
        """Stateless."""


DETECT = (Stage.DETECT,)
TOYS: dict[str, tuple[type, tuple[Stage, ...], dict[str, Any]]] = {
    "toy-detector": (Detector, DETECT, {"config": DetectorConfig}),
    "toy-exploder": (Exploder, DETECT, {"config": ExplodeConfig}),
    "toy-tracker": (Tracker, (Stage.TRACK,), {}),
    "toy-joint-detector": (JointDetector, DETECT, {"scope": Scope.JOINT}),
    "toy-joint-tracker": (Tracker, (Stage.TRACK,), {"scope": Scope.JOINT}),
    "toy-detect-track": (
        DetectTrack,
        (Stage.DETECT, Stage.TRACK),
        {"config": DetectorConfig},
    ),
    "toy-classifier": (
        Classifier,
        (Stage.CLASSIFY,),
        {"cadence": 2, "evidence": (EVIDENCE,)},
    ),
    "toy-classifier-every-step": (
        Classifier,
        (Stage.CLASSIFY,),
        {"evidence": (EVIDENCE,)},
    ),
    "toy-batch-classifier": (BatchClassifier, (Stage.CLASSIFY,), {}),
    "toy-feedback-tracker": (
        Tracker,
        (Stage.TRACK,),
        {"feedback": (Stage.CLASSIFY, Stage.DECIDE)},
    ),
    "toy-joint-feedback-tracker": (
        Tracker,
        (Stage.TRACK,),
        {"scope": Scope.JOINT, "feedback": (Stage.CLASSIFY,)},
    ),
    "toy-feedback-classifier": (
        Classifier,
        (Stage.CLASSIFY,),
        {"cadence": 2, "evidence": (EVIDENCE,), "feedback": (Stage.CLASSIFY,)},
    ),
    "toy-feedback-batch-classifier": (
        BatchClassifier,
        (Stage.CLASSIFY,),
        {"feedback": (Stage.CLASSIFY,)},
    ),
    "toy-decision": (
        Decision,
        (Stage.DECIDE,),
        {"evidence": (EVIDENCE,)},
    ),
    "toy-needs-module": (
        Detector,
        DETECT,
        {"config": DetectorConfig, "dependencies": ("no_such_module_xyz",)},
    ),
    "toy-wants-ghost": (
        Detector,
        DETECT,
        {"config": DetectorConfig, "evidence": ("ghost-view",)},
    ),
}


@contextmanager
def registered() -> Iterator[None]:
    """The toys and the `toy-count` provider, for the length of the block."""
    LOG.clear()
    for name, (cls, stages, options) in TOYS.items():
        # a subclass per name, since the manifest is stored on the class
        method(name, stages=stages, **options)(type(cls.__name__, (cls,), {}))
    evidence_provider(EVIDENCE)(Count)
    try:
        yield
    finally:
        for name in TOYS:
            del METHODS[name]
        del PROVIDERS[EVIDENCE]
