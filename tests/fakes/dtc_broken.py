"""Toy DTC methods for the conformance kit: one sound, the rest broken.

Each broken toy violates exactly one rule the kit checks. They are plain
classes; the test registers them under unique names and removes them after.
`TOYS` maps a name to (class, stages, manifest options).
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Sequence
from typing import Any

from evio.dtc.manifest import Scope, Stage
from evio.dtc.protocols import Context
from evio.dtc.registry import NoConfig
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

_NOISE = itertools.count()


def _centroids(step: Step, seen: int) -> Candidates:
    """One candidate per non-empty window; `seen` is the state to expose."""
    return Candidates(
        tuple(
            Candidate(
                stream,
                Region(
                    point=(
                        float(w.events.x_coords.mean()),
                        float(w.events.y_coords.mean()),
                    )
                ),
                float(len(w.events.timestamps)),
                {"seen": float(seen)},
            )
            for stream, w in step.windows.items()
            if len(w.events.timestamps)
        )
    )


class Centroid:
    """Sound: a centroid per window, counting the steps it has seen."""

    def __init__(self, config: NoConfig, context: Context) -> None:
        self.seen = 0

    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        self.seen += 1
        return _centroids(step, self.seen)

    def reset(self) -> None:
        self.seen = 0


class JointCentroid(Centroid):
    """Sound, and takes every stream at once."""


class Nondeterministic(Centroid):
    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        found = super().step(step, priors)
        return Candidates(
            tuple(
                Candidate(c.stream, c.region, float(next(_NOISE)))
                for c in found.items
            )
        )


class ForgetsReset(Centroid):
    def reset(self) -> None:
        pass


class SharesState(Centroid):
    shared_seen = 0

    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        type(self).shared_seen += 1
        return _centroids(step, type(self).shared_seen)


class MutatesWindow(Centroid):
    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        for window in step.windows.values():
            window.events.x_coords += 1
        return super().step(step, priors)


class CrashesOnEmpty(Centroid):
    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        for window in step.windows.values():
            _ = window.events.x_coords[0]
        return super().step(step, priors)


class ReadsClock(Centroid):
    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        time.monotonic()
        return super().step(step, priors)


class NeedsFact(Centroid):
    pass


class ReactsToUnknownCue(Centroid):
    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        self.seen += len(step.cues)
        return super().step(step, priors)


class CrashesOnFeedback(Centroid):
    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        _ = step.feedback.verdicts.items[0]
        return super().step(step, priors)


class FeedbackReader(Centroid):
    """Sound: counts the last step's verdicts and the radar cues it sees."""

    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        self.seen += len(step.feedback.verdicts.items)
        self.seen += sum(c.kind == "radar" for c in step.cues)
        return super().step(step, priors)


class WrongType(Centroid):
    def step(self, step: Step, priors: Sequence[Region]) -> Tracks:
        return Tracks()


class SoundTracker(Centroid):
    def step(self, step: Step, candidates: Candidates) -> Tracks:
        self.seen += 1
        items = tuple(
            Track(c.stream, f"{c.stream}:0", c.region, 0.9, {"seen": self.seen})
            for c in candidates.items
        )
        return Tracks(items, frozenset(t.id for t in items))


class SoundClassifier(Centroid):
    def step(self, step: Step, tracks: Tracks, evidence: Evidence) -> Verdicts:
        self.seen += 1
        return Verdicts(
            tuple(
                Verdict(t.stream, t.id, 1 / (1 + self.seen))
                for t in tracks.items
            )
        )


class SoundDecision(Centroid):
    def step(
        self, step: Step, tracks: Tracks, verdicts: Verdicts, evidence: Evidence
    ) -> Alarms:
        self.seen += 1
        return Alarms(
            tuple(
                Alarm(v.stream, v.track_id, v.probability * self.seen)
                for v in verdicts.items
            )
        )


class SoundFused(Centroid):
    def step(self, step: Step, priors: Sequence[Region]) -> Output:
        found = super().step(step, priors)
        return Output(
            found, Tracks(priors=tuple(c.region for c in found.items))
        )


DETECT = (Stage.DETECT,)
TOYS: dict[str, tuple[type, tuple[Stage, ...], dict[str, Any]]] = {
    "toy-centroid": (Centroid, DETECT, {}),
    "toy-joint": (JointCentroid, DETECT, {"scope": Scope.JOINT}),
    "toy-tracker": (SoundTracker, (Stage.TRACK,), {}),
    "toy-classifier": (SoundClassifier, (Stage.CLASSIFY,), {}),
    "toy-decision": (SoundDecision, (Stage.DECIDE,), {}),
    "toy-fused": (SoundFused, (Stage.DETECT, Stage.TRACK), {}),
    "toy-nondeterministic": (Nondeterministic, DETECT, {}),
    "toy-forgets-reset": (ForgetsReset, DETECT, {}),
    "toy-shares-state": (SharesState, DETECT, {}),
    "toy-mutates-window": (MutatesWindow, DETECT, {}),
    "toy-crashes-on-empty": (CrashesOnEmpty, DETECT, {}),
    "toy-wrong-type": (WrongType, DETECT, {}),
    "toy-reads-clock": (ReadsClock, DETECT, {}),
    "toy-needs-fact": (NeedsFact, DETECT, {"requires": ("deg_per_px",)}),
    "toy-reacts-to-unknown-cue": (ReactsToUnknownCue, DETECT, {}),
    "toy-crashes-on-feedback": (
        CrashesOnFeedback, DETECT, {"feedback": (Stage.CLASSIFY,)},
    ),
    "toy-feedback-reader": (
        FeedbackReader, DETECT, {"feedback": (Stage.CLASSIFY,)},
    ),
}
