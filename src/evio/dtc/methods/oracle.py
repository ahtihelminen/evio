"""Ground-truth stages, to isolate one stage's error.

`oracle-detect` emits the labelled boxes as candidates and `oracle-track`
the labelled tracks, so a downstream stage is scored against perfect input.
`replay-detect` re-emits recorded candidates, so a stage can be re-scored
without re-running what is upstream of it. All three read their boxes from
config data keyed by step index: `(x0, y0, x1, y1, label)` per box. Config
may arrive from a spec as lists with JSON-style string keys; it is normalised
to immutable tuples and int keys. A step with no entry has no boxes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from evio.dtc.manifest import Stage
from evio.dtc.protocols import Context
from evio.dtc.registry import method
from evio.dtc.types import (
    Candidate,
    Candidates,
    Region,
    Step,
    Track,
    Tracks,
)

LabelledBox = tuple[float, float, float, float, str]
Labels = Mapping[int, tuple[LabelledBox, ...]]


def _normalised(
    raw: Mapping[int | str, Sequence[Sequence[float | str]]],
) -> Labels:
    return MappingProxyType(
        {
            int(index): tuple(
                (float(x0), float(y0), float(x1), float(y1), str(label))
                for x0, y0, x1, y1, label in boxes
            )
            for index, boxes in raw.items()
        }
    )


@dataclass(frozen=True)
class OracleConfig:
    labels: Labels = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", _normalised(self.labels))


@dataclass(frozen=True)
class ReplayConfig:
    recorded: Labels = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "recorded", _normalised(self.recorded))


class _BoxesByStep:
    """Stateless: what it emits depends only on the step index."""

    def __init__(self, labels: Labels, context: Context) -> None:
        self.labels = labels
        self.stream = context.stream

    def reset(self) -> None:
        """Stateless: nothing to return to the built state."""

    def _boxes(self, step: Step) -> tuple[LabelledBox, ...]:
        return self.labels.get(step.index, ())


class _DetectBoxes(_BoxesByStep):
    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        return Candidates(
            tuple(
                Candidate(self.stream, Region(box=box[:4]), 1.0)
                for box in self._boxes(step)
            )
        )


@method("oracle-detect", stages=(Stage.DETECT,), config=OracleConfig)
class OracleDetect(_DetectBoxes):
    """Ground-truth detector: the step's labelled boxes, score 1.0."""

    def __init__(self, config: OracleConfig, context: Context) -> None:
        super().__init__(config.labels, context)


@method("replay-detect", stages=(Stage.DETECT,), config=ReplayConfig)
class ReplayDetect(_DetectBoxes):
    """Re-emits recorded candidates, score 1.0; the label is ignored."""

    def __init__(self, config: ReplayConfig, context: Context) -> None:
        super().__init__(config.recorded, context)


@method("oracle-track", stages=(Stage.TRACK,), config=OracleConfig)
class OracleTrack(_BoxesByStep):
    """Ground-truth tracks: id is the label, no priors, candidates ignored."""

    def __init__(self, config: OracleConfig, context: Context) -> None:
        super().__init__(config.labels, context)

    def step(self, step: Step, candidates: Candidates) -> Tracks:
        tracks = tuple(
            Track(self.stream, box[4], Region(box=box[:4]), 1.0)
            for box in self._boxes(step)
        )
        return Tracks(tracks, frozenset(t.id for t in tracks))
