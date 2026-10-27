"""Plain, serializable data that flows through a DTC pipeline.

Each stage emits its own type and sees the previous stage's output: a
detector emits `Candidates`, a tracker `Tracks`, a classifier `Verdicts`, a
decision policy `Alarms`. They are joined by track id, never by a type that
grows the fields of every later stage. All of it is frozen data of tuples,
numbers and strings, so any stage boundary can cross a process.

Time is sensor time in integer microseconds; nothing here reads a clock.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evio.core.events import EventsBatch

StreamKey = str
Box = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Window:
    """Events of one stream over `[t_start_us, t_end_us)`.

    `events` are (x, y, t, polarity) in the stream's coordinate space, held
    as an `EventsBatch` and possibly empty. `extent` is that space's
    (width, height) when known. The stream key is opaque: the runtime only
    uses it to key state.
    """

    stream: StreamKey
    events: EventsBatch
    t_start_us: int
    t_end_us: int
    extent: tuple[int, int] | None = None


class State(StrEnum):
    OK = "ok"
    CALIBRATING = "calibrating"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class Status:
    """Whether an output can be trusted yet. Calibrating is a status."""

    state: State = State.OK
    progress: float | None = None
    detail: str = ""


OK = Status()


@dataclass(frozen=True, slots=True)
class Region:
    """A place in the window's coordinate space.

    A box is the common minimum; a point or a velocity may accompany it, or
    stand alone for a method that only emits centroids.
    """

    box: Box | None = None
    point: Point | None = None
    velocity_px_s: Point | None = None

    def __post_init__(self) -> None:
        if self.box is None and self.point is None:
            msg = "a region needs a box or a point"
            raise ValueError(msg)

    @property
    def center(self) -> Point:
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            return ((x0 + x1) / 2, (y0 + y1) / 2)
        assert self.point is not None
        return self.point

    def moved(self, dx: float, dy: float) -> Region:
        """The same region shifted by (dx, dy)."""
        box = self.box and (
            self.box[0] + dx, self.box[1] + dy,
            self.box[2] + dx, self.box[3] + dy,
        )
        point = self.point and (self.point[0] + dx, self.point[1] + dy)
        return Region(box, point, self.velocity_px_s)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A detector's proposal; `score` is its own confidence scale."""

    stream: StreamKey
    region: Region
    score: float
    features: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Candidates:
    items: tuple[Candidate, ...] = ()
    status: Status = OK


@dataclass(frozen=True, slots=True)
class Track:
    """A candidate that persisted, under a stable id."""

    stream: StreamKey
    id: str
    region: Region
    confidence: float
    features: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Tracks:
    """The tracker's output for one step.

    `live_ids` includes tracks alive but unseen this step (coasting); `items`
    holds only the tracks observed. `priors` are regions the detector may use
    on the next step. Track ids are unique across streams (a per-stream
    tracker prefixes the stream key), so a joint stage can merge them.
    """

    items: tuple[Track, ...] = ()
    live_ids: frozenset[str] = frozenset()
    priors: tuple[Region, ...] = ()
    status: Status = OK


@dataclass(frozen=True, slots=True)
class Verdict:
    """A classifier's probability for a track and how stale it is.

    `age_steps` is 0 when scored this step, otherwise the number of steps
    since the classifier last ran for the track.
    """

    stream: StreamKey
    track_id: str
    probability: float
    age_steps: int = 0


@dataclass(frozen=True, slots=True)
class Verdicts:
    items: tuple[Verdict, ...] = ()
    status: Status = OK


@dataclass(frozen=True, slots=True)
class Alarm:
    stream: StreamKey
    track_id: str
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class Alarms:
    items: tuple[Alarm, ...] = ()
    status: Status = OK


@dataclass(frozen=True, slots=True)
class Output:
    """What a fused method emits: every stage it implements, others None."""

    candidates: Candidates | None = None
    tracks: Tracks | None = None
    verdicts: Verdicts | None = None
    alarms: Alarms | None = None


NO_FEEDBACK = Output(Candidates(), Tracks(), Verdicts(), Alarms())


@dataclass(frozen=True, slots=True)
class Cue:
    """A prior from outside the chain: another camera's hand-off, a radar
    cue, an operator-confirmed target, the wind. `kind` says which and a
    method ignores kinds it does not know; `source` says where it came from.
    It names a region in `stream`'s window space, a track id, or neither and
    carries its values in `features`. A cue with `stream` None reaches every
    stream, so it carries no region."""

    kind: str
    stream: StreamKey | None = None
    region: Region | None = None
    track_id: str | None = None
    score: float = 1.0
    source: str = ""
    features: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Step:
    """One tick of sensor time: one or more aligned windows.

    Two kinds of prior ride along, and a method may read or ignore either.
    `cues` come from outside the chain, set by the source. `feedback` holds
    the chain's own outputs of the last step it ran, set by the chain only
    for the stage kinds a method's manifest names in `feedback`; every other
    part, and the whole of it for any other method, stays empty.
    """

    index: int
    t_start_us: int
    t_end_us: int
    windows: Mapping[StreamKey, Window]
    cues: tuple[Cue, ...] = ()
    feedback: Output = NO_FEEDBACK

    @property
    def duration_us(self) -> int:
        return self.t_end_us - self.t_start_us

    def only(self, stream: StreamKey) -> Step:
        """The same step restricted to one stream's window and cues."""
        return replace(
            self,
            windows={stream: self.windows[stream]},
            cues=tuple(c for c in self.cues if c.stream in (stream, None)),
        )


@dataclass(frozen=True, slots=True)
class Fault:
    """A surfaced `StageFault`, as data in the result stream."""

    stage: str
    stream: StreamKey | None
    step_index: int
    message: str


# Evidence: view name -> track id -> that view's value for the track.
Evidence = Mapping[str, Mapping[str, object]]
