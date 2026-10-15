"""What a pipeline emits per step, and the interface the runtime offers.

Results carry a status per stream, surfaced faults, per-stage timing and a
provenance hash of the spec and method versions, so a stored result says what
produced it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from evio.dtc.taps import TapBus
from evio.dtc.timing import StageTiming, TimingStats
from evio.dtc.types import (
    Alarm,
    Candidate,
    Fault,
    Status,
    Step,
    StreamKey,
    Track,
    Verdict,
)


class FaultPolicy(StrEnum):
    """FIELD isolates a faulting stage, counts and surfaces the fault with a
    degraded status. RESEARCH fails fast by raising the `StageFault`."""

    FIELD = "field"
    RESEARCH = "research"


@dataclass(frozen=True, slots=True)
class StepResult:
    """Everything one step produced, across streams.

    `status` is the worst status among a stream's stages. `timings` maps a
    stage name to its wall and CPU time this step, summed over streams.
    """

    index: int
    t_start_us: int
    t_end_us: int
    candidates: tuple[Candidate, ...]
    tracks: tuple[Track, ...]
    live_ids: frozenset[str]
    verdicts: tuple[Verdict, ...]
    alarms: tuple[Alarm, ...]
    status: Mapping[StreamKey, Status]
    faults: tuple[Fault, ...]
    timings: Mapping[str, StageTiming]
    provenance: str


class Pipeline(Protocol):
    """A built chain, ready to take steps in order."""

    provenance: str
    bus: TapBus
    stats: Mapping[str, TimingStats]

    def step(self, step: Step) -> StepResult: ...

    def reset(self) -> None: ...
