"""A DTC pipeline double whose result says how many steps it has seen.

The result's `provenance` holds the running step count and its `tracks` are
empty, so a driver that reorders, skips or repeats steps shows up in the
results. It keeps the real pipeline's property that stepping is stateful and
order-dependent; `reset` returns it to the start.
"""

from __future__ import annotations

from evio.dtc.result import StepResult
from evio.dtc.taps import TapBus
from evio.dtc.timing import TimingStats
from evio.dtc.types import Step


class CountingPipeline:
    def __init__(self) -> None:
        self.provenance = "counting"
        self.bus = TapBus()
        self.stats: dict[str, TimingStats] = {}
        self.seen: list[int] = []

    def step(self, step: Step) -> StepResult:
        self.seen.append(step.index)
        return StepResult(
            step.index,
            step.t_start_us,
            step.t_end_us,
            candidates=(),
            tracks=(),
            live_ids=frozenset(),
            verdicts=(),
            alarms=(),
            status={},
            faults=(),
            timings={},
            provenance=f"{len(self.seen)}",
        )

    def reset(self) -> None:
        self.seen.clear()
