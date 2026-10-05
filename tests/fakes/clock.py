"""A stand-in for the `time` module that only moves when told to.

Installed in place of a module's `time` name, so `perf_counter()` reads
`now` and `sleep(seconds)` advances it by that much instead of blocking --
by at least a nanosecond, as a real clock never stands still, so a sleep
loop cannot stall on a remainder smaller than the float spacing at `now`.
A test moves time between batches by assigning `now`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeClock:
    now: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += max(seconds, 1e-9)
