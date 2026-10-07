"""A stand-in for the `time` module that only moves when told to.

Installed in place of a module's `time` name, so `perf_counter()` and
`monotonic()` read `now` and `sleep(seconds)` advances it by that much
instead of blocking -- by at least a nanosecond, as a real clock never
stands still, so a sleep loop cannot stall on a remainder smaller than
the float spacing at `now`.
A test moves time between batches by assigning `now`. With `max_sleeps`
set, a sleep past that many raises, so a loop that never ends fails fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeClock:
    now: float = 0.0
    sleeps: list[float] = field(default_factory=list)
    max_sleeps: int | None = None

    def perf_counter(self) -> float:
        return self.now

    monotonic = perf_counter

    def sleep(self, seconds: float) -> None:
        assert self.max_sleeps is None or len(self.sleeps) < self.max_sleeps, (
            "sleep loop did not terminate"
        )
        self.sleeps.append(seconds)
        self.now += max(seconds, 1e-9)
