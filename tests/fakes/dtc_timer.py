"""A DTC `Timer` that only moves when read.

Every `wall_ns()` read advances the wall clock by `tick_ns` and every
`cpu_ns()` read advances the CPU clock by `cpu_tick_ns`, so a span between
two reads of the same clock is exactly one tick. A stage timed by the chain
therefore takes a known, deterministic time and no real clock is touched.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakeTimer:
    tick_ns: int = 1_000
    cpu_tick_ns: int = 400
    wall: int = 0
    cpu: int = 0

    def wall_ns(self) -> int:
        self.wall += self.tick_ns
        return self.wall

    def cpu_ns(self) -> int:
        self.cpu += self.cpu_tick_ns
        return self.cpu
