"""Per-stage timing: wall and CPU time per step, against the window length.

The timer is injectable, so a test (or a replay) controls the clock and the
step path itself never reads one.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Timer(Protocol):
    def wall_ns(self) -> int: ...

    def cpu_ns(self) -> int: ...


class SystemTimer:
    def wall_ns(self) -> int:
        return time.perf_counter_ns()

    def cpu_ns(self) -> int:
        return time.process_time_ns()


@dataclass(frozen=True, slots=True)
class StageTiming:
    wall_ns: int
    cpu_ns: int


class TimingStats:
    """Rolling wall-time samples of one stage, in nanoseconds."""

    def __init__(self, window_us: int, keep: int = 10_000) -> None:
        self._window_us = window_us
        self._wall_ns: deque[int] = deque(maxlen=keep)
        self._cpu_ns: deque[int] = deque(maxlen=keep)

    def record(self, timing: StageTiming) -> None:
        self._wall_ns.append(timing.wall_ns)
        self._cpu_ns.append(timing.cpu_ns)

    @property
    def count(self) -> int:
        return len(self._wall_ns)

    def percentile_ns(self, q: float) -> float:
        return float(np.percentile(self._wall_ns, q)) if self._wall_ns else 0.0

    @property
    def real_time_factor(self) -> float:
        """Mean wall time over window length; under 1.0 keeps up."""
        if not self._wall_ns:
            return 0.0
        return float(np.mean(self._wall_ns)) / (self._window_us * 1_000)
