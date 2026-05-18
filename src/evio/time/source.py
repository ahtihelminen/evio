"""TimeSource ABC + initial host-clock implementations.

`now_ns()` is the only method the rest of evio cares about — it returns the
wall-clock UTC nanosecond timestamp the SAPIENT mapper will stamp onto a
DetectionReport. `status()` is what `StatusReport.health()` surfaces so the
fusion node can tell whether our timestamps are free-running, NTP-disciplined,
or hardware-locked.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

TimeSourceKind = Literal["host", "external", "imu_pcb"]


@dataclass(frozen=True, slots=True)
class TimeSourceStatus:
    """Snapshot of the active clock's discipline state."""

    kind: TimeSourceKind
    locked: bool
    source_id: str = ""


class TimeSource(ABC):
    @abstractmethod
    def now_ns(self) -> int:
        """Current UTC time in nanoseconds since the Unix epoch."""

    @abstractmethod
    def status(self) -> TimeSourceStatus:
        """Current discipline state for `StatusReport`."""


class HostSystemClock(TimeSource):
    """Uses `time.time_ns()` directly. No external discipline assumed."""

    def now_ns(self) -> int:
        return time.time_ns()

    def status(self) -> TimeSourceStatus:
        return TimeSourceStatus(kind="host", locked=False)


class ExternalDisciplinedClock(TimeSource):
    """Reads the host clock but declares it as externally disciplined.

    Use when the appliance is provisioned with NTP / PTP / GNSS-disciplined NIC
    or chrony peer. The discipline happens in the OS; we just label the source
    so SAPIENT downstream knows our timestamps are trustworthy.
    """

    def __init__(self, source_id: str) -> None:
        self._source_id = source_id

    def now_ns(self) -> int:
        return time.time_ns()

    def status(self) -> TimeSourceStatus:
        return TimeSourceStatus(
            kind="external", locked=True, source_id=self._source_id,
        )
