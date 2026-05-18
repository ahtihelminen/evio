"""Time-source abstraction for evio.

`TimeSource` decouples the rest of evio from how wall-clock time is
disciplined. Today's implementations are `HostSystemClock` (just
`time.time_ns()`) and `ExternalDisciplinedClock` (same call, but the
appliance is being disciplined by something upstream — NTP, PTP, a
GNSS-disciplined NIC). When the IMU PCB lands, an `ImuPcbClock` impl
plugs in here without touching call sites.
"""

from evio.time.source import (
    ExternalDisciplinedClock,
    HostSystemClock,
    TimeSource,
    TimeSourceStatus,
)

__all__ = [
    "ExternalDisciplinedClock",
    "HostSystemClock",
    "TimeSource",
    "TimeSourceStatus",
]
