# Time Module (`src/evio/time`)

This document describes the time-source abstraction under `src/evio/time`.

Files covered:

- `src/evio/time/source.py`
- `src/evio/time/__init__.py`

## What this module owns

The time package owns how the rest of evio asks for wall-clock time:

1. The `TimeSource` abstract class, whose `now_ns()` returns the wall-clock
   UTC nanosecond timestamp and whose `status()` reports how the clock is
   disciplined.
2. `TimeSourceStatus`, the status value surfaced to the fusion node.
3. The host-clock implementations: `HostSystemClock` and
   `ExternalDisciplinedClock` (both `time.time_ns()`).

It does not own:

- disciplining the clock: NTP, PTP or GNSS setup is done upstream of evio,
- event timestamps inside a recording or `EventsBatch`, which are sensor time
  (`src/evio/core/*`),
- stamping reports: the SAPIENT mapper calls `now_ns()` but lives elsewhere,
- a hardware-locked clock: none exists yet.
