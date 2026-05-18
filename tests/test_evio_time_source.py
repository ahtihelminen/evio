"""TimeSource: smoke tests on the two host-backed implementations."""

from __future__ import annotations

import time

from evio.time import ExternalDisciplinedClock, HostSystemClock


def test_host_system_clock_is_monotonic_within_resolution() -> None:
    clock = HostSystemClock()
    first = clock.now_ns()
    time.sleep(0.001)
    second = clock.now_ns()
    assert second >= first
    assert clock.status().kind == "host"
    assert clock.status().locked is False


def test_external_disciplined_clock_advertises_source_id() -> None:
    clock = ExternalDisciplinedClock(source_id="chrony@ntp.pool.org")
    status = clock.status()
    assert status.kind == "external"
    assert status.locked is True
    assert status.source_id == "chrony@ntp.pool.org"
    assert clock.now_ns() > 0
