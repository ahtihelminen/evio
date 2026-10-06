"""Pacer: replays BatchRanges at a speed-scaled wall-clock rate."""

from __future__ import annotations

import pytest
from fakes.clock import FakeClock

from evio.core import pacer as pacer_module
from evio.core.events import BatchRange
from evio.core.pacer import Pacer

pytestmark = pytest.mark.t1


def _ranges(*end_ts_us: int) -> list[BatchRange]:
    """Consecutive ranges starting at 1 s of event time."""
    starts = (1_000_000, *end_ts_us[:-1])
    return [
        BatchRange(0, 0, start, end)
        for start, end in zip(starts, end_ts_us, strict=True)
    ]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock(now=100.0)  # Pacer treats a 0.0 start as unset
    monkeypatch.setattr(pacer_module, "time", clock)
    return clock


def _overdue_run(clock: FakeClock) -> tuple[Pacer, list[BatchRange]]:
    """Emit the first range on time, then fall behind twice."""
    pacer = Pacer(speed=1.0, force_speed=True, drop_tolerance_s=0.05)
    ranges = _ranges(1_100_000, 1_200_000, 1_300_000, 1_400_000, 1_500_000)
    paced = pacer.pace(ranges)
    emitted = [next(paced)]  # target 0.1 s: waited for
    clock.now = 100.43  # 2nd and 3rd 0.23 s / 0.13 s late, 4th 0.03 s
    emitted.append(next(paced))
    clock.now = 102.0  # the last range is 1.5 s late
    emitted.extend(paced)
    return pacer, emitted


def test_pacer_without_force_speed_never_drops_and_waits_for_target(
    clock: FakeClock,
) -> None:
    ranges = _ranges(1_200_000, 1_400_000, 1_600_000, 1_800_000)
    pacer = Pacer(speed=2.0, force_speed=False)
    paced = pacer.pace(ranges)

    emitted, emitted_at = [], []
    for batch_range in paced:
        emitted.append(batch_range)
        emitted_at.append(clock.now)
        if len(emitted) == 2:
            clock.now += 1.0  # fall far behind: still nothing is dropped

    assert emitted == ranges
    assert pacer.dropped_batches == 0
    assert pacer.emitted_batches == 4
    # Event time 0.2 s per range at speed 2 -> 0.1 s of wall time apiece.
    assert emitted_at == pytest.approx([100.1, 100.2, 101.2, 101.2])
    assert all(0 < seconds <= 0.05 for seconds in clock.sleeps)


def test_pacer_force_speed_drops_overdue_batches(clock: FakeClock) -> None:
    pacer, emitted = _overdue_run(clock)

    assert [r.end_ts_us for r in emitted] == [1_100_000, 1_400_000, 1_500_000]
    assert pacer.emitted_batches == 3
    # The overdue last range is counted as dropped, yet still emitted.
    assert pacer.dropped_batches == 3


def test_pacer_reset_stats_zeroes_counters(clock: FakeClock) -> None:
    pacer, _ = _overdue_run(clock)
    assert pacer.instantaneous_drop_rate > 0
    assert pacer.average_drop_rate > 0

    pacer.reset_stats()

    assert pacer.dropped_batches == 0
    assert pacer.emitted_batches == 0
    assert pacer.instantaneous_drop_rate == 0.0
    assert pacer.average_drop_rate == 0.0


def test_pacer_sleep_until_terminates_when_the_clock_starts_at_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock(now=0.0, max_sleeps=100)
    monkeypatch.setattr(pacer_module, "time", clock)
    ranges = _ranges(1_100_000, 1_200_000)

    assert list(Pacer(speed=1.0, force_speed=False).pace(ranges)) == ranges
    assert clock.now == pytest.approx(0.2)
