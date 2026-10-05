"""StreamingPacer: replays EventsBatches at a speed-scaled wall-clock rate.

Clock readings are binary fractions of a second, so the pacer's integer
microsecond arithmetic is exact and the expected values are hand-computed.
"""

from __future__ import annotations

import numpy as np
import pytest
from fakes.clock import FakeClock

from evio.core import streaming_pacer as streaming_pacer_module
from evio.core.events import EventsBatch
from evio.core.streaming_pacer import StreamingPacer

pytestmark = pytest.mark.t1


def _batch(start_us: int, end_us: int) -> EventsBatch:
    stamps = np.array([start_us, end_us], np.int64)
    zeros = np.zeros(2, np.int16)
    return EventsBatch(stamps, zeros, zeros, np.zeros(2, np.int8))


def _batches(*end_us: int) -> list[EventsBatch]:
    """Consecutive batches starting at 1 s of event time."""
    starts = (1_000_000, *end_us[:-1])
    return [
        _batch(start, end) for start, end in zip(starts, end_us, strict=True)
    ]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock(now=10.0)
    monkeypatch.setattr(streaming_pacer_module, "time", clock)
    return clock


def test_streaming_pacer_waits_when_early(clock: FakeClock) -> None:
    batches = _batches(1_500_000, 2_000_000)
    pacer = StreamingPacer(speed=2.0, force_speed=False)

    emitted_at = [clock.now for _ in pacer.pace(batches)]

    # 0.5 s of event time per batch at speed 2 -> 0.25 s of wall time.
    assert emitted_at == [10.25, 10.5]
    assert pacer.emitted_batches == 2
    assert pacer.dropped_batches == 0


def test_streaming_pacer_drops_overdue_batches(clock: FakeClock) -> None:
    batches = _batches(1_250_000, 1_500_000, 1_750_000, 2_000_000)
    pacer = StreamingPacer(speed=1.0, force_speed=True, drop_tolerance_ms=100)
    paced = pacer.pace(batches)

    first = next(paced)  # target 0.25 s: waited for
    clock.now = 10.875  # 2nd 0.375 s and 3rd 0.125 s late; 4th 0.125 s early
    rest = list(paced)

    assert [first, *rest] == [batches[0], batches[3]]
    assert clock.now == 11.0
    assert pacer.dropped_batches == 2
    assert pacer.emitted_batches == 2


def test_streaming_pacer_reports_elapsed_and_speed(clock: FakeClock) -> None:
    pacer = StreamingPacer(speed=2.0, force_speed=False)
    list(pacer.pace(_batches(1_500_000)))  # anchors at wall 10 s, event 1 s

    clock.now = 10.5

    assert pacer.wall_elapsed() == 0.5
    assert pacer.recording_elapsed(2_000_000) == 1.0
    assert pacer.actual_playback_speed(2_000_000) == 2.0
