"""Fidelity for the pacer doubles in `tests/fakes/pacer.py`.

`FakePacer` and `FakeStreamingPacer` stand in for `Pacer` and
`StreamingPacer` in the HUD tests. These tests fail when a fake and its
real class drift apart.
"""

from __future__ import annotations

import numpy as np
import pytest
from fakes.clock import FakeClock
from fakes.pacer import FakePacer, FakePacerConfig, FakeStreamingPacer

from evio.core import streaming_pacer as streaming_pacer_module
from evio.core.events import EventsBatch
from evio.core.pacer import Pacer
from evio.core.streaming_pacer import StreamingPacer

pytestmark = pytest.mark.t1


def _public_properties(cls: type) -> set[str]:
    return {
        name for name, member in vars(cls).items()
        if isinstance(member, property) and not name.startswith("_")
    }


def _batch(start_us: int, end_us: int) -> EventsBatch:
    stamps = np.array([start_us, end_us], np.int64)
    zeros = np.zeros(2, np.int16)
    return EventsBatch(stamps, zeros, zeros, np.zeros(2, np.int8))


def test_fake_pacers_expose_every_public_property_of_the_real_classes() -> None:
    """Deliberate difference: the fakes hold each property as a plain
    settable field, where the real classes compute a read-only property
    from their pacing state."""
    for real, fake in (
        (Pacer, FakePacer()),
        (StreamingPacer, FakeStreamingPacer()),
    ):
        assert _public_properties(real) - set(dir(fake)) == set(), real


def test_fake_streaming_pacer_values_match_the_real_class_for_a_scripted_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliberate differences, as `tests/fakes/pacer.py` states:
    `wall_elapsed` is the fixed `wall_s` rather than a clock reading, and
    `FakePacerConfig` has no `drop_tolerance`, so only the config fields
    the HUD reads are compared."""
    clock = FakeClock(now=10.0)
    monkeypatch.setattr(streaming_pacer_module, "time", clock)
    ends = (1_500_000, 2_000_000, 2_500_000, 3_500_000)
    batches = [
        _batch(start, end)
        for start, end in zip((1_000_000, *ends[:-1]), ends, strict=True)
    ]
    real = StreamingPacer(speed=2.0, force_speed=True, drop_tolerance_ms=100)
    paced = real.pace(batches)
    next(paced)  # 1st waited for until wall 0.25 s: 0.5 s event / 0.25 s
    clock.now = 11.125  # 2nd and 3rd overdue and dropped; 4th waited for
    list(paced)  # until wall 1.25 s: 2 drops over 2 s event / 1 s wall
    clock.now = 11.5

    fake = FakeStreamingPacer(
        config=FakePacerConfig(speed=2.0, force_speed=True),
        wall_s=1.5,
        rec_start_us=1_000_000,
        dropped_batches=2,
        emitted_batches=2,
        instantaneous_speed=2.0,
        instantaneous_drop_rate=0.001,
        average_drop_rate=0.0005,
    )

    for pacer in (real, fake):
        assert (pacer.config.speed, pacer.config.force_speed) == (2.0, True)
    assert real.wall_elapsed() == fake.wall_elapsed()
    for end_ts_us in (3_500_000, 4_000_000):
        assert (
            real.recording_elapsed(end_ts_us)
            == fake.recording_elapsed(end_ts_us)
        )
        assert (
            real.actual_playback_speed(end_ts_us)
            == fake.actual_playback_speed(end_ts_us)
        )
    for name in sorted(_public_properties(StreamingPacer)):
        assert getattr(real, name) == getattr(fake, name), name
