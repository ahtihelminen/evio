"""A camera clock reset at stream start must not cost the recording.

The reproduction of ops001, 2026-10-19: the first decoded buffer holds a
stale prefix on the old clock, then the clock restarts near zero. The rolling
window anchored itself at the stale time, so every later event looked older
than the window and 99% of the stream was never emitted.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from fakes.imx636_stream import FakeStreamingCamera, cd_buffer

pytestmark = pytest.mark.t2

STALE_EVENTS = 6_000
LIVE_EVENTS = 100_000
LIVE_STEP_US = 10
POLL_EVENTS = 4_000


def _drain(source: object) -> np.ndarray:
    stamps = []
    while (batch := next(source)).timestamps.size:
        stamps.append(batch.timestamps)
    return np.concatenate(stamps) if stamps else np.zeros(0, np.int64)


def test_a_stale_prefix_does_not_swallow_the_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules, "imx636",
        sys.modules.get("imx636", types.ModuleType("imx636")),
    )
    from evio.io.imx636_camera import Imx636CameraSource
    stale = 3_764_000 + np.arange(STALE_EVENTS) * 4
    live = np.arange(LIVE_EVENTS) * LIVE_STEP_US
    buffers = [cd_buffer(np.concatenate([stale, live[:POLL_EVENTS]]))]
    buffers += [
        cd_buffer(live[start:start + POLL_EVENTS])
        for start in range(POLL_EVENTS, LIVE_EVENTS, POLL_EVENTS)
    ]
    source = Imx636CameraSource(serial="test")
    source.cam = FakeStreamingCamera(buffers)
    source.cam.set_cd_callback(source._on_cd_events)

    emitted = _drain(source)

    assert emitted.size > 0.98 * LIVE_EVENTS
    assert emitted.max() < LIVE_EVENTS * LIVE_STEP_US, "stale events emitted"
    assert np.all(np.diff(emitted) >= 0)
