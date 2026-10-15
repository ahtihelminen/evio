"""Builders for DTC steps and windows in tests.

Pure numpy; sensor time only, no clock.
"""

from __future__ import annotations

import numpy as np

from evio.core.events import EventsBatch
from evio.dtc.types import Step, Window


def events(
    t_us: list[int], x: list[int], y: list[int], polarity: int = 1
) -> EventsBatch:
    return EventsBatch(
        timestamps=np.asarray(t_us, dtype=np.int64),
        x_coords=np.asarray(x, dtype=np.int16),
        y_coords=np.asarray(y, dtype=np.int16),
        polarities=np.full(len(t_us), polarity, dtype=np.int8),
    )


def empty_events() -> EventsBatch:
    return events([], [], [])


def blob_events(
    t_start_us: int,
    t_end_us: int,
    cx: int,
    cy: int,
    *,
    count: int = 200,
    radius: int = 3,
    seed: int = 0,
) -> EventsBatch:
    """`count` events of a small blob around (cx, cy), time-sorted."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(t_start_us, t_end_us, count))
    x = cx + rng.integers(-radius, radius + 1, count)
    y = cy + rng.integers(-radius, radius + 1, count)
    return events(t.tolist(), x.tolist(), y.tolist())


def window(
    stream: str,
    batch: EventsBatch,
    t_start_us: int,
    t_end_us: int,
    extent: tuple[int, int] | None = (1280, 720),
) -> Window:
    return Window(stream, batch, t_start_us, t_end_us, extent)


def step_of(index: int, step_us: int, *windows: Window) -> Step:
    return Step(
        index,
        index * step_us,
        (index + 1) * step_us,
        {w.stream: w for w in windows},
    )
