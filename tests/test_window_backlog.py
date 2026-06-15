"""Drop-to-latest window-backlog collapse for the live display path."""

from __future__ import annotations

from collections import deque

import numpy as np

from evio.core.events import EventsBatch
from evio.core.window import drain_window_backlog


def _batch(n: int) -> EventsBatch:
    """An EventsBatch carrying ``n`` events (only the count matters here)."""
    return EventsBatch(
        x_coords=np.zeros(n, np.int16),
        y_coords=np.zeros(n, np.int16),
        timestamps=np.arange(n, dtype=np.int64),
        polarities=np.zeros(n, np.int8),
    )


def test_noop_when_disabled() -> None:
    queue = deque(_batch(1) for _ in range(9))
    current = _batch(7)
    newest, stale, dropped = drain_window_backlog(queue, current, 0)
    assert newest is current
    assert stale == []
    assert dropped == 0
    assert len(queue) == 9  # untouched


def test_noop_under_threshold() -> None:
    queue = deque([_batch(1), _batch(1)])
    current = _batch(5)
    newest, stale, dropped = drain_window_backlog(queue, current, 2)
    assert newest is current
    assert stale == []
    assert dropped == 0
    assert len(queue) == 2


def test_collapses_to_freshest_over_threshold() -> None:
    windows = [_batch(i) for i in range(1, 5)]  # 1,2,3 events then the newest
    queue = deque(windows)
    current = _batch(10)
    newest, stale, dropped = drain_window_backlog(queue, current, 2)
    # Newest queued window survives; everything older is stale.
    assert newest is windows[-1]
    assert stale == [current, windows[0], windows[1], windows[2]]
    # Dropped count sums the stale windows so event_rate stays honest.
    assert dropped == 10 + 1 + 2 + 3
    assert len(queue) == 0


def test_stale_windows_preserve_record_order() -> None:
    queue = deque([_batch(1), _batch(2), _batch(3)])
    current = _batch(9)
    _, stale, _ = drain_window_backlog(queue, current, 1)
    # current is oldest, then FIFO order — the caller records them in this
    # order so footage stays monotonic.
    assert [int(b.timestamps.size) for b in stale] == [9, 1, 2]
