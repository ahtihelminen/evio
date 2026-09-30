"""Write-then-read through the real recording path: FootageRecorder writes
a `.dat`, MultiprocessingDatFileSource (real forked reader + worker
processes) reads it back windowed. Nothing pinned this before --
plans/test_suite.md section 9.2 item 3 names it as the untested
precondition for T4 (replay e2e), T5 (benchmark) and SN-452 (frozen
dataset).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.io.dat_file_mp import MultiprocessingDatFileSource
from evio.recording import FootageRecorder

pytestmark = pytest.mark.t2

_WIDTH = 640
_HEIGHT = 480


def _random_events(n: int, *, span_us: int, seed: int = 11) -> EventsBatch:
    rng = np.random.default_rng(seed)
    return EventsBatch(
        timestamps=np.sort(
            rng.integers(0, span_us, size=n, dtype=np.int64)
        ),
        x_coords=rng.integers(0, _WIDTH, size=n, dtype=np.int16),
        y_coords=rng.integers(0, _HEIGHT, size=n, dtype=np.int16),
        polarities=rng.integers(0, 2, size=n, dtype=np.int8),
    )


def _record(root: Path, batch: EventsBatch) -> Path:
    recorder = FootageRecorder(
        root=root, camera_id="cam0", width=_WIDTH, height=_HEIGHT,
        time_provider=lambda: 0.0,
    )
    recorder.write(batch)
    path = recorder.current_path
    recorder.close()  # truncates the mmap extent to its true length
    assert path is not None
    return path


def _multiset(batch: EventsBatch) -> Counter[tuple[int, int, int, int]]:
    return Counter(
        zip(
            batch.timestamps.tolist(),
            batch.x_coords.tolist(),
            batch.y_coords.tolist(),
            batch.polarities.tolist(),
            strict=True,
        )
    )


def test_replay_reconstructs_the_written_event_multiset(
    tmp_path: Path,
) -> None:
    span_us = 200_000
    original = _random_events(4000, span_us=span_us)
    dat = _record(tmp_path / "footage", original)

    # window_length == step_us: tiles the timeline with no overlap, so a
    # window boundary can split an event stream but never duplicate or
    # drop an event across two windows.
    replayed: list[EventsBatch] = []
    with MultiprocessingDatFileSource(
        path=dat, window_length=50_000, step_us=50_000,
        width=_WIDTH, height=_HEIGHT,
    ) as source:
        replayed.extend(source)

    replayed_count = sum(b.timestamps.size for b in replayed)
    assert replayed_count == original.timestamps.size

    seen: Counter[tuple[int, int, int, int]] = Counter()
    for batch in replayed:
        window_counts = _multiset(batch)
        overlap = window_counts & seen
        assert not overlap, "an event was replayed twice"
        seen += window_counts

    assert seen == _multiset(original)
