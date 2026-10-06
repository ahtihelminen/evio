"""`MultiprocessingFileSource` over a synthetic `.dat`, with real processes."""

from __future__ import annotations

import multiprocessing
import time
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from support.dat import events_at, write_dat

from evio.codec.dt8 import Dt8Codec, find_dat_data_offset
from evio.core.events import EventsBatch
from evio.core.window import RollingWindowBatchProcessor
from evio.io.mp_file import MultiprocessingFileSource

pytestmark = pytest.mark.t2


def _source(
    dat: Path, *, chunk_words: int, **kwargs: object,
) -> MultiprocessingFileSource:
    return MultiprocessingFileSource(
        path=dat,
        codec_factory=partial(Dt8Codec, width=1280, height=720),
        data_offset=find_dat_data_offset(dat),
        chunk_words=chunk_words,
        dtype=np.dtype("<u2"),
        **kwargs,
    )


def _drain(source: MultiprocessingFileSource) -> list[EventsBatch]:
    """Poll until end of stream; fail rather than hang if it never comes."""
    batches: list[EventsBatch] = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        batch, ended = source.get_batch_nowait()
        if ended:
            return batches
        if batch is None:
            time.sleep(0.001)
        else:
            batches.append(batch)
    pytest.fail("no end of stream within 30 s")


def _concat(batches: list[EventsBatch]) -> EventsBatch:
    return EventsBatch(
        *(np.concatenate([getattr(b, field) for b in batches])
          for field in ("timestamps", "x_coords", "y_coords", "polarities"))
    )


def test_mp_file_source_delivers_all_events_in_order_then_ends(
    tmp_path: Path,
) -> None:
    original = events_at(list(range(0, 50_000, 10)))
    dat = write_dat(tmp_path / "events.dat", original)
    source = _source(dat, chunk_words=1000)

    delivered = _concat(_drain(source))

    np.testing.assert_array_equal(delivered.timestamps, original.timestamps)
    np.testing.assert_array_equal(delivered.x_coords, original.x_coords)
    np.testing.assert_array_equal(delivered.y_coords, original.y_coords)
    np.testing.assert_array_equal(delivered.polarities, original.polarities)
    assert source.get_batch_nowait() == (None, True)


def test_mp_file_source_applies_batch_processor(tmp_path: Path) -> None:
    original = events_at(list(range(0, 50_000, 10)))
    dat = write_dat(tmp_path / "events.dat", original)
    windowing = partial(
        RollingWindowBatchProcessor, window_length=1000, step_us=1000,
    )
    source = _source(dat, chunk_words=1000, batch_processor_factory=windowing)

    delivered = _drain(source)

    expected = list(windowing().process([original], final=True))
    assert len(delivered) == len(expected) == 50
    for got, want in zip(delivered, expected, strict=True):
        np.testing.assert_array_equal(got.timestamps, want.timestamps)


def test_mp_file_source_close_stops_the_worker(tmp_path: Path) -> None:
    dat = write_dat(
        tmp_path / "events.dat", events_at(list(range(200_000))),
    )
    source = _source(dat, chunk_words=1000, queue_size=2)
    deadline = time.monotonic() + 30
    batch, ended = source.get_batch_nowait()
    while batch is None and time.monotonic() < deadline:
        time.sleep(0.001)
        batch, ended = source.get_batch_nowait()
    assert batch is not None
    assert not ended
    assert multiprocessing.active_children()

    source.close()

    assert multiprocessing.active_children() == []


def test_mp_file_source_unaligned_chunks_lose_no_events(
    tmp_path: Path,
) -> None:
    original = events_at(list(range(0, 50_000, 10)))
    dat = write_dat(tmp_path / "events.dat", original)
    source = _source(dat, chunk_words=1001)

    delivered = _concat(_drain(source))

    np.testing.assert_array_equal(delivered.timestamps, original.timestamps)
    np.testing.assert_array_equal(delivered.x_coords, original.x_coords)
