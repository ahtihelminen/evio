from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from multiprocessing import Event as MpEvent
from multiprocessing import Process, Queue
from multiprocessing.synchronize import Event as MpSyncEvent
from pathlib import Path
from types import TracebackType
from typing import Protocol

import numpy as np

from evio.codec.codec_protocol import EventCodec
from evio.core.events import EventsBatch


def _sentinel_factory() -> object:
    """Return module-level sentinel — helps pickle keep identity."""
    return _SENTINEL


class _Sentinel:
    def __reduce__(self) -> tuple[Callable[[], object], tuple[()]]:
        return (_sentinel_factory, ())


_SENTINEL = _Sentinel()  # unique end-of-stream marker


class BatchProcessor(Protocol):
    """Optional stateful post-processing on decoded batches."""

    def process(
        self, batches: Iterable[EventsBatch], *, final: bool = False
    ) -> Iterable[EventsBatch]: ...


class _IdentityBatchProcessor:
    def process(
        self, batches: Iterable[EventsBatch], *, final: bool = False
    ) -> Iterable[EventsBatch]:
        return batches


def _reader_process(
    path: str,
    data_offset: int,
    chunk_words: int,
    dtype_str: str,
    in_queue: Queue,
    stop_event: MpSyncEvent,
) -> None:
    """Read raw words from file and push them to the worker input queue."""
    dtype = np.dtype(dtype_str)

    with Path(path).open("rb") as file:
        file.seek(data_offset)
        itemsize = dtype.itemsize
        bytes_per_chunk = chunk_words * itemsize

        while not stop_event.is_set():
            raw = file.read(bytes_per_chunk)
            if not raw:
                break

            words = np.frombuffer(raw, dtype=dtype)
            if words.size == 0:
                break

            in_queue.put(words)

    in_queue.put(_SENTINEL)


def _worker_proc(
    codec_factory: Callable[[], EventCodec],
    in_queue: Queue,
    out_queue: Queue,
    stop_event: MpSyncEvent,
    batch_processor_factory: Callable[[], BatchProcessor] | None,
) -> None:
    """Decode words from input queue with codec and push EventsBatch out."""
    codec = codec_factory()
    processor = (
        batch_processor_factory()
        if batch_processor_factory is not None
        else _IdentityBatchProcessor()
    )

    while not stop_event.is_set():
        item = in_queue.get()
        if item is _SENTINEL:
            for batch in processor.process((), final=True):
                out_queue.put(batch)
            break

        decoded_batches = codec.decode_chunk(item)
        for batch in processor.process(decoded_batches):
            out_queue.put(batch)

    out_queue.put(_SENTINEL)


@dataclass
class MultiprocessingFileSource:
    """Generic MP file source that uses an EventCodec for decoding."""

    path: Path
    codec_factory: Callable[[], EventCodec]
    data_offset: int
    chunk_words: int
    dtype: np.dtype[np.uint16]
    batch_processor_factory: Callable[[], BatchProcessor] | None = None
    queue_size: int = 8

    def __post_init__(self) -> None:
        self._in_queue: Queue = Queue(maxsize=self.queue_size)
        self._out_queue: Queue = Queue(maxsize=self.queue_size)
        self._stop_event: MpSyncEvent = MpEvent()

        self._reader = Process(
            target=_reader_process,
            args=(
                str(self.path),
                self.data_offset,
                self.chunk_words,
                self.dtype.str,
                self._in_queue,
                self._stop_event,
            ),
            daemon=True,
        )
        self._worker = Process(
            target=_worker_proc,
            args=(
                self.codec_factory,
                self._in_queue,
                self._out_queue,
                self._stop_event,
                self.batch_processor_factory,
            ),
            daemon=True,
        )

        self._started = False
        self._closed = False

    def _start(self) -> None:
        if self._started:
            return
        self._reader.start()
        self._worker.start()
        self._started = True

    def __iter__(self) -> Iterator[EventsBatch]:
        self._start()
        try:
            while True:
                item = self._out_queue.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            self.close()

    def close(self) -> None:
        # Double close prevention
        if self._closed:
            return
        self._closed = True

        self._stop_event.set()

        # Drain the queue until empty
        while not self._out_queue.empty():
            _ = self._out_queue.get_nowait()

        for process in (self._reader, self._worker):
            if process.is_alive():
                process.terminate()

        self._in_queue.close()
        self._out_queue.close()

    def __enter__(self) -> MultiprocessingFileSource:
        self._start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
