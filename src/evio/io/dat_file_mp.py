from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import TracebackType

import numpy as np

from evio.codec.dt8 import Dt8Codec, find_dat_data_offset
from evio.core.events import EventsBatch
from evio.core.window import RollingWindowBatchProcessor
from evio.io.mp_file import MultiprocessingFileSource


@dataclass
class MultiprocessingDatFileSource:
    """Multiprocessing DAT (DT8) source that yields windowed EventsBatch.

    This is a *thin* wrapper around `MultiprocessingFileSource` that:

    - Computes the DAT data offset (like `open_dat_dt8`).
    - Uses `Dt8Codec` to decode the 8-byte CD events from raw uint16 words.
    - Uses `RollingWindowBatchProcessor` to group events into
      `window_length`-wide rolling time windows before they hit the consumer.

    Typical usage:

        src = MultiprocessingDatFileSource(
            path="events.dat",
            width=1280,
            height=720,
            window_length=1000,
            step_us=16667,
        )

        for batch in src:
            ...  # batch is EventsBatch for one time window
    """

    path: Path | str
    window_length: int
    step_us: int
    chunk_words: int = 1 * 1024 * 1024
    queue_size: int = 8
    width: int = 1280
    height: int = 720

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(path)

        data_offset = find_dat_data_offset(path)

        # Underlying generic MP file source:
        batch_processor_factory = partial(
            RollingWindowBatchProcessor,
            window_length=self.window_length,
            step_us=self.step_us,
        )

        self._inner = MultiprocessingFileSource(
            path=path,
            codec_factory=partial(
                Dt8Codec, width=self.width, height=self.height
            ),
            data_offset=data_offset,
            chunk_words=self.chunk_words,
            dtype=np.dtype("<u2"),  # uint16 words
            batch_processor_factory=batch_processor_factory,
            queue_size=self.queue_size,
        )

    # --- Public surface: delegate to the inner source ---------------------

    def __iter__(self) -> Iterator[EventsBatch]:
        return iter(self._inner)

    def get_batch_nowait(self) -> tuple[EventsBatch | None, bool]:
        return self._inner.get_batch_nowait()

    def close(self) -> None:
        self._inner.close()

    def __enter__(self) -> MultiprocessingDatFileSource:
        self._inner.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._inner.__exit__(exc_type, exc, tb)
