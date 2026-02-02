from collections.abc import Iterator

import numpy as np

from evio.core.events import BatchRange
from evio.core.recording import open_dat_dt8


class DatFileSource:
    """
    Minimal file-backed source for event batches.

    - Opens a .dat file with fixed geometry (defaults to 1280x720).
    - Precomputes time windows. If step_us is set, windows are rolling and
      left-aligned; otherwise they are non-overlapping.
    - Exposes:
        * words_u32 : np.ndarray  (packed uint32 words)
        * order     : np.ndarray  (time-sorted indices)
        * width, height : ints
        * ranges()  : iterator of BatchRange
    - Does NOT decode or sleep; decoding happens downstream in the render path.
    """

    def __init__(
        self,
        path: str,
        window_length_us: int,
        step_us: int,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        rec = open_dat_dt8(path, width=width, height=height)
        self._event_words = rec.event_words
        self._order = rec.order
        self._width, self._height = int(rec.width), int(rec.height)

        t_sorted = getattr(rec, "timestamps", getattr(rec, "ts_us", None))
        if t_sorted is None:
            msg = "Recording missing timestamps ('timestamps' or 'ts_us')."
            raise AttributeError(msg)
        t_sorted = np.asarray(t_sorted).astype(np.int64, copy=False)
        if t_sorted.shape[0] != self._event_words.shape[0]:
            msg = "timestamps length must match event count"
            raise ValueError(msg)

        t_start = t_sorted[0]
        t_end = t_sorted[-1]
        start_times = np.arange(
            t_start, t_end + step_us, step_us, dtype=np.int64
        )
        end_times = start_times + window_length_us
        starts = np.searchsorted(t_sorted, start_times, side="left").astype(
            np.int32
        )
        stops = np.searchsorted(t_sorted, end_times, side="left").astype(
            np.int32
        )

        self._ranges = [
            BatchRange(
                start_idx,
                end_idx,
                int(start_time),
                int(end_time),
            )
            for start_idx, end_idx, start_time, end_time in zip(
                starts, stops, start_times, end_times, strict=True
            )
        ]

    # --- Public surface ----------------------------------------------------

    @property
    def event_words(self) -> np.ndarray:
        """Packed event words (uint32)."""
        return self._event_words

    @property
    def order(self) -> np.ndarray:
        """Time-sorted indices into the raw event arrays."""
        return self._order

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def ranges(self) -> Iterator[BatchRange]:
        """Iterate precomputed BatchRange slices."""
        return iter(self._ranges)

    def __len__(self) -> int:
        """Number of precomputed ranges."""
        return len(self._ranges)
