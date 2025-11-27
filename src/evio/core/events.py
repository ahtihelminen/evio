from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class BatchRange(NamedTuple):
    """Lightweight slice into the time-ordered event stream."""

    start_idx: int  # inclusive index into time_ordered stream
    end_idx: int  # exclusive index into time_ordered stream
    start_ts_us: int  # best-effort start timestamp (microseconds)
    end_ts_us: int  # best-effort end timestamp (microseconds)


@dataclass
class EventsBatch:
    timestamps: NDArray[np.int64]
    x_coords: NDArray[np.int16]
    y_coords: NDArray[np.int16]
    polarities: NDArray[np.int8]
