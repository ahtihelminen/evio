"""Write synthetic DT8 `.dat` files: a `%` header, type/size bytes, events."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evio.codec.dt8 import encode_dt8_pairs
from evio.core.events import EventsBatch


def events_at(timestamps: list[int], *, width: int = 1280) -> EventsBatch:
    """One event per timestamp, with a distinct x so order is checkable."""
    count = len(timestamps)
    return EventsBatch(
        timestamps=np.asarray(timestamps, dtype=np.int64),
        x_coords=(np.arange(count) % width).astype(np.int16),
        y_coords=np.zeros(count, dtype=np.int16),
        polarities=(np.arange(count) % 2).astype(np.int8),
    )


def write_dat(
    path: Path, batch: EventsBatch, *, width: int = 1280, height: int = 720,
) -> Path:
    t32, w32 = encode_dt8_pairs(
        batch.timestamps, batch.x_coords, batch.y_coords, batch.polarities,
        width, height,
    )
    records = np.empty(t32.size, dtype=[("t32", "<u4"), ("w32", "<u4")])
    records["t32"], records["w32"] = t32, w32
    path.write_bytes(b"% synthetic\n" + bytes([0, 8]) + records.tobytes())
    return path
