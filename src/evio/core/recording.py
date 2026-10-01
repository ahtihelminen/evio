from dataclasses import dataclass
from pathlib import Path

import numpy as np

DT8 = np.dtype([("t32", "<u4"), ("w32", "<u4")])


@dataclass(frozen=True)
class Recording:
    width: int
    height: int
    timestamps: np.ndarray  # int64, sorted monotonic, [microseconds]
    event_words: np.ndarray  # each word is uint32 memmap view of
    # packed XY+polarity
    order: np.ndarray  # int32 permutation mapping sort order


def memmap_dat_dt8(
    path: str | Path, *, data_offset: int | None = None,
) -> np.memmap:
    """Memmap a DAT8 file's ``[(t32,u4),(w32,u4)]`` event records (lazy).

    Skips the leading ``%`` header lines + 2-byte type/size when
    ``data_offset`` is None. Reads nothing beyond the header — the returned
    memmap pages in event data only as it is indexed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    off = 0
    if data_offset is None:
        with Path.open(path, "rb") as file:
            while True:
                pos = file.tell()
                first_byte = file.read(1)
                if not first_byte or first_byte != b"%":
                    file.seek(pos)
                    break
                file.readline()  # consume the header line
            type_b = file.read(1)  # two bytes: type + decoded size
            size_b = file.read(1)
            if len(type_b) != 1 or len(size_b) != 1:
                msg = "Invalid DAT header"
                raise RuntimeError(msg)
            if int.from_bytes(size_b, "little") != 8:
                msg = "Only 8B CD events supported"
                raise RuntimeError(msg)
            off = file.tell()
    else:
        off = int(data_offset)
    return np.memmap(path, dtype=DT8, mode="r", offset=off)


def open_dat_dt8(
    path: str | Path,
    *,
    width: int,
    height: int,
    data_offset: int | None = None,
) -> Recording:
    """Whole-file loader: memmap [(t32,u4),(w32,u4)], unwrap + time order.

    Materialises every timestamp; for windowed replay memmap the file with
    ``memmap_dat_dt8`` and slice on demand.
    """
    events = memmap_dat_dt8(path, data_offset=data_offset)

    timestamps = events["t32"].astype(np.int64, copy=False)
    if timestamps.size == 0:
        msg = "No events"
        raise RuntimeError(msg)
    # The DAT ``t32`` field is a 32-bit µs counter that wraps every 2**32 µs
    # (~71.6 min). Events are stored in acquisition order, so each large
    # backward jump marks a wrap; unwrap to a monotonic timeline before the
    # sort (otherwise sorting interleaves pre- and post-wrap events and the
    # wall-clock anchor breaks for any recording longer than one wrap period).
    for wrap_idx in np.flatnonzero(np.diff(timestamps) < -(2 ** 31)):
        timestamps[int(wrap_idx) + 1:] += 2 ** 32
    order = np.argsort(timestamps, kind="stable").astype(np.int32, copy=False)
    timestamps_sorted = timestamps[order]
    event_words = events["w32"]

    return Recording(
        width=width,
        height=height,
        timestamps=timestamps_sorted,
        event_words=event_words,
        order=order,
    )
