from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Recording:
    width: int
    height: int
    t_us: np.ndarray  # int64, sorted monotonic
    w32: np.ndarray  # uint32 memmap view of packed XY+polarity
    order: np.ndarray  # int32 permutation mapping sort order


def open_dat(
    path: str | Path,
    *,
    width: int,
    height: int,
    data_offset: int | None = None,
    time_unit: str = "auto",
) -> Recording:
    """Minimal loader: memmap [(t32,u4),(w32,u4)], build stable time order.

    If data_offset is None, tries to auto-scan header lines starting with '%'.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # --- header / offset ---
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
            # two bytes: type + decoded size
            type_b = file.read(1)
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

    DT8 = np.dtype([("t32", "<u4"), ("w32", "<u4")])
    m = np.memmap(path, dtype=DT8, mode="r", offset=off)

    t_raw = m["t32"].astype(np.int64, copy=False)
    if t_raw.size == 0:
        msg = "No events"
        raise RuntimeError(msg)
    if time_unit == "auto":
        span = int(t_raw[-1] - t_raw[0]) if t_raw.size > 1 else 0
        unit = (
            "ns" if span > 1_000_000_000 else ("us" if span > 100_000 else "ms")
        )
    else:
        unit = time_unit
    if unit == "ns":
        t_us = (t_raw // 1_000).astype(np.int64, copy=False)
    elif unit == "ms":
        t_us = (t_raw * 1_000).astype(np.int64, copy=False)
    else:
        t_us = t_raw

    order = np.argsort(t_us, kind="stable").astype(np.int32, copy=False)
    t_us_sorted = t_us[order]
    w32 = m["w32"]  # keep as memmap; we'll index with order on demand

    return Recording(
        width=width, height=height, t_us=t_us_sorted, w32=w32, order=order
    )
