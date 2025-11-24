from dataclasses import dataclass
from pathlib import Path

import numpy as np

EVT_ADDR_Y = 0x0
EVT_ADDR_X = 0x2
VECT_BASE_X = 0x3
VECT_12 = 0x4
VECT_8 = 0x5
EVT_TIME_LOW = 0x6
EVT_TIME_HIGH = 0x8


@dataclass(frozen=True)
class Recording:
    width: int
    height: int
    timestamps: np.ndarray  # int64, sorted monotonic, [microseconds]
    event_words: np.ndarray  # each word is uint32 memmap view of
    # packed XY+polarity
    order: np.ndarray  # int32 permutation mapping sort order


def open_dat(
    path: str | Path,
    *,
    width: int,
    height: int,
    data_offset: int | None = None,
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
    events = np.memmap(path, dtype=DT8, mode="r", offset=off)

    timestamps = events["t32"].astype(np.int64, copy=False)
    if timestamps.size == 0:
        msg = "No events"
        raise RuntimeError(msg)
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


def _parse_evt3_header(path: Path) -> tuple[int, int, int]:
    """Parse Metavision RAW EVT3 header, return (width, height, data_offset)."""
    width = -1
    height = -1

    with path.open("rb") as f:
        while True:
            pos = f.tell()
            ch = f.read(1)
            if not ch:
                msg = "Unexpected EOF while reading RAW header"
                raise RuntimeError(msg)
            if ch != b"%":
                # no more header lines
                f.seek(pos)
                break

            line = f.readline().decode("ascii", errors="ignore").rstrip("\n")
            # line now starts with a leading space, e.g.
            # "format EVT3; width=1280; height=720"

            if line.startswith(" format "):
                rest = line[len(" format ") :].strip()
                parts = [p.strip() for p in rest.split(";") if p.strip()]
                if not parts:
                    continue

                fmt_name = parts[0]
                if fmt_name not in {"EVT3", "EVT 3.0"}:
                    msg = f"Unsupported RAW format {fmt_name!r}, expected EVT3"
                    raise RuntimeError(msg)

                for opt in parts[1:]:
                    name, _, value = opt.partition("=")
                    name = name.strip()
                    value = value.strip()
                    if not name or not value:
                        continue
                    if name == "width":
                        width = int(value)
                    elif name == "height":
                        height = int(value)

            elif line.startswith(" geometry "):
                # e.g. " geometry 1280x720"
                geo = line[len(" geometry ") :].strip()
                if "x" in geo:
                    sw, sh = geo.split("x", 1)
                    width = int(sw)
                    height = int(sh)

            elif line.strip() == "end":
                # "% end"
                break

        offset = f.tell()

    # fall back to a sane default if header didn't give us geometry
    if width <= 0:
        width = 1280
    if height <= 0:
        height = 720

    return width, height, offset


def _pack_xy_pol_to_w32(
    x: np.ndarray,
    y: np.ndarray,
    pol: np.ndarray,
) -> np.ndarray:
    """Pack x, y, polarity into a 32-bit word for Recording.event_words.

    Layout (bit positions):
        bits  0..10 : x  (11 bits)
        bit      11 : polarity
        bits 12..22 : y  (11 bits)
        bits 23..31 : 0

    This is an internal evio packing and does NOT attempt to match Metavision's
    event_cd.h.
    """
    x_u = x.astype(np.uint32, copy=False)
    y_u = y.astype(np.uint32, copy=False)
    p_u = pol.astype(np.uint32, copy=False)
    return (y_u << 12) | (p_u << 11) | x_u


def open_raw_evt3(
    path: str | Path, *, max_events: int | None = None
) -> Recording:
    """Open a .raw EVT3 file and return a Recording.

    - Decodes EVT3 stream into CD events.
    - Timestamps are in microseconds (int64), sorted.
    - event_words are packed XY+polarity uint32 words (unsorted, use `order`).
    - If max_events is given, stops decoding after that many CD events
      (useful for debugging).
    """

    p = Path(path)
    width, height, data_offset = _parse_evt3_header(p)

    # All remaining data is 16-bit little-endian EVT3 words
    words = np.memmap(p, dtype="<u2", mode="r", offset=data_offset)
    if words.size == 0:
        msg = "No EVT3 data in RAW file"
        raise RuntimeError(msg)

    # ---- Time-base state (same logic as Metavision C++ sample) ----
    first_time_base_set = False
    current_time_base = np.int64(0)
    current_time_low = np.int64(0)
    current_time = np.int64(0)
    n_time_high_loop = 0

    MaxTimestampBase = ((np.int64(1) << 12) - 1) << 12  # 16773120 us
    TimeLoop = MaxTimestampBase + (1 << 12)  # 16777216 us
    LoopThreshold = 10 << 12

    # ---- Spatial state ----
    current_y = 0
    current_base_x = 0
    current_pol = 0

    xs: list[int] = []
    ys: list[int] = []
    ps: list[int] = []
    ts: list[int] = []

    idx = 0
    size = int(words.size)

    # Skip everything until first EVT_TIME_HIGH so time base is valid
    while idx < size:
        w = int(words[idx])
        etype = (w >> 12) & 0xF
        if etype == EVT_TIME_HIGH:
            data = w & 0x0FFF
            current_time_base = np.int64(data << 12)
            current_time = current_time_base
            first_time_base_set = True
            idx += 1
            break
        idx += 1

    if not first_time_base_set:
        msg = "No EVT_TIME_HIGH found in RAW stream; cannot establish time base"
        raise RuntimeError(msg)

    def limit_reached() -> bool:
        return max_events is not None and len(ts) >= max_events

    # Main decode loop
    while idx < size:
        w = int(words[idx])
        idx += 1

        etype = (w >> 12) & 0xF
        data = w & 0x0FFF

        if etype == EVT_ADDR_X:
            x = data & 0x7FF
            pol = (data >> 11) & 0x1

            xs.append(x)
            ys.append(current_y)
            ps.append(pol)
            ts.append(int(current_time))

            if limit_reached():
                break

        elif etype == VECT_12:
            valid = data
            end_x = current_base_x + 12
            x = current_base_x
            while x < end_x:
                if valid & 0x1:
                    xs.append(x)
                    ys.append(current_y)
                    ps.append(current_pol)
                    ts.append(int(current_time))
                    if limit_reached():
                        break
                valid >>= 1
                x += 1
            current_base_x = end_x
            if limit_reached():
                break

        elif etype == VECT_8:
            valid = data & 0xFF
            end_x = current_base_x + 8
            x = current_base_x
            while x < end_x:
                if valid & 0x1:
                    xs.append(x)
                    ys.append(current_y)
                    ps.append(current_pol)
                    ts.append(int(current_time))
                    if limit_reached():
                        break
                valid >>= 1
                x += 1
            current_base_x = end_x
            if limit_reached():
                break

        elif etype == EVT_ADDR_Y:
            current_y = data & 0x7FF

        elif etype == VECT_BASE_X:
            current_base_x = data & 0x7FF
            current_pol = (data >> 11) & 0x1

        elif etype == EVT_TIME_HIGH:
            time_field = data
            new_time_base = (
                np.int64(time_field << 12) + n_time_high_loop * TimeLoop
            )

            if (
                current_time_base > new_time_base
                and current_time_base - new_time_base
                >= MaxTimestampBase - LoopThreshold
            ):
                # time_high wrapped
                new_time_base += TimeLoop
                n_time_high_loop += 1

            current_time_base = new_time_base
            current_time = current_time_base

        elif etype == EVT_TIME_LOW:
            time_field = data
            current_time_low = np.int64(time_field)
            current_time = current_time_base + current_time_low

        else:
            # EXT_TRIGGER and any future types are ignored here
            continue

    if not ts:
        msg = "No CD events decoded from EVT3 RAW"
        raise RuntimeError(msg)

    timestamps = np.asarray(ts, dtype=np.int64)
    order = np.argsort(timestamps, kind="stable").astype(np.int32, copy=False)
    timestamps_sorted = timestamps[order]

    x_arr = np.asarray(xs, dtype=np.int32)
    y_arr = np.asarray(ys, dtype=np.int32)
    p_arr = np.asarray(ps, dtype=np.int8)
    w32 = _pack_xy_pol_to_w32(x_arr, y_arr, p_arr)

    return Recording(
        width=width,
        height=height,
        timestamps=timestamps_sorted,
        event_words=w32,
        order=order,
    )
