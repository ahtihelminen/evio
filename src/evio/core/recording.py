from dataclasses import dataclass
from pathlib import Path

import numpy as np

# .raw
EVT_ADDR_Y = 0x0
EVT_ADDR_X = 0x2
VECT_BASE_X = 0x3
VECT_12 = 0x4
VECT_8 = 0x5
EVT_TIME_LOW = 0x6
EVT_TIME_HIGH = 0x8

# .dat
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

    Materialises every timestamp; for windowed replay see ``memmap_dat_dt8``
    plus ``evio.io.multi_dat_replay`` which slices on demand.
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


def _evt3_header_offset(path: Path) -> int:
    """Find the offset past the EVT3 ASCII header lines (% ...)."""
    with path.open("rb") as file:
        while True:
            position = file.tell()
            first_byte = file.read(1)
            if not first_byte or first_byte != b"%":
                file.seek(position)
                break
            line = file.readline()
            if line.strip() == b"% end":
                break
        return file.tell()


# EVT3 type codes (kept module-level so the numba decoder can see them)
_EVT3_EVT_ADDR_Y = 0x0
_EVT3_EVT_ADDR_X = 0x2
_EVT3_VECT_BASE_X = 0x3
_EVT3_VECT_12 = 0x4
_EVT3_VECT_8 = 0x5
_EVT3_EVT_TIME_LOW = 0x6
_EVT3_EVT_TIME_HIGH = 0x8

_EVT3_MAX_TIMESTAMP_BASE_US = ((1 << 12) - 1) << 12  # 16773120 us
_EVT3_TIME_LOOP_US = _EVT3_MAX_TIMESTAMP_BASE_US + (1 << 12)  # 16777216 us
_EVT3_LOOP_THRESHOLD_US = 10 << 12  # 40960 us


def _evt3_count_events(words: np.ndarray, width: int, height: int) -> int:
    """Pass 1: count CD events surviving width/height clipping. No alloc."""
    n_words = words.size
    current_y = 0
    current_base_x = 0
    n_events = 0
    # Skip past header words to first EVT_TIME_HIGH.
    index = 0
    while index < n_words:
        word = int(words[index])
        index += 1
        if (word >> 12) & 0xF == _EVT3_EVT_TIME_HIGH:
            break
    while index < n_words:
        word = int(words[index])
        index += 1
        event_type = (word >> 12) & 0xF
        payload = word & 0x0FFF
        if event_type == _EVT3_EVT_ADDR_Y:
            current_y = payload & 0x7FF
        elif event_type == _EVT3_EVT_ADDR_X:
            x = payload & 0x7FF
            if x < width and current_y < height:
                n_events += 1
        elif event_type == _EVT3_VECT_BASE_X:
            current_base_x = payload & 0x7FF
        elif event_type == _EVT3_VECT_12:
            validity_mask = payload
            x = current_base_x
            end_x = current_base_x + 12
            while x < end_x:
                if validity_mask & 0x1 and x < width and current_y < height:
                    n_events += 1
                validity_mask >>= 1
                x += 1
            current_base_x = end_x
        elif event_type == _EVT3_VECT_8:
            validity_mask = payload & 0xFF
            x = current_base_x
            end_x = current_base_x + 8
            while x < end_x:
                if validity_mask & 0x1 and x < width and current_y < height:
                    n_events += 1
                validity_mask >>= 1
                x += 1
            current_base_x = end_x
    return n_events


def _evt3_decode_into_buffers(
    words: np.ndarray,
    xs: np.ndarray, ys: np.ndarray,
    polarities: np.ndarray, timestamps_us: np.ndarray,
    *, width: int, height: int,
) -> int:
    """Pass 2: decode CD events into pre-allocated buffers. Returns count."""
    n_words = words.size
    current_y = 0
    current_base_x = 0
    current_polarity = 0
    current_time_base_us = 0
    current_time_low_us = 0
    current_time_us = 0
    time_high_loops = 0
    first_time_base_set = False
    n_events = 0
    index = 0
    while index < n_words and not first_time_base_set:
        word = int(words[index])
        index += 1
        if (word >> 12) & 0xF == _EVT3_EVT_TIME_HIGH:
            payload = word & 0x0FFF
            current_time_base_us = payload << 12
            current_time_us = current_time_base_us
            first_time_base_set = True
            break
    if not first_time_base_set:
        return 0
    while index < n_words:
        word = int(words[index])
        index += 1
        event_type = (word >> 12) & 0xF
        payload = word & 0x0FFF
        if event_type == _EVT3_EVT_ADDR_X:
            x = payload & 0x7FF
            polarity = (payload >> 11) & 0x1
            if x < width and current_y < height:
                xs[n_events] = x
                ys[n_events] = current_y
                polarities[n_events] = polarity
                timestamps_us[n_events] = current_time_us
                n_events += 1
        elif event_type == _EVT3_VECT_12:
            validity_mask = payload
            x = current_base_x
            end_x = current_base_x + 12
            while x < end_x:
                if validity_mask & 0x1 and x < width and current_y < height:
                    xs[n_events] = x
                    ys[n_events] = current_y
                    polarities[n_events] = current_polarity
                    timestamps_us[n_events] = current_time_us
                    n_events += 1
                validity_mask >>= 1
                x += 1
            current_base_x = end_x
        elif event_type == _EVT3_VECT_8:
            validity_mask = payload & 0xFF
            x = current_base_x
            end_x = current_base_x + 8
            while x < end_x:
                if validity_mask & 0x1 and x < width and current_y < height:
                    xs[n_events] = x
                    ys[n_events] = current_y
                    polarities[n_events] = current_polarity
                    timestamps_us[n_events] = current_time_us
                    n_events += 1
                validity_mask >>= 1
                x += 1
            current_base_x = end_x
        elif event_type == _EVT3_EVT_ADDR_Y:
            current_y = payload & 0x7FF
        elif event_type == _EVT3_VECT_BASE_X:
            current_base_x = payload & 0x7FF
            current_polarity = (payload >> 11) & 0x1
        elif event_type == _EVT3_EVT_TIME_HIGH:
            time_high_field = payload & 0x0FFF
            base_without_loops = time_high_field << 12
            new_time_base_us = (
                base_without_loops + time_high_loops * _EVT3_TIME_LOOP_US
            )
            if current_time_base_us > new_time_base_us and (
                current_time_base_us - new_time_base_us
            ) >= (_EVT3_MAX_TIMESTAMP_BASE_US - _EVT3_LOOP_THRESHOLD_US):
                time_high_loops += 1
                new_time_base_us = (
                    base_without_loops + time_high_loops * _EVT3_TIME_LOOP_US
                )
            current_time_base_us = new_time_base_us
            current_time_low_us = 0
            current_time_us = current_time_base_us
        elif event_type == _EVT3_EVT_TIME_LOW:
            time_low_field = payload & 0x0FFF
            current_time_low_us = time_low_field
            current_time_us = current_time_base_us + current_time_low_us
    return n_events


# Try to JIT-compile the two hot loops with numba. Without numba we
# fall back to pure Python (still ~20x slower than C, but ~5x less
# RAM than the legacy list-based decoder).
try:
    import numba
    _evt3_count_events = numba.njit(cache=True)(_evt3_count_events)
    _evt3_decode_into_buffers = numba.njit(cache=True)(
        _evt3_decode_into_buffers,
    )
except ImportError:
    pass


def open_raw_evt3(
    path: str | Path,
    *,
    width: int,
    height: int,
    data_offset: int | None = None,
) -> "Recording":
    """Load an EVT3 .raw file into a Recording (CD events only).

    - Assumes little-endian EVT3 (IMX636 / GenX320 default).
    - Decodes CD events: EVT_ADDR_X, VECT_BASE_X, VECT_12, VECT_8, EVT_ADDR_Y.
    - Ignores triggers and other event types.
    - Timestamps are reconstructed from EVT_TIME_HIGH/EVT_TIME_LOW in us.
      Handles time-high wraparound for recordings longer than ~16.7 s.

    Uses a two-pass numba-jitted decoder with pre-allocated buffers,
    so peak memory is ~5x the output array size instead of ~30x with
    the legacy Python-list implementation. A 2 GB events.raw decodes
    in ~3-5 GB peak RAM (down from ~30 GB), which lets a 24 GB Mac
    handle every FRED clip.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if data_offset is None:
        data_offset = _evt3_header_offset(path)
    else:
        data_offset = int(data_offset)

    words = np.memmap(path, dtype="<u2", mode="r", offset=data_offset)
    if words.size == 0:
        msg = "No EVT3 data found in RAW file"
        raise RuntimeError(msg)

    # Two-pass decode: count exactly, then allocate, then fill.
    n_events = int(
        _evt3_count_events(words, int(width), int(height)),
    )
    if n_events == 0:
        msg = "No CD events decoded from EVT3 RAW stream"
        raise RuntimeError(msg)

    xs = np.zeros(n_events, dtype=np.int16)
    ys = np.zeros(n_events, dtype=np.int16)
    polarities = np.zeros(n_events, dtype=np.int8)
    timestamps_us = np.zeros(n_events, dtype=np.int64)
    n_decoded = int(_evt3_decode_into_buffers(
        words, xs, ys, polarities, timestamps_us,
        width=int(width), height=int(height),
    ))
    if n_decoded != n_events:
        msg = (
            f"EVT3 decode mismatch: counted {n_events} but wrote "
            f"{n_decoded} (file may have changed)"
        )
        raise RuntimeError(msg)

    # EVT3 emits events time-monotonically, but we still compute a stable
    # order so the existing Recording contract is unchanged (downstream
    # callers may rely on `event_words[order]` returning sorted events).
    order = np.argsort(timestamps_us, kind="stable").astype(
        np.int32, copy=False,
    )
    sorted_timestamps = timestamps_us[order]

    # Pack x, y, polarity into a single 32-bit word (same layout as before):
    #   bits  0..10 : x
    #   bit      11 : polarity
    #   bits 12..22 : y
    event_words = (
        (ys.astype(np.uint32) << 12)
        | (polarities.astype(np.uint32) << 11)
        | xs.astype(np.uint32)
    )

    return Recording(
        width=width,
        height=height,
        timestamps=sorted_timestamps,
        event_words=event_words,
        order=order,
    )


# ---------------------------------------------------------------------------
# Streaming / cache: decode-once, memmap-forever
# ---------------------------------------------------------------------------


def cache_dir_for(event_path: str | Path) -> Path:
    """Cache directory path for a given events file (no I/O)."""
    p = Path(event_path)
    return p.parent / f"{p.name}.cache"


def cache_is_ready(event_path: str | Path) -> bool:
    """True iff a complete memmap cache exists for this events file."""
    cache = cache_dir_for(event_path)
    if not cache.is_dir():
        return False
    required = ("timestamps.npy", "xs.npy", "ys.npy", "polarities.npy")
    if not all((cache / name).exists() for name in required):
        return False
    return (cache / ".ready").exists()


def decode_to_cache(
    event_path: str | Path,
    *,
    width: int,
    height: int,
    overwrite: bool = False,
) -> Path:
    """One-time decode of an events.raw/.dat into memmappable arrays.

    Writes four files under ``<event_path>.cache/``:

      timestamps.npy   int64, ascending
      xs.npy           int16, in [0, width)
      ys.npy           int16, in [0, height)
      polarities.npy   int8,  {0, 1}

    Plus an empty ``.ready`` flag once all four files are committed.
    Idempotent: returns immediately if the cache is already ready (and
    ``overwrite=False``). On a re-decode after a failed run the partial
    cache is removed first.
    """
    event_path = Path(event_path)
    cache = cache_dir_for(event_path)
    if cache_is_ready(event_path) and not overwrite:
        return cache

    if cache.exists():
        for child in cache.iterdir():
            child.unlink()
    else:
        cache.mkdir(parents=True, exist_ok=True)

    if event_path.suffix == ".dat":
        rec = open_dat_dt8(event_path, width=width, height=height)
        sorted_words = rec.event_words[rec.order]
        xs = (sorted_words & 0x3FFF).astype(np.int16, copy=False)
        ys = ((sorted_words >> 14) & 0x3FFF).astype(np.int16, copy=False)
        polarities = ((sorted_words >> 28) & 0x1).astype(np.int8, copy=False)
    elif event_path.suffix == ".raw":
        rec = open_raw_evt3(event_path, width=width, height=height)
        sorted_words = rec.event_words[rec.order]
        xs = (sorted_words & 0x7FF).astype(np.int16, copy=False)
        ys = ((sorted_words >> 12) & 0x7FF).astype(np.int16, copy=False)
        polarities = ((sorted_words >> 11) & 0x1).astype(np.int8, copy=False)
    else:
        msg = f"unsupported event file: {event_path}"
        raise ValueError(msg)

    valid = (xs < width) & (ys < height)
    timestamps = rec.timestamps[valid]
    xs = xs[valid]
    ys = ys[valid]
    polarities = polarities[valid]

    np.save(cache / "timestamps.npy", np.ascontiguousarray(timestamps))
    np.save(cache / "xs.npy", np.ascontiguousarray(xs))
    np.save(cache / "ys.npy", np.ascontiguousarray(ys))
    np.save(cache / "polarities.npy", np.ascontiguousarray(polarities))
    (cache / ".ready").touch()
    return cache


def open_cached(
    event_path: str | Path,
    *,
    width: int,
    height: int,
    decode_if_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(timestamps, xs, ys, polarities)`` as memory-mapped arrays.

    If the cache is missing and ``decode_if_missing`` is True, decodes
    first (memory-heavy on big clips). Once the cache exists, this call
    is O(open-file) and uses essentially zero RAM up-front — pages are
    faulted in only when the caller actually indexes into them.

    This is the streaming-friendly path: callers can ``searchsorted``
    on ``timestamps`` and slice xs/ys/polarities with bounded resident
    memory regardless of clip size.
    """
    event_path = Path(event_path)
    cache = cache_dir_for(event_path)
    if not cache_is_ready(event_path):
        if not decode_if_missing:
            msg = (
                f"no cache at {cache}; rerun with decode_if_missing=True "
                "or call decode_to_cache(...) once."
            )
            raise FileNotFoundError(msg)
        decode_to_cache(event_path, width=width, height=height)

    timestamps = np.load(cache / "timestamps.npy", mmap_mode="r")
    xs = np.load(cache / "xs.npy", mmap_mode="r")
    ys = np.load(cache / "ys.npy", mmap_mode="r")
    polarities = np.load(cache / "polarities.npy", mmap_mode="r")
    return timestamps, xs, ys, polarities
