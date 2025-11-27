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


def open_dat_dt8(
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
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # --- Find start of binary data (skip ASCII header lines starting with '%')
    if data_offset is None:
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
            data_offset = file.tell()
    else:
        data_offset = int(data_offset)

    # --- Read EVT3 words as 16-bit little-endian unsigned integers ---
    words = np.memmap(path, dtype="<u2", mode="r", offset=data_offset)
    if words.size == 0:
        msg = "No EVT3 data found in RAW file"
        raise RuntimeError(msg)

    words = np.asarray(words, dtype=np.uint16)
    n_words = int(words.size)

    # --- EVT3 event type codes (high nibble of the 16-bit word) ---
    EVT_ADDR_Y = 0x0
    EVT_ADDR_X = 0x2
    VECT_BASE_X = 0x3
    VECT_12 = 0x4
    VECT_8 = 0x5
    EVT_TIME_LOW = 0x6
    EVT_TIME_HIGH = 0x8
    EXT_TRIGGER = 0xA  # ignored

    # --- Time-base wrap constants (same as Metavision C++ example) ---
    max_timestamp_base_us = ((1 << 12) - 1) << 12  # 16773120 us
    time_loop_us = max_timestamp_base_us + (1 << 12)  # 16777216 us
    loop_threshold_us = 10 << 12  # 40960 us

    # --- Decoder state ---
    current_y = 0
    current_base_x = 0
    current_polarity = 0

    current_time_base_us = 0  # high part (aligned to 12 bits)
    current_time_low_us = 0
    current_time_us = 0
    time_high_loops = 0
    first_time_base_set = False

    # Output buffers
    x_coords: list[int] = []
    y_coords: list[int] = []
    polarities: list[int] = []
    timestamps_us: list[int] = []

    # --- Phase 1: skip until first EVT_TIME_HIGH to get a valid time base ---
    index = 0
    while index < n_words and not first_time_base_set:
        word = int(words[index])
        index += 1

        event_type = word >> 12
        payload = word & 0x0FFF

        if event_type == EVT_TIME_HIGH:
            time_high_field = payload & 0x0FFF
            current_time_base_us = time_high_field << 12
            current_time_low_us = 0
            current_time_us = current_time_base_us
            first_time_base_set = True
            break

    if not first_time_base_set:
        msg = "No EVT_TIME_HIGH found; cannot establish time base"
        raise RuntimeError(msg)

    # --- Phase 2: main decode loop from current index ---
    while index < n_words:
        word = int(words[index])
        index += 1

        event_type = word >> 12
        payload = word & 0x0FFF

        if event_type == EVT_ADDR_X:
            # Single CD event
            x = payload & 0x7FF
            polarity = (payload >> 11) & 0x1

            x_coords.append(x)
            y_coords.append(current_y)
            polarities.append(polarity)
            timestamps_us.append(current_time_us)

        elif event_type == VECT_12:
            # Vector of up to 12 CD events
            validity_mask = payload
            x = current_base_x
            end_x = current_base_x + 12

            while x < end_x:
                if validity_mask & 0x1:
                    x_coords.append(x)
                    y_coords.append(current_y)
                    polarities.append(current_polarity)
                    timestamps_us.append(current_time_us)
                validity_mask >>= 1
                x += 1

            current_base_x = end_x

        elif event_type == VECT_8:
            # Vector of up to 8 CD events (lower 8 bits used)
            validity_mask = payload & 0xFF
            x = current_base_x
            end_x = current_base_x + 8

            while x < end_x:
                if validity_mask & 0x1:
                    x_coords.append(x)
                    y_coords.append(current_y)
                    polarities.append(current_polarity)
                    timestamps_us.append(current_time_us)
                validity_mask >>= 1
                x += 1

            current_base_x = end_x

        elif event_type == EVT_ADDR_Y:
            # Update current y coordinate (bit 11 is system type, ignored here)
            current_y = payload & 0x7FF

        elif event_type == VECT_BASE_X:
            # Base x and polarity for following VECT_12 / VECT_8 events
            current_base_x = payload & 0x7FF
            current_polarity = (payload >> 11) & 0x1

        elif event_type == EVT_TIME_HIGH:
            # High 12 bits of timestamp; detect wrap of the high segment
            time_high_field = payload & 0x0FFF
            base_without_loops = time_high_field << 12

            new_time_base_us = (
                base_without_loops + time_high_loops * time_loop_us
            )

            # If new base appears to go "backwards" enough, assume a loop
            if current_time_base_us > new_time_base_us and (
                current_time_base_us - new_time_base_us
            ) >= (max_timestamp_base_us - loop_threshold_us):
                time_high_loops += 1
                new_time_base_us = (
                    base_without_loops + time_high_loops * time_loop_us
                )

            current_time_base_us = new_time_base_us
            current_time_low_us = 0
            current_time_us = current_time_base_us

        elif event_type == EVT_TIME_LOW:
            # Low 12 bits of timestamp, attached to current high base
            time_low_field = payload & 0x0FFF
            current_time_low_us = time_low_field
            current_time_us = current_time_base_us + current_time_low_us

        elif event_type == EXT_TRIGGER:
            # Ignore triggers in this loader
            continue

        else:
            # Ignore any unknown/unused event types
            continue

    if not timestamps_us:
        msg = "No CD events decoded from EVT3 RAW stream"
        raise RuntimeError(msg)

    # --- Build Recording fields: timestamps (sorted), event_words, order ---

    timestamps_array = np.asarray(timestamps_us, dtype=np.int64)
    order = np.argsort(timestamps_array, kind="stable").astype(
        np.int32, copy=False
    )
    sorted_timestamps = timestamps_array[order]

    x_array = np.asarray(x_coords, dtype=np.uint32)
    y_array = np.asarray(y_coords, dtype=np.uint32)
    polarity_array = np.asarray(polarities, dtype=np.uint32)

    # Pack x, y, polarity into a single 32-bit word:
    #   bits  0..10 : x
    #   bit      11 : polarity
    #   bits 12..22 : y
    event_words = (y_array << 12) | (polarity_array << 11) | x_array

    return Recording(
        width=width,
        height=height,
        timestamps=sorted_timestamps,
        event_words=event_words,
        order=order,
    )
