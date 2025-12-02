from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numba import njit
from numpy.typing import NDArray

from evio.codec.protocol import CodecInput, EventCodec
from evio.core.events import EventsBatch


def find_dat_data_offset(path: Path) -> int:
    """Scan a Prophesee/Metavision-style DAT header and return data offset.

    Matches the logic used in `open_dat_dt8`:

    - Header lines starting with '%' (ASCII)
    - Then 2 bytes: [event_type (1B), decoded_size (1B)]
    - We only accept decoded_size == 8 (8-byte CD events).
    """
    with path.open("rb") as file:
        while True:
            pos = file.tell()
            first = file.read(1)
            if not first or first != b"%":
                file.seek(pos)
                break
            # consume the header line
            file.readline()

        type_b = file.read(1)
        size_b = file.read(1)
        if len(type_b) != 1 or len(size_b) != 1:
            msg = "Invalid DAT header"
            raise RuntimeError(msg)

        decoded_size = int.from_bytes(size_b, "little")
        if decoded_size != 8:
            msg = f"Only 8B CD events supported (got {decoded_size}B)"
            raise RuntimeError(msg)

        return file.tell()


# ---------------------------------------------------------------------------
# Core DT8 unpacker
# ---------------------------------------------------------------------------


@njit
def encode_dt8_pairs(
    timestamps: NDArray[np.int64],
    x_coords: NDArray[np.int16],
    y_coords: NDArray[np.int16],
    polarities: NDArray[np.int8],
    width: int,
    height: int,
) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    """
    Encode (t, x, y, p) arrays into DT8 (t32, w32) words.

    This is the exact inverse of decode_dt8_pairs().
    Invalid events (outside width/height) are dropped.
    """
    n = timestamps.size

    t_out = np.empty(n, dtype=np.uint32)
    w_out = np.empty(n, dtype=np.uint32)

    out = 0
    for i in range(n):
        x = int(x_coords[i])
        y = int(y_coords[i])
        p = int(polarities[i])

        # Drop invalid coordinates to mirror decoder behavior
        if x < 0 or x >= width:
            continue
        if y < 0 or y >= height:
            continue

        # t32 = low 32 bits of timestamp
        t32 = np.uint32(timestamps[i] & 0xFFFFFFFF)

        # pack DT8 w32 word
        w32 = np.uint32((x & 0x3FFF) | ((y & 0x3FFF) << 14) | ((p & 0x1) << 28))

        t_out[out] = t32
        w_out[out] = w32
        out += 1

    return t_out[:out], w_out[:out]


@njit
def decode_dt8_pairs(
    timestamps: NDArray[np.uint32],
    words: NDArray[np.uint32],
    width: int,
    height: int,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int16],
    NDArray[np.int16],
    NDArray[np.int8],
]:
    """Unpack N DT8 events into (timestamps, x_coords, y_coords, polarities)."""
    n_timestamps = timestamps.size

    x_coords = np.empty(n_timestamps, dtype=np.int16)
    y_coords = np.empty(n_timestamps, dtype=np.int16)
    polarities = np.empty(n_timestamps, dtype=np.int8)

    out = 0
    for i in range(n_timestamps):
        word = words[i]

        x_coord = int(word & 0x3FFF)
        y_coord = int((word >> 14) & 0x3FFF)
        polarity = (word >> 28) & 0x1

        if 0 <= x_coord < width and 0 <= y_coord < height:
            x_coords[out] = x_coord
            y_coords[out] = y_coord
            polarities[out] = polarity
            out += 1
    return (
        timestamps.astype(np.int64),
        x_coords[:out],
        y_coords[:out],
        polarities[:out],
    )


# ---------------------------------------------------------------------------
# DT8 Codec — follows the same structure as evt3.py
# ---------------------------------------------------------------------------


@dataclass
class Dt8Codec(EventCodec):
    """Codec for Prophesee-style 8-byte CD events (DT8).

    Input:  One uint16 array of *raw words* (little-endian chunks).
            DT8 consists of two consecutive 16-bit values:
                - first  16 bits → lower half of t32 or w32
                - second 16 bits → upper half

            Format per event:
                [t32 (u4), w32 (u4)]

    State:
        Partial 16-bit fragments are buffered until full 32-bit numbers exist.
    """

    width: int
    height: int

    # Buffers as uint16 arrays holding incomplete 32-bit pairs
    _buf: NDArray[np.uint16] = field(
        default_factory=lambda: np.empty(0, np.uint16)
    )
    _partial: bool = False  # True → we are holding one unpaired word

    def reset(self) -> None:
        self._buf = np.empty(0, np.uint16)
        self._partial = False

    def decode_chunk(
        self,
        words: CodecInput,
        *,
        final: bool = False,
    ) -> Iterable[EventsBatch]:
        if words.size == 0:
            return ()

        # Prepend any leftover partial word
        if self._partial:
            words = np.concatenate([self._buf, words])
            self._partial = False

        # If odd number of uint16 → keep last uint16 for next chunk
        if words.size & 1:
            self._buf = words[-1:].copy()
            words = words[:-1]
            self._partial = True
        else:
            self._buf = np.empty(0, np.uint16)

        if words.size == 0:
            return ()

        # Reinterpret as uint32 little endian
        pairs = words.view(np.uint32)

        # Now each DT8 event is two uint32s: (t32, w32)
        if pairs.size & 1:
            # Should never happen, but guard
            return ()

        timestamps_encoded = pairs[0::2]
        words_encoded = pairs[1::2]

        timestamps, x_coords, y_coords, polarities = decode_dt8_pairs(
            timestamps_encoded, words_encoded, self.width, self.height
        )

        if timestamps.size == 0:
            return ()

        batch = EventsBatch(
            timestamps=timestamps,
            x_coords=x_coords,
            y_coords=y_coords,
            polarities=polarities,
        )
        return (batch,)
