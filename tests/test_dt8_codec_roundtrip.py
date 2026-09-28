"""The DT8 codec and its DAT header offset finder, both previously untested.

plans/test_suite.md section 9.2 item 3 names this as the precondition for
T4 (replay e2e), T5 (benchmark) and SN-452 (frozen dataset): nothing pinned
that the recording format preserves an event before this file existed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evio.codec.dt8 import Dt8Codec, find_dat_data_offset
from evio.core.events import EventsBatch

pytestmark = pytest.mark.t1

_WIDTH = 1280
_HEIGHT = 720


def _events(n: int, *, seed: int = 7) -> EventsBatch:
    rng = np.random.default_rng(seed)
    return EventsBatch(
        timestamps=np.sort(
            rng.integers(0, 2**32, size=n, dtype=np.int64)
        ),
        x_coords=rng.integers(0, _WIDTH, size=n, dtype=np.int16),
        y_coords=rng.integers(0, _HEIGHT, size=n, dtype=np.int16),
        polarities=rng.integers(0, 2, size=n, dtype=np.int8),
    )


def _decode_all(words: np.ndarray) -> EventsBatch:
    codec = Dt8Codec(width=_WIDTH, height=_HEIGHT)
    (batch,) = codec.decode_chunk(words, final=True)
    return batch


def test_decode_of_encode_is_the_identity() -> None:
    original = _events(500)
    words = Dt8Codec(width=_WIDTH, height=_HEIGHT).encode_chunk(original)
    decoded = _decode_all(words)

    np.testing.assert_array_equal(decoded.timestamps, original.timestamps)
    np.testing.assert_array_equal(decoded.x_coords, original.x_coords)
    np.testing.assert_array_equal(decoded.y_coords, original.y_coords)
    np.testing.assert_array_equal(decoded.polarities, original.polarities)


def test_encode_drops_out_of_bounds_coordinates() -> None:
    original = EventsBatch(
        timestamps=np.array([1, 2, 3], dtype=np.int64),
        x_coords=np.array([-1, 0, _WIDTH], dtype=np.int16),
        y_coords=np.array([0, 0, 0], dtype=np.int16),
        polarities=np.array([0, 1, 0], dtype=np.int8),
    )
    words = Dt8Codec(width=_WIDTH, height=_HEIGHT).encode_chunk(original)
    decoded = _decode_all(words)

    np.testing.assert_array_equal(decoded.timestamps, [2])
    np.testing.assert_array_equal(decoded.x_coords, [0])
    np.testing.assert_array_equal(decoded.y_coords, [0])
    np.testing.assert_array_equal(decoded.polarities, [1])


def test_decode_survives_an_odd_word_split_across_chunks() -> None:
    """Regression pin for the partial-pair buffer (dt8.py:186-196).

    Each event is 4 uint16 words; splitting the stream at an odd offset
    forces `_buf`/`_partial` to hold a lone word across `decode_chunk`
    calls, which is exactly the shape a chunked file reader produces.
    """
    original = _events(50)
    words = Dt8Codec(width=_WIDTH, height=_HEIGHT).encode_chunk(original)
    split = 9  # odd, mid-event

    codec = Dt8Codec(width=_WIDTH, height=_HEIGHT)
    first_batches = list(codec.decode_chunk(words[:split]))
    second_batches = list(codec.decode_chunk(words[split:], final=True))
    decoded = EventsBatch(
        timestamps=np.concatenate(
            [b.timestamps for b in (*first_batches, *second_batches)]
        ),
        x_coords=np.concatenate(
            [b.x_coords for b in (*first_batches, *second_batches)]
        ),
        y_coords=np.concatenate(
            [b.y_coords for b in (*first_batches, *second_batches)]
        ),
        polarities=np.concatenate(
            [b.polarities for b in (*first_batches, *second_batches)]
        ),
    )

    np.testing.assert_array_equal(decoded.timestamps, original.timestamps)
    np.testing.assert_array_equal(decoded.x_coords, original.x_coords)
    np.testing.assert_array_equal(decoded.y_coords, original.y_coords)
    np.testing.assert_array_equal(decoded.polarities, original.polarities)


def _header(*, event_type: int = 0, decoded_size: int = 8) -> bytes:
    return (
        b"% evio footage recorder\n"
        + bytes([event_type, decoded_size])
    )


def test_find_dat_data_offset_skips_header_lines(tmp_path: Path) -> None:
    header = b"% line one\n% line two\n" + bytes([0, 8])
    payload = b"\x01\x02\x03\x04"
    path = tmp_path / "events.dat"
    path.write_bytes(header + payload)

    assert find_dat_data_offset(path) == len(header)


def test_find_dat_data_offset_rejects_non_8_byte_events(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.dat"
    path.write_bytes(_header(decoded_size=4) + b"\x00\x00\x00\x00")

    with pytest.raises(RuntimeError, match="8B CD events"):
        find_dat_data_offset(path)


def test_find_dat_data_offset_rejects_truncated_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.dat"
    header = b"% evio footage recorder\n" + bytes([0])  # size byte missing
    path.write_bytes(header)

    with pytest.raises(RuntimeError, match="Invalid DAT header"):
        find_dat_data_offset(path)
