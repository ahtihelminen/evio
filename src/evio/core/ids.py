"""Identifier generators.

`new_ulid` is byte-for-byte the SAPIENT library's generator, kept here so the
detection chain needs no SAPIENT dependency: a track's id becomes the
`object_id` of every DetectionReport sent for it, so it must stay a ULID.
"""

from __future__ import annotations

import os
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_LEN = 26


def new_ulid() -> str:
    """A ULID: 48-bit ms timestamp + 80 random bits, Crockford base32."""
    n = (int(time.time() * 1000) << 80) | int.from_bytes(os.urandom(10), "big")
    out = bytearray(_ULID_LEN)
    for i in range(_ULID_LEN - 1, -1, -1):
        out[i] = ord(_CROCKFORD[n & 0x1F])
        n >>= 5
    return out.decode("ascii")
