# tests/test_types_packet.py
import numpy as np
import pytest

from evio.core.packet import Packet
from evio.core.types import EventPacket, EventArray


def make_arrays(n: int = 8, *, width: int = 320, height: int = 240):
    # Simple synthetic data: monotonic timestamps in microseconds
    x = np.arange(n, dtype=np.uint16) % width
    y = (np.arange(n, dtype=np.uint16) * 2) % height
    t = (np.arange(n, dtype=np.int64) * 100).astype(np.int64)
    p = (np.ones(n, dtype=np.int8))
    p[::2] = -1
    return x, y, t, p


def test_packet_basic_and_protocol_conformance():
    x, y, t, p = make_arrays(10)
    pkt = Packet.from_arrays(x, y, t, p, width=320, height=240)

    # Runtime protocol checks
    assert isinstance(pkt, EventPacket)

    a = pkt.as_numpy()
    assert isinstance(a, EventArray)

    # Shapes and dtypes
    assert a.x.shape == (10,)
    assert a.y.shape == (10,)
    assert a.t.shape == (10,)
    assert a.p.shape == (10,)
    assert a.x.dtype == np.uint16
    assert a.y.dtype == np.uint16
    assert a.t.dtype == np.int64
    assert a.p.dtype == np.int8

    # Metadata and count
    assert pkt.width == 320
    assert pkt.height == 240
    assert pkt.count == 10
    assert pkt.t0 == int(t[0])
    assert pkt.t1 == int(t[-1])
    assert pkt.t0 <= pkt.t1


def test_zero_copy_and_readonly_views():
    x, y, t, p = make_arrays(6)
    pkt = Packet.from_arrays(x, y, t, p, width=64, height=64, readonly=True)
    a = pkt.as_numpy()

    # The returned arrays should be views over the originals (no copies)
    # Heuristic: either same base object or share memory.
    for arr, src in [(a.x, x), (a.y, y), (a.t, t), (a.p, p)]:
        # No new ownership
        assert arr.flags["OWNDATA"] is False
        # Share memory with the source
        assert np.shares_memory(arr, src)
        # And not writeable (discourage mutation through the packet)
        assert arr.flags["WRITEABLE"] is False


def test_empty_packet_behavior():
    x = np.array([], dtype=np.uint16)
    y = np.array([], dtype=np.uint16)
    t = np.array([], dtype=np.int64)
    p = np.array([], dtype=np.int8)

    pkt = Packet.from_arrays(x, y, t, p, width=1, height=1)
    assert pkt.count == 0
    # For empty packets, t0 and t1 collapse to the same value (default 0)
    assert pkt.t0 == pkt.t1


def test_invalid_dtypes_raise():
    x, y, t, p = make_arrays(4)
    with pytest.raises(TypeError):
        Packet.from_arrays(x.astype(np.int32), y, t, p, width=1, height=1)
    with pytest.raises(TypeError):
        Packet.from_arrays(x, y.astype(np.int32), t, p, width=1, height=1)
    with pytest.raises(TypeError):
        Packet.from_arrays(x, y, t.astype(np.int32), p, width=1, height=1)
    with pytest.raises(TypeError):
        Packet.from_arrays(x, y, t, p.astype(np.int16), width=1, height=1)


def test_non_monotonic_timestamps_raise():
    x, y, t, p = make_arrays(5)
    t = t.copy()
    t[3] = t[2] - 1  # break monotonicity
    with pytest.raises(ValueError):
        Packet.from_arrays(x, y, t, p, width=10, height=10)


def test_custom_t0_t1_override_and_property_count():
    x, y, t, p = make_arrays(3)
    pkt = Packet.from_arrays(x, y, t, p, width=10, height=10, t0=123, t1=456)
    assert pkt.t0 == 123
    assert pkt.t1 == 456
    # count is a property computed from t, not a stored field
    assert pkt.count == t.size


def test_packet_is_immutable_and_repr_is_useful():
    x, y, t, p = make_arrays(2)
    pkt = Packet.from_arrays(x, y, t, p, width=10, height=10)
    # Immutability: frozen dataclass should refuse attribute assignment
    with pytest.raises(AttributeError):
        pkt.width = 20  # type: ignore[attr-defined]

    rep = repr(pkt)
    assert "Packet(" in rep and "count=" in rep and "t0=" in rep and "t1=" in rep
