"""
Minimal public typing contracts for evio.

These Protocols define the *shape* of event data handled by evio:
- EventArray: struct-of-arrays (x, y, t, p)
- EventPacket: immutable packet wrapper with metadata and access to the SoA.

The @runtime_checkable decorator allows isinstance(obj, EventPacket)
or isinstance(obj, EventArray) to succeed based on structure rather than inheritance.
"""

from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class EventArray(Protocol):
    """
    A lightweight, read-only struct-of-arrays view for events.

    Expected attributes:
      x: (N,) np.uint16
      y: (N,) np.uint16
      t: (N,) np.int64    # timestamps in microseconds
      p: (N,) np.int8     # polarity values in {+1, -1}
    """

    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray


@runtime_checkable
class EventPacket(Protocol):
    """
    Immutable packet of events with geometry and time bounds.

    Implementations should:
      - avoid copying data;
      - provide geometry (width, height);
      - report packet time bounds (t0, t1);
      - expose event count via `count`;
      - return the struct-of-arrays view through `as_numpy()`.
    """

    width: int
    height: int
    t0: int
    t1: int

    @property
    def count(self) -> int:
        ...

    def as_numpy(self) -> EventArray:
        """Return a struct-of-arrays view of the packet's events."""
        ...
