"""
Concrete NumPy-backed implementation of EventPacket.

Design goals:
- Immutable: the Packet object cannot be mutated after creation.
- Zero-copy: stores views into existing NumPy arrays where possible.
- Minimal validation by default (dtype/shape); can be relaxed via `validate=False`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .types import EventPacket, EventArray


def _readonly(arr: np.ndarray) -> np.ndarray:
    """Return a read-only view of `arr` without copying."""
    view = arr.view()
    view.flags.writeable = False  # type: ignore[attr-defined]
    return view


@dataclass(frozen=True, slots=True)
class _EventArrayView(EventArray):
    """Lightweight struct-of-arrays view implementing EventArray."""
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray


@dataclass(frozen=True, slots=True)
class Packet(EventPacket):
    """
    NumPy-backed packet of events.

    Parameters
    ----------
    x, y, t, p:
        1-D NumPy arrays (same length). Expected dtypes:
          x: uint16, y: uint16, t: int64 (microseconds), p: int8 (+1/-1).
    width, height:
        Sensor geometry in pixels.
    t0, t1:
        Packet time bounds in microseconds (inclusive). If omitted, they are derived
        from `t` as first/last element.
    validate:
        If True (default), enforce basic shape/dtype checks.
    """

    _x: np.ndarray
    _y: np.ndarray
    _t: np.ndarray
    _p: np.ndarray
    width: int
    height: int
    t0: int
    t1: int

    # ---- Constructors ---------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
        *,
        width: int,
        height: int,
        t0: int | None = None,
        t1: int | None = None,
        validate: bool = True,
        readonly: bool = True,
    ) -> "Packet":
        if validate:
            _validate_arrays(x, y, t, p)

        # Avoid copies; optionally mark read-only to discourage mutation.
        xv = _readonly(x) if readonly else x
        yv = _readonly(y) if readonly else y
        tv = _readonly(t) if readonly else t
        pv = _readonly(p) if readonly else p

        if t.size > 0:
            t0_eff = int(t[0]) if t0 is None else int(t0)
            t1_eff = int(t[-1]) if t1 is None else int(t1)
        else:
            # Empty packet: keep t0<=t1 invariant; both set to 0 by default.
            t0_eff = int(t0) if t0 is not None else 0
            t1_eff = int(t1) if t1 is not None else t0_eff

        return cls(xv, yv, tv, pv, int(width), int(height), t0_eff, t1_eff)

    # ---- EventPacket interface -----------------------------------------------

    @property
    def count(self) -> int:
        return int(self._t.size)

    def as_numpy(self) -> EventArray:
        # Return a tiny SoA view; no copies.
        return _EventArrayView(self._x, self._y, self._t, self._p)

    # ---- Convenience ----------------------------------------------------------

    def is_empty(self) -> bool:
        return self.count == 0

    def __repr__(self) -> str:  # helpful when printing in scripts/tests
        return (
            f"Packet(count={self.count}, "
            f"t0={self.t0}, t1={self.t1}, "
            f"width={self.width}, height={self.height})"
        )


# ---- Helpers -----------------------------------------------------------------

def _validate_arrays(x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray) -> None:
    if not (x.ndim == y.ndim == t.ndim == p.ndim == 1):
        raise ValueError("x, y, t, p must be 1-D arrays")

    n = x.size
    if not (y.size == t.size == p.size == n):
        raise ValueError("x, y, t, p must have the same length")

    # Dtype checks (kept soft: we only check kind/width; users can cast upstream if needed)
    if x.dtype != np.uint16 or y.dtype != np.uint16:
        raise TypeError("x and y must have dtype uint16")
    if t.dtype != np.int64:
        raise TypeError("t must have dtype int64 (microseconds)")
    if p.dtype != np.int8:
        raise TypeError("p must have dtype int8 with values in {+1, -1}")

    # Optional monotonicity sanity check (non-decreasing)
    if n > 1 and np.any(t[1:] < t[:-1]):
        raise ValueError("t must be monotonically non-decreasing")


__all__ = ["Packet"]
