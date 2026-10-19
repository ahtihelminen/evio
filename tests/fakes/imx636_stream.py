"""An IMX636 stand-in that streams scripted buffers through the CD callback.

`FakeStreamingCamera` models what `Imx636CameraSource.__next__` relies on of
the driver:
- `set_cd_callback(callback)` registers `callback(buffer)`; each `poll()`
  fires it once with the next scripted buffer and returns 1, or returns 0
  when none is ready (the driver's `< 0` USB error is not modelled);
- a buffer is `CD_DTYPE`, whose `t` is the camera clock in microseconds. The
  clock is the script's to move, so it can restart near zero mid-stream.

Behaviours assumed of the real driver, not proved here: that its buffers
hold events in stream order across a clock restart, and that `CD_DTYPE`
matches `imx636.CD_DTYPE` (the hardware tier compares the two).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable

import numpy as np

CD_DTYPE = np.dtype([
    ("x", "<i2"), ("y", "<i2"), ("p", "i1"), ("_pad", "u1", (3,)), ("t", "<i8"),
])


def cd_buffer(timestamps: np.ndarray) -> np.ndarray:
    """A `CD_DTYPE` buffer holding events at `timestamps`, all at pixel 0,0."""
    buffer = np.zeros(timestamps.size, CD_DTYPE)
    buffer["t"] = timestamps
    return buffer


class FakeStreamingCamera:
    def __init__(self, buffers: Iterable[np.ndarray]) -> None:
        self._buffers = deque(buffers)
        self._callback: Callable[[np.ndarray], None] | None = None
        self.started = False

    def set_cd_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        self._callback = callback

    def start(self) -> None:
        self.started = True

    def poll(self) -> int:
        if not self._buffers or self._callback is None:
            return 0
        self._callback(self._buffers.popleft())
        return 1
