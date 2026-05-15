"""IMX636 event-camera source using the HAL-free libusb driver.

Mirrors `EventCameraSource` from `evio.io.event_camera`:
- Context manager opens/closes the device.
- `__next__()` returns an `EventsBatch` over the configured rolling window.

Backend: the in-house `imx636` Python module (built from
github.com/sensofusion/IMX636 → `~/Documents/IMX636`), which talks directly
to the Cypress FX3 USB bridge via libusb-1.0. No Prophesee / Metavision /
OpenEB dependency at runtime.

The IMX636 driver is push-style (registers a CD callback, host calls
`poll()` to pump USB transfers); we adapt to evio's pull-style iterator
by accumulating callback deliveries into a deque and draining it in
`__next__()`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

import imx636  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from evio.core.events import EventsBatch
from evio.core.window import RollingWindowBatchProcessor


def get_camera_serials() -> list[str]:
    """Return attached IMX636 camera serial numbers in ascending order."""
    return sorted(imx636.IMX636.list_cameras())


@dataclass()
class Imx636CameraSource:
    serial: str = ""
    window_length_us: int = 10_000  # 10 ms rolling window
    step_us: int | None = None

    cam: Any = field(init=False, default=None)
    width: int = field(init=False, default=0)
    height: int = field(init=False, default=0)
    decoded_queue: deque[NDArray[np.void]] = field(
        init=False, default_factory=deque,
    )
    window_queue: deque[EventsBatch] = field(init=False, default_factory=deque)
    _processor: RollingWindowBatchProcessor | None = field(
        init=False, default=None,
    )
    _processor_window_us: int = field(init=False, default=0)
    _processor_step_us: int = field(init=False, default=0)

    def __enter__(self) -> Imx636CameraSource:
        self.cam = imx636.IMX636.open(self.serial or "")
        self.width = int(self.cam.width())
        self.height = int(self.cam.height())
        self.cam.set_cd_callback(self._on_cd_events)
        self.cam.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        if self.cam is not None:
            self.cam.stop()

    def __iter__(self) -> Imx636CameraSource:
        return self

    def _on_cd_events(self, arr: NDArray[np.void]) -> None:
        # Callback fires from poll(); arr is dtype imx636.CD_DTYPE
        # with fields (x int16, y int16, p int8, _pad u1x3, t int64).
        if arr.size:
            self.decoded_queue.append(arr.copy())

    def _ensure_processor(self) -> None:
        window_us = max(1, int(self.window_length_us))
        step_us = max(1, int(self.step_us or window_us))
        if (
            self._processor is None
            or window_us != self._processor_window_us
            or step_us != self._processor_step_us
        ):
            self._processor = RollingWindowBatchProcessor(window_us, step_us)
            self._processor_window_us = window_us
            self._processor_step_us = step_us
            self.window_queue.clear()

    def __next__(self) -> EventsBatch:
        if self.cam is None:
            msg = "Imx636CameraSource not started; use as a context manager."
            raise RuntimeError(msg)
        self._ensure_processor()

        while True:
            if self.window_queue:
                return self.window_queue.popleft()

            # Pump poll() until the CD callback delivers at least one chunk.
            while not self.decoded_queue:
                r = self.cam.poll()
                if r < 0:
                    msg = "imx636: USB error from poll()"
                    raise RuntimeError(msg)

            buffer = self.decoded_queue.popleft()
            if buffer.size == 0:
                continue

            batch = EventsBatch(
                x_coords=buffer["x"].astype(np.int16, copy=False),
                y_coords=buffer["y"].astype(np.int16, copy=False),
                timestamps=buffer["t"].astype(np.int64, copy=False),
                polarities=buffer["p"].astype(np.int8, copy=False),
            )

            if self._processor is None:
                continue
            for window in self._processor.process((batch,)):
                self.window_queue.append(window)
