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

import time
from collections import deque
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

import imx636  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from evio.core.events import EventsBatch
from evio.core.window import RollingWindowBatchProcessor


def _empty_batch() -> EventsBatch:
    return EventsBatch(
        timestamps=np.zeros(0, dtype=np.int64),
        x_coords=np.zeros(0, dtype=np.int16),
        y_coords=np.zeros(0, dtype=np.int16),
        polarities=np.zeros(0, dtype=np.int8),
    )


def get_camera_serials() -> list[str]:
    """Return camera IDs for each connected IMX636 camera.

    If the FX3 EEPROMs are programmed with real serial numbers, returns
    those (sorted). If serials are blank — common on the IDS UE-39B0XCP
    + custom-built carriers we use — returns positional IDs like
    "event-0" through "event-N-1". The order matches successive calls to
    `imx636.IMX636.open()`, which is stable for a given physical USB
    topology (libusb enumerates by bus+port).
    """
    raw = list(imx636.IMX636.list_cameras())
    if all(s == "" for s in raw):
        return [f"event-{i}" for i in range(len(raw))]
    return sorted(raw)


@dataclass()
class Imx636CameraSource:
    serial: str = ""
    window_length_us: int = 10_000  # 10 ms rolling window
    step_us: int | None = None
    # Mains-lighting flicker (50/60 Hz + ballast harmonics) shows up on
    # event cameras as a strong synchronized event burst across the whole
    # sensor — at the panorama level it looks like every tile pulsing
    # bright/dark on each light cycle. Default ON: band-stop 50-500 Hz
    # covers EU and US mains plus their first few harmonics. Override
    # with anti_flicker=False or a custom AntiFlickerConfig from imx636.
    anti_flicker: bool = True
    # Optional Event Rate Controller — caps events/sec at the sensor
    # before they leave the chip. Off by default (0); set to a positive
    # value (e.g. 1_000_000 = 1 M ev/sec/cam) for high-clutter scenes
    # where downstream consumers can't keep up.
    erc_event_rate: int = 0

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
        # Synthetic positional IDs ("event-N") aren't real USB serials; pass
        # "" so the driver opens whichever camera is next-available. The
        # caller relies on the open order matching get_camera_serials().
        real_serial = (
            "" if self.serial.startswith("event-") else self.serial
        )
        self.cam = imx636.IMX636.open(real_serial)
        self.width = int(self.cam.width())
        self.height = int(self.cam.height())
        self.cam.set_cd_callback(self._on_cd_events)
        if self.anti_flicker:
            af = imx636.AntiFlickerConfig()
            af.enabled = True
            af.band_stop = True
            af.low_freq = 50
            af.high_freq = 500
            af.duty_cycle = 50
            self.cam.set_antiflicker(af)
        if self.erc_event_rate > 0:
            erc = imx636.ERCConfig()
            erc.enabled = True
            erc.event_rate = int(self.erc_event_rate)
            self.cam.set_erc(erc)
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

        # Match EventCameraSource pacing: even if no events arrive in a
        # window, return an empty batch so downstream renderers tick at
        # the configured cadence (otherwise a static scene blocks
        # next() indefinitely and panorama frames stop updating).
        deadline_ns = (
            time.monotonic_ns() + int(self.window_length_us) * 1000
        )

        while True:
            if self.window_queue:
                return self.window_queue.popleft()

            # Pump poll() until the CD callback delivers at least one chunk
            # OR the window deadline elapses.
            while not self.decoded_queue:
                r = self.cam.poll()
                if r < 0:
                    msg = "imx636: USB error from poll()"
                    raise RuntimeError(msg)
                if time.monotonic_ns() >= deadline_ns:
                    return _empty_batch()

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
