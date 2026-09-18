from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from types import TracebackType

import numpy as np

# in-house driver, built by `make build-externals` (see Makefile), not a
# pip dependency — absent until built, hence imported lazily by the facade.
import u3v  # type: ignore[import-not-found]

from evio.core.events import EventsBatch
from evio.core.generate_events import get_events_from_frames

# U3vUsb::PID: the real OPS mono camera. Other 0x1409-vendor PIDs (e.g.
# 0x8e00) are event cameras, and the PID is the only discriminator that
# holds: since SN-551 the driver opens no device it cannot select, so an
# event camera's serial comes back empty by construction -- and it was
# already empty in fact, because these cameras leave USB iSerialNumber
# unset and keep their real serial behind Treuzell (measured on ops001-dev
# 2026-09-18; `u3v.enumerate()` returned `('', 0x8e00)` six times).
MONO_CAMERA_PID = 0x8000


def get_mono_camera_serials() -> list[str]:
    """Returns attached U3V mono-camera serials in ascending order.

    `u3v.enumerate()` filters by VID 0x1409 only — it also returns an
    entry per attached 0x1409:0x8e00 event camera, so entries are
    filtered by PID (`MONO_CAMERA_PID`). Filtering on serial emptiness
    would happen to work on today's cameras and would silently stop
    working on any event camera that does publish a USB serial.
    """
    return sorted(
        serial for serial, pid in u3v.enumerate() if pid == MONO_CAMERA_PID
    )


# ---------------- Camera source ----------------


@dataclass
class MonoCameraSource:
    device_index: int = 0
    serial: str | None = None  # optional: open by serial if you want
    pixel_format: str = "Mono8"
    buffer_count: int | None = None  # unused: driver manages its own buffers
    fps: float | None = None
    exposure_us: float | None = None
    gain: float | None = None
    auto_exposure: bool = True
    auto_gain: bool = True

    camera: u3v.Camera = field(init=False)

    width: int = field(init=False, default=0)
    height: int = field(init=False, default=0)

    def __enter__(self) -> MonoCameraSource:
        self.camera = u3v.Camera.open(self.serial or "")
        self.width = self.camera.Width
        self.height = self.camera.Height

        if self.fps is not None:
            self.camera.AcquisitionFrameRate = float(self.fps)

        self.camera.ExposureAuto = (
            u3v.AutoMode.Continuous if self.auto_exposure else u3v.AutoMode.Off
        )
        if not self.auto_exposure and self.exposure_us is not None:
            self.camera.ExposureTime = float(self.exposure_us)

        self.camera.GainAuto = (
            u3v.AutoMode.Continuous if self.auto_gain else u3v.AutoMode.Off
        )
        if not self.auto_gain and self.gain is not None:
            self.camera.Gain = float(self.gain)

        self.camera.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self.camera.stop()
        self.camera.close()

    def read(
        self,
        timeout_ms: int = 1000,
        *,
        with_ts: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, int]:
        result = self.camera.read(timeout_ms)
        if result is None:
            msg = f"u3v: no frame within {timeout_ms}ms"
            raise TimeoutError(msg)
        frame, ts = result
        if with_ts:
            return frame, int(ts)
        return frame

    def iter_events(
        self,
        *,
        threshold: int = 8,
        timeout_ms: int = 1000,
    ) -> Iterator[EventsBatch]:
        prev, _ = self.read(timeout_ms=timeout_ms, with_ts=True)
        while True:
            curr, ts_us = self.read(timeout_ms=timeout_ms, with_ts=True)
            yield get_events_from_frames(
                prev, curr, timestamp_us=ts_us, threshold=threshold
            )
            prev = curr
