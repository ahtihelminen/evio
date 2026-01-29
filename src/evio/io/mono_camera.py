from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from types import TracebackType

import numpy as np
from ids_peak import ids_peak, ids_peak_ipl_extension  # type: ignore
from ids_peak_ipl import ids_peak_ipl  # type: ignore

from evio.core.events import EventsBatch


def _frames_to_events(
    prev: np.ndarray,
    curr: np.ndarray,
    *,
    ts_us: int,
    threshold: int,
) -> EventsBatch:
    # signed difference
    diff = curr.astype(np.int16) - prev.astype(np.int16)

    mask = np.abs(diff) >= threshold
    ys, xs = np.nonzero(mask)

    if xs.size == 0:
        return EventsBatch(
            x_coords=np.empty(0, dtype=np.int16),
            y_coords=np.empty(0, dtype=np.int16),
            timestamps=np.empty(0, dtype=np.int64),
            polarities=np.empty(0, dtype=np.int8),
        )

    d = diff[ys, xs]
    # polarity convention: 1 for increase, 0 for decrease
    ps = (d > 0).astype(np.int8)

    ts = np.full(xs.shape[0], ts_us, dtype=np.int64)

    return EventsBatch(
        x_coords=xs.astype(np.int16, copy=False),
        y_coords=ys.astype(np.int16, copy=False),
        timestamps=ts,
        polarities=ps,
    )


@dataclass
class MonoCameraSource:
    device_index: int = 0
    serial: str | None = None  # optional: open by serial if you want
    pixel_format: ids_peak_ipl.PixelFormatName = (
        ids_peak_ipl.PixelFormatName_Mono8
    )
    buffer_count: int | None = (
        None  # if None -> use NumBuffersAnnouncedMinRequired()
    )
    fps: float | None = None
    exposure_us: float | None = None
    gain: float | None = None
    auto_exposure: bool = True
    auto_gain: bool = True

    # runtime handles
    device: ids_peak.Device = field(init=False)
    remote: ids_peak.NodeMap = field(init=False)
    datastream: ids_peak.DataStream = field(init=False)

    width: int = field(init=False, default=0)
    height: int = field(init=False, default=0)
    _acquiring: bool = field(init=False, default=False)

    def __enter__(self) -> MonoCameraSource:
        ids_peak.Library.Initialize()
        device_manager = ids_peak.DeviceManager.Instance()
        device_manager.Update()

        devices = device_manager.Devices()
        dev_info = self._select_device(devices)
        self.device = dev_info.OpenDevice(ids_peak.DeviceAccessType_Control)
        self.remote = self.device.RemoteDevice().NodeMaps()[0]
        self.datastream = self.device.DataStreams()[0].OpenDataStream()

        width_node = self.remote.FindNode("Width")
        self.width = width_node.Value()

        height_node = self.remote.FindNode("Height")
        self.height = height_node.Value()

        if self.fps is not None:
            self.remote.FindNode("AcquisitionFrameRate").SetValue(
                float(self.fps)
            )

        exposure_auto = self.remote.FindNode("ExposureAuto")
        if self.auto_exposure:
            exposure_auto.SetCurrentEntry(exposure_auto.FindEntry("Continuous"))
        else:
            exposure_auto.SetCurrentEntry(exposure_auto.FindEntry("Off"))
            if self.exposure_us is not None:
                self.remote.FindNode("ExposureTime").SetValue(
                    float(self.exposure_us)
                )

        gain_auto = self.remote.FindNode("GainAuto")
        if self.auto_gain:
            gain_auto.SetCurrentEntry(gain_auto.FindEntry("Continuous"))
        else:
            gain_auto.SetCurrentEntry(gain_auto.FindEntry("Off"))
            if self.gain is not None:
                self.remote.FindNode("Gain").SetValue(float(self.gain))

        print("ExposureTime:", self.remote.FindNode("ExposureTime").Value())
        print(
            "AcquisitionFrameRate:",
            self.remote.FindNode("AcquisitionFrameRate").Value(),
        )

        self._allocate_buffers()
        self._start_acquisition()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self._shutdown()

    def read(
        self,
        timeout_ms: int = 1000,
        *,
        with_ts: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, int]:
        """Fetch one monochrome frame.

        Returns:
          - frame: np.ndarray[uint8] shape (H, W)
          - if with_ts=True: (frame, timestamp_us) where timestamp comes from
            IDS IPL if available, else 0.
        """
        buf = self.datastream.WaitForFinishedBuffer(int(timeout_ms))
        try:
            img = ids_peak_ipl_extension.BufferToImage(buf)
            img_mono = img.ConvertTo(self.pixel_format)
            frame = img_mono.get_numpy_2D().copy()
            if with_ts:
                return frame, int(img.Timestamp())
            return frame
        finally:
            self.datastream.QueueBuffer(buf)

    def iter_events(
        self,
        *,
        threshold: int = 8,
        timeout_ms: int = 1000,
    ) -> Iterator[EventsBatch]:
        """Yield synthetic CD-like events from consecutive mono frames."""
        prev, _ = self.read(timeout_ms=timeout_ms, with_ts=True)

        while True:
            curr, ts_us = self.read(timeout_ms=timeout_ms, with_ts=True)
            yield _frames_to_events(
                prev, curr, ts_us=ts_us, threshold=threshold
            )
            prev = curr

    def _select_device(
        self, devices: ids_peak.DeviceList
    ) -> ids_peak.DeviceInfo:
        device_count = int(devices.size())
        if self.serial is not None:
            return next(
                devices[i]
                for i in range(device_count)
                if str(devices[i].SerialNumber()) == self.serial
            )
        return devices[self.device_index]

    def _allocate_buffers(self) -> None:
        payload_node = self.remote.FindNode("PayloadSize")
        payload = int(payload_node.Value())
        min_buf = int(self.datastream.NumBuffersAnnouncedMinRequired())
        if self.buffer_count is None:
            nbuf = min_buf
        else:
            nbuf = max(min_buf, int(self.buffer_count))

        for _ in range(nbuf):
            buf = self.datastream.AllocAndAnnounceBuffer(payload)
            self.datastream.QueueBuffer(buf)

    def _start_acquisition(self) -> None:
        tl_params_locked = self.remote.FindNode("TLParamsLocked")
        tl_params_locked.SetValue(1)

        self.datastream.StartAcquisition()
        acquisition_start = self.remote.FindNode("AcquisitionStart")
        acquisition_start.Execute()
        acquisition_start.WaitUntilDone()
        self._acquiring = True

    def _stop_acquisition(self) -> None:
        acquisition_stop = self.remote.FindNode("AcquisitionStop")
        acquisition_stop.Execute()
        acquisition_stop.WaitUntilDone()
        self.datastream.StopAcquisition()
        tl_params_locked = self.remote.FindNode("TLParamsLocked")
        tl_params_locked.SetValue(0)
        self._acquiring = False

    def _shutdown(self) -> None:
        if self._acquiring:
            self._stop_acquisition()
        self._acquiring = False
        ids_peak.Library.Close()
