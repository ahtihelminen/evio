from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType

import numpy as np
from ids_peak import ids_peak, ids_peak_ipl_extension  # type: ignore
from ids_peak_ipl import ids_peak_ipl  # type: ignore


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
    fps: float = 10.0

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

        self.remote.FindNode("AcquisitionFrameRate").SetValue(float(self.fps))

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
