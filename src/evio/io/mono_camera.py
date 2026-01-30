from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from types import TracebackType

import numpy as np

# not available on mac, hence ignored
from ids_peak import ids_peak, ids_peak_ipl_extension  # type: ignore
from ids_peak_ipl import ids_peak_ipl  # type: ignore
from numba import njit, prange

from evio.core.events import EventsBatch


@njit(cache=True, fastmath=True, parallel=True)
def get_mean_change(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
    frame_height, frame_width = prev_frame.shape
    total_change = 0.0
    for row in prange(frame_height):
        row_total_change = 0.0
        for col in range(frame_width):
            row_total_change += float(curr_frame[row, col]) - float(
                prev_frame[row, col]
            )
        total_change += row_total_change
    return total_change / (frame_height * frame_width)


@njit(cache=True, fastmath=True, parallel=True)
def get_event_count(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    mean_change: float,
    threshold: float,
) -> int:
    frame_height, frame_weight = prev_frame.shape
    event_count = 0
    for row in prange(frame_height):
        row_count = 0
        for col in range(frame_weight):
            normalized_change = (
                float(curr_frame[row, col]) - float(prev_frame[row, col])
            ) - mean_change
            if (
                normalized_change >= threshold
                or normalized_change <= -threshold
            ):
                row_count += 1
        event_count += row_count
    return event_count


@njit(cache=True, fastmath=True)
def fill_event_arrays(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    mean_change: float,
    threshold: float,
    timestamp_us: int,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    timestamps: np.ndarray,
    polarities: np.ndarray,
) -> None:
    frame_height, frame_width = prev_frame.shape
    idx = 0
    for row in range(frame_height):
        for col in range(frame_width):
            d = (
                float(curr_frame[row, col]) - float(prev_frame[row, col])
            ) - mean_change
            if d >= threshold or d <= -threshold:
                x_coords[idx] = col
                y_coords[idx] = row
                polarities[idx] = 1 if d > 0 else 0
                timestamps[idx] = timestamp_us
                idx += 1


def get_events_from_frames(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    timestamp_us: int,
    threshold: float,
) -> EventsBatch:
    mean_change = get_mean_change(prev_frame, curr_frame)
    event_count = get_event_count(
        prev_frame, curr_frame, mean_change, threshold
    )

    x_coords = np.empty(event_count, dtype=np.int16)
    y_coords = np.empty(event_count, dtype=np.int16)
    polarities = np.empty(event_count, dtype=np.int8)
    timestamps = np.empty(event_count, dtype=np.int64)

    fill_event_arrays(
        prev_frame,
        curr_frame,
        mean_change,
        threshold,
        timestamp_us,
        x_coords,
        y_coords,
        timestamps,
        polarities,
    )
    return EventsBatch(
        timestamps,
        x_coords,
        y_coords,
        polarities,
    )


# ---------------- Camera source ----------------


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

        # Warm up JIT once so first iter_events() yield doesn't pay compile cost
        dummy_prev = np.zeros((self.height, self.width), dtype=np.uint8)
        dummy_curr = np.zeros((self.height, self.width), dtype=np.uint8)
        get_events_from_frames(dummy_prev, dummy_curr, 0, 8)

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
        prev, _ = self.read(timeout_ms=timeout_ms, with_ts=True)
        while True:
            curr, ts_us = self.read(timeout_ms=timeout_ms, with_ts=True)
            yield get_events_from_frames(
                prev, curr, timestamp_us=ts_us, threshold=threshold
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
        nbuf = (
            min_buf
            if self.buffer_count is None
            else max(min_buf, int(self.buffer_count))
        )

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
