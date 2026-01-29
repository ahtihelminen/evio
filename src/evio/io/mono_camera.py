from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType

import numpy as np
from ids_peak import ids_peak, ids_peak_ipl_extension  # type: ignore
from ids_peak_ipl import ids_peak_ipl  # type: ignore


def _node(remote: ids_peak.NodeMap, name: str) -> ids_peak.Node:
    n = remote.FindNode(name)
    if n is None:
        msg = f"Missing GenICam node: {name}"
        raise RuntimeError(msg)
    return n


@dataclass
class IDSMonoSource:
    """Minimal 'CameraSource-like' wrapper for IDS peak monochrome frames.

    Usage:
        with IDSMonoSource(device_index=0) as cam:
            frame = cam.read()              # np.ndarray[uint8] (H, W)
            frame, ts = cam.read(with_ts=True)
    """

    device_index: int = 0
    serial: str | None = None  # optional: open by serial if you want
    pixel_format: ids_peak_ipl.PixelFormatName = (
        ids_peak_ipl.PixelFormatName_Mono8
    )
    buffer_count: int | None = (
        None  # if None -> use NumBuffersAnnouncedMinRequired()
    )

    # runtime handles
    device: ids_peak.Device | None = field(init=False, default=None)
    remote: ids_peak.NodeMap | None = field(init=False, default=None)
    datastream: ids_peak.DataStream | None = field(init=False, default=None)

    width: int = field(init=False, default=0)
    height: int = field(init=False, default=0)

    def __enter__(self) -> IDSMonoSource:
        ids_peak.Library.Initialize()

        dm = ids_peak.DeviceManager.Instance()
        dm.Update()

        if dm.Devices().empty():
            ids_peak.Library.Close()
            msg = "No IDS cameras found."
            raise RuntimeError(msg)

        if self.serial is not None:
            dev_info = None
            for i in range(int(dm.Devices().size())):
                d = dm.Devices()[i]
                if str(d.SerialNumber()) == self.serial:
                    dev_info = d
                    break
            if dev_info is None:
                ids_peak.Library.Close()
                msg = f"IDS camera with serial={self.serial!r} not found."
                raise RuntimeError(msg)
        else:
            if self.device_index < 0 or self.device_index >= int(
                dm.Devices().size()
            ):
                ids_peak.Library.Close()
                msg = (
                    f"device_index={self.device_index} out of range "
                    f"(0..{int(dm.Devices().size()) - 1})"
                )
                raise RuntimeError(msg)
            dev_info = dm.Devices()[self.device_index]

        dev = dev_info.OpenDevice(ids_peak.DeviceAccessType_Control)
        self.device = dev

        remote = dev.RemoteDevice().NodeMaps()[0]
        self.remote = remote

        ds = dev.DataStreams()[0].OpenDataStream()
        self.datastream = ds

        # Geometry (best-effort)
        self.width = int(_node(remote, "Width").Value())
        self.height = int(_node(remote, "Height").Value())

        # Allocate/announce buffers
        payload = int(_node(remote, "PayloadSize").Value())
        min_buf = int(ds.NumBuffersAnnouncedMinRequired())
        nbuf = self.buffer_count if self.buffer_count is not None else min_buf
        if nbuf < min_buf:
            nbuf = min_buf

        for _ in range(nbuf):
            buf = ds.AllocAndAnnounceBuffer(payload)
            ds.QueueBuffer(buf)

        # Start acquisition
        _node(remote, "TLParamsLocked").SetValue(1)
        ds.StartAcquisition()
        _node(remote, "AcquisitionStart").Execute()
        _node(remote, "AcquisitionStart").WaitUntilDone()

        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        try:
            if self.remote is not None and self.datastream is not None:
                _node(self.remote, "AcquisitionStop").Execute()
                _node(self.remote, "AcquisitionStop").WaitUntilDone()

                self.datastream.StopAcquisition()
                _node(self.remote, "TLParamsLocked").SetValue(0)
        finally:
            self.device = None
            self.remote = None
            self.datastream = None
            ids_peak.Library.Close()

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
        if self.datastream is None:
            msg = "IDSMonoSource not started; use as a context manager."
            raise RuntimeError(msg)

        buf = self.datastream.WaitForFinishedBuffer(int(timeout_ms))
        try:
            img = ids_peak_ipl_extension.BufferToImage(buf)
            img_mono = img.ConvertTo(self.pixel_format)

            h = int(img_mono.Height())
            w = int(img_mono.Width())

            # ids_peak_ipl gives mono frames as 2D numpy array
            frame = img_mono.get_numpy_2D().reshape(h, w).copy()

            ts_us = 0
            if with_ts:
                # Not all configurations expose a timestamp;
                # keep best-effort + safe default.
                ts_us = int(img.Timestamp())

            return (frame, ts_us) if with_ts else frame
        finally:
            # Re-queue buffer ASAP
            self.datastream.QueueBuffer(buf)
