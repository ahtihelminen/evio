from __future__ import annotations

import logging
import sys
import types
from unittest.mock import MagicMock

import pytest


def _install_fake_ids_peak() -> None:
    """Stub sys.modules for ids_peak/ids_peak_ipl/ids_peak_ipl_extension so
    `evio.io.mono_backends.ids_peak` is importable without the real IDS Peak
    SDK.
    """
    ids_peak_pkg = types.ModuleType("ids_peak")
    ids_peak_sub = types.ModuleType("ids_peak.ids_peak")
    ids_peak_ext_sub = types.ModuleType("ids_peak.ids_peak_ipl_extension")

    ids_peak_sub.Library = MagicMock()
    ids_peak_sub.DeviceManager = MagicMock()
    ids_peak_sub.DeviceAccessType_Control = object()
    ids_peak_pkg.ids_peak = ids_peak_sub
    ids_peak_pkg.ids_peak_ipl_extension = ids_peak_ext_sub

    ids_peak_ipl_pkg = types.ModuleType("ids_peak_ipl")
    ids_peak_ipl_sub = types.ModuleType("ids_peak_ipl.ids_peak_ipl")
    ids_peak_ipl_sub.PixelFormatName_Mono8 = object()
    ids_peak_ipl_pkg.ids_peak_ipl = ids_peak_ipl_sub

    sys.modules["ids_peak"] = ids_peak_pkg
    sys.modules["ids_peak.ids_peak"] = ids_peak_sub
    sys.modules["ids_peak.ids_peak_ipl_extension"] = ids_peak_ext_sub
    sys.modules["ids_peak_ipl"] = ids_peak_ipl_pkg
    sys.modules["ids_peak_ipl.ids_peak_ipl"] = ids_peak_ipl_sub


_install_fake_ids_peak()

from evio.io.mono_backends import ids_peak as mono_camera  # noqa: E402


class _FakeNode:
    def __init__(self, value: object = 0) -> None:
        self._value = value

    def Value(self) -> object:
        return self._value

    def SetValue(self, value: object) -> None:
        self._value = value

    def FindEntry(self, name: str) -> str:
        return name

    def SetCurrentEntry(self, _entry: object) -> None:
        pass

    def Execute(self) -> None:
        pass

    def WaitUntilDone(self) -> None:
        pass


class _FakeRemote:
    def __init__(self, width: int, height: int) -> None:
        self._nodes = {
            "Width": _FakeNode(width),
            "Height": _FakeNode(height),
            "ExposureAuto": _FakeNode(),
            "GainAuto": _FakeNode(),
            "ExposureTime": _FakeNode(100.0),
            "AcquisitionFrameRate": _FakeNode(30.0),
            "PayloadSize": _FakeNode(width * height),
            "TLParamsLocked": _FakeNode(),
            "AcquisitionStart": _FakeNode(),
            "AcquisitionStop": _FakeNode(),
        }

    def FindNode(self, name: str) -> _FakeNode:
        return self._nodes[name]


class _FakeRemoteDevice:
    def __init__(self, remote: _FakeRemote) -> None:
        self._remote = remote

    def NodeMaps(self) -> list[_FakeRemote]:
        return [self._remote]


class _FakeDataStream:
    def NumBuffersAnnouncedMinRequired(self) -> int:
        return 1

    def AllocAndAnnounceBuffer(self, _payload: int) -> object:
        return object()

    def QueueBuffer(self, _buf: object) -> None:
        pass

    def StartAcquisition(self) -> None:
        pass

    def StopAcquisition(self) -> None:
        pass


class _FakeDataStreamHolder:
    def __init__(self, datastream: _FakeDataStream) -> None:
        self._datastream = datastream

    def OpenDataStream(self) -> _FakeDataStream:
        return self._datastream


class _FakeDevice:
    def __init__(
        self, remote: _FakeRemote, datastream: _FakeDataStream
    ) -> None:
        self._remote = remote
        self._datastream_holder = _FakeDataStreamHolder(datastream)

    def RemoteDevice(self) -> _FakeRemoteDevice:
        return _FakeRemoteDevice(self._remote)

    def DataStreams(self) -> list[_FakeDataStreamHolder]:
        return [self._datastream_holder]


class _FakeDeviceInfo:
    def __init__(self, device: _FakeDevice) -> None:
        self._device = device

    def OpenDevice(self, _access_type: object) -> _FakeDevice:
        return self._device


class _FakeDevices(list["_FakeDeviceInfo"]):
    def size(self) -> int:
        return len(self)


def _make_source(
    width: int = 4, height: int = 3
) -> mono_camera.MonoCameraSource:
    remote = _FakeRemote(width, height)
    datastream = _FakeDataStream()
    device = _FakeDevice(remote, datastream)
    dev_info = _FakeDeviceInfo(device)

    manager = MagicMock()
    manager.Devices.return_value = _FakeDevices([dev_info])
    mono_camera.ids_peak.DeviceManager.Instance.return_value = manager

    return mono_camera.MonoCameraSource()


def test_enter_succeeds_when_warmup_compile_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mono_camera,
        "get_events_from_frames",
        MagicMock(side_effect=RuntimeError("numba boom")),
    )

    source = _make_source()
    with source as entered:
        assert entered is source


def test_iter_events_raises_recorded_warmup_failure_before_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmup_error = RuntimeError("numba boom")
    monkeypatch.setattr(
        mono_camera,
        "get_events_from_frames",
        MagicMock(side_effect=warmup_error),
    )

    source = _make_source()
    with source:
        read_mock = MagicMock()
        monkeypatch.setattr(source, "read", read_mock)

        events = source.iter_events()
        with pytest.raises(RuntimeError) as exc_info:
            next(events)

        assert exc_info.value is warmup_error
        read_mock.assert_not_called()


def test_warmup_failure_logged_as_error_without_lazy_compile_promise(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        mono_camera,
        "get_events_from_frames",
        MagicMock(side_effect=RuntimeError("numba boom")),
    )

    source = _make_source()
    with (
        caplog.at_level(
            logging.WARNING, logger="evio.io.mono_backends.ids_peak"
        ),
        source,
    ):
        pass

    assert not any("compile lazily" in r.getMessage() for r in caplog.records)
    warmup_records = [r for r in caplog.records if "warm-up" in r.getMessage()]
    assert warmup_records
    assert all(r.levelno == logging.ERROR for r in warmup_records)
