"""A U3V mono-camera stand-in that records its lifecycle calls.

`FakeU3vCamera` records `start`, `stop` and `close` in `calls`, in order.
Given a `stop_error`, `stop()` raises it after recording the call, as the
real camera does once it has stopped answering. `fake_u3v_module(camera)`
is the driver module to put in `sys.modules["u3v"]`; its `Camera.open`
hands back that camera whatever serial it is given.

Behaviours assumed of the real driver, not proved here:
- `stop()` raises `RuntimeError` ("Gencp: bulk IN failed or short ack")
  when the camera no longer answers its control channel;
- `close()` after a failed `stop()` releases the camera instead of failing
  the same way (U3V `fix/stop-survives-usb-failure`).

Differences from the real device:
- it records calls and grabs no frames: there is no stream, exposure or
  gain, only `start`, `stop` and `close`;
- `Camera.open` returns the one camera for any serial, where the driver
  fails for a serial that is not attached;
- the two assumed behaviours above are not checked against the driver.
"""

from __future__ import annotations

import types


class FakeU3vCamera:
    Width = 8
    Height = 8

    def __init__(self, *, stop_error: Exception | None = None) -> None:
        self.calls: list[str] = []
        self._stop_error = stop_error

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")
        if self._stop_error is not None:
            raise self._stop_error

    def close(self) -> None:
        self.calls.append("close")


def fake_u3v_module(camera: FakeU3vCamera) -> types.ModuleType:
    module = types.ModuleType("u3v")
    module.Camera = types.SimpleNamespace(open=lambda _serial: camera)
    module.AutoMode = types.SimpleNamespace(Continuous="Continuous", Off="Off")
    return module
