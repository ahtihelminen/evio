"""A U3V mono camera stand-in that a killed recorder has left streaming.

`fake_u3v_module()` is the driver module (`Camera`, `AutoMode`) to put in
`sys.modules["u3v"]`. Its camera starts out streaming, as the real one does
after its recorder is SIGKILLed, and `start()` rejects until `stop()` has
released that stream. `calls` lists the `start`/`stop` calls in order.

Behaviour measured on ops001 (2026-10-20), not assumed: the rejection is
`RuntimeError: Gencp: command 0x0802 rejected, status 0x8fff`, and
`stop()` on an idle camera is harmless.
"""

from __future__ import annotations

import types
from enum import Enum


class FakeAutoMode(Enum):
    Off = 0
    Continuous = 2


class FakeU3vCamera:
    Width = 4504
    Height = 4504

    def __init__(self) -> None:
        self.streaming = True
        self.calls: list[str] = []

    def start(self) -> None:
        self.calls.append("start")
        if self.streaming:
            msg = "Gencp: command 0x0802 rejected, status 0x8fff (unknown)"
            raise RuntimeError(msg)
        self.streaming = True

    def stop(self) -> None:
        self.calls.append("stop")
        self.streaming = False

    def close(self) -> None:
        pass


def fake_u3v_module(camera: FakeU3vCamera) -> types.ModuleType:
    module = types.ModuleType("u3v")
    module.AutoMode = FakeAutoMode  # type: ignore[attr-defined]
    module.Camera = types.SimpleNamespace(open=lambda _serial: camera)  # type: ignore[attr-defined]
    return module
