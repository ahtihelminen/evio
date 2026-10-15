"""Named taps: observe any stage output without the stage knowing.

Overlays, telemetry, recorders and test assertions subscribe to a tap by
name. Publishing costs one dict lookup when nobody listens, and callers can
ask `listening()` to skip building the value at all. A subscriber's
exception propagates to the publisher; whether that is isolated is the fault
policy's decision, not the bus's. A callback receives `(step_index, value)`;
a stage tap is published per stream and per call, so one step can publish a
tap several times.
"""

from __future__ import annotations

from collections.abc import Callable

TapCallback = Callable[[int, object], None]


class TapBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[TapCallback]] = {}

    def subscribe(self, name: str, callback: TapCallback) -> Callable[[], None]:
        """Listen to `name`; returns a function that unsubscribes."""
        self._subscribers.setdefault(name, []).append(callback)

        def unsubscribe() -> None:
            listeners = self._subscribers.get(name, [])
            if callback in listeners:
                listeners.remove(callback)

        return unsubscribe

    def listening(self, name: str) -> bool:
        return bool(self._subscribers.get(name))

    def publish(self, name: str, step_index: int, value: object) -> None:
        for callback in self._subscribers.get(name, ()):
            callback(step_index, value)
