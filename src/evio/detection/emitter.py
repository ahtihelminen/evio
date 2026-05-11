"""Detection emitter — subscribe/emit hook for detections.

A small pub/sub utility that the detection pipeline calls into so downstream
consumers (next: the SAPIENT adapter) can react to each detection as it is
produced, without the pipeline knowing what the consumer does.

Subscriber callbacks run synchronously from inside `emit()`. Exceptions raised
inside a callback are logged and swallowed: a buggy consumer must not be able
to take down the detection pipeline.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable

from evio.detection.types import Detection

DetectionCallback = Callable[[Detection], None]

_log = logging.getLogger(__name__)


class DetectionEmitter:
    """Tiny synchronous pub/sub for `Detection` objects."""

    def __init__(self) -> None:
        self._subscribers: list[DetectionCallback] = []

    def subscribe(self, callback: DetectionCallback) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: DetectionCallback) -> None:
        """Remove `callback` if it's subscribed; no-op otherwise."""
        with contextlib.suppress(ValueError):
            self._subscribers.remove(callback)

    def emit(self, detection: Detection) -> None:
        """Call every subscriber. Exceptions are logged, not propagated."""
        for cb in self._subscribers:
            try:
                cb(detection)
            except Exception:
                _log.exception(
                    "Detection subscriber raised; continuing",
                )

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
