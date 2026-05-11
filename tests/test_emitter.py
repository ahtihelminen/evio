"""Unit tests for evio.detection.emitter.DetectionEmitter."""

from __future__ import annotations

import logging

from evio.detection.emitter import DetectionEmitter
from evio.detection.types import Detection


def _make_detection(x0: int = 0) -> Detection:
    """Cheapest valid Detection — defaults handle the optional fields."""
    return Detection(
        x0=x0,
        y0=0,
        x1=x0 + 10,
        y1=10,
        vx_px_s=0.0,
        vy_px_s=0.0,
        m_coh=0.5,
        best_psr=1.0,
        event_count=1,
        bg_class=0,
        row_start=0,
        row_end=1,
        col_start=0,
        col_end=1,
    )


def test_subscribe_and_emit_calls_callback() -> None:
    emitter = DetectionEmitter()
    received: list[Detection] = []
    emitter.subscribe(received.append)
    emitter.emit(_make_detection(x0=1))
    assert len(received) == 1
    assert received[0].x0 == 1


def test_multiple_subscribers_all_fire_in_order() -> None:
    emitter = DetectionEmitter()
    a: list[Detection] = []
    b: list[Detection] = []
    emitter.subscribe(a.append)
    emitter.subscribe(b.append)
    emitter.emit(_make_detection(x0=2))
    assert len(a) == 1
    assert len(b) == 1


def test_unsubscribe_stops_callback() -> None:
    emitter = DetectionEmitter()
    received: list[Detection] = []
    emitter.subscribe(received.append)
    emitter.unsubscribe(received.append)
    emitter.emit(_make_detection())
    assert received == []


def test_unsubscribe_unknown_callback_is_safe() -> None:
    emitter = DetectionEmitter()

    def never_subscribed(_: Detection) -> None: ...

    emitter.unsubscribe(never_subscribed)  # must not raise


def test_subscriber_exception_does_not_break_others(
    caplog: logging.LogCaptureFixture,
) -> None:
    """A buggy subscriber must not prevent later subscribers firing."""
    emitter = DetectionEmitter()
    received: list[Detection] = []

    def explodes(_: Detection) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    emitter.subscribe(explodes)
    emitter.subscribe(received.append)
    with caplog.at_level(logging.ERROR, logger="evio.detection.emitter"):
        emitter.emit(_make_detection())

    assert len(received) == 1, "second subscriber must still fire"
    assert any(
        "subscriber raised" in r.getMessage() for r in caplog.records
    ), "exception should be logged"


def test_subscriber_count_reflects_subscriptions() -> None:
    emitter = DetectionEmitter()
    assert emitter.subscriber_count == 0
    cb1: list[Detection] = []
    cb2: list[Detection] = []
    emitter.subscribe(cb1.append)
    emitter.subscribe(cb2.append)
    assert emitter.subscriber_count == 2
    emitter.unsubscribe(cb1.append)
    assert emitter.subscriber_count == 1
