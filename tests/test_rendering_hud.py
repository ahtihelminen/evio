"""Playback HUD: timing text drawn over the rendered frame."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from fakes.pacer import FakePacer, FakeStreamingPacer

from evio.core.events import BatchRange
from evio.rendering.frame import draw_hud, draw_hud_stream

pytestmark = pytest.mark.t1


def _blank_frame() -> np.ndarray:
    return np.zeros((60, 640, 3), np.uint8)


def _reference(*rows: str) -> np.ndarray:
    frame = _blank_frame()
    for row_index, text in enumerate(rows):
        cv2.putText(
            frame, text, (8, 20 + 20 * row_index), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame


def test_draw_hud_overlays_timing_without_changing_frame_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pacer = FakePacer(speed=2.0, _t_start=100.0, _e_start=1_000_000)
    monkeypatch.setattr(time, "perf_counter", lambda: 101.5)  # wall = 1.5 s
    frame = _blank_frame()

    draw_hud(frame, pacer, BatchRange(0, 0, 0, end_ts_us=4_000_000))

    assert frame.shape == (60, 640, 3)
    assert frame.dtype == np.uint8
    assert frame.any()
    assert np.array_equal(frame, _reference(
        "target speed=2.00x  actual speed=2.00x",
        "wall=  1.500s  rec=  3.000s",
    ))


def test_draw_hud_stream_overlays_timing_without_changing_frame_shape() -> None:
    pacer = FakeStreamingPacer(wall_s=1.5, rec_start_us=1_000_000)
    frame = _blank_frame()

    draw_hud_stream(frame, pacer, end_ts_us=4_000_000)

    assert frame.shape == (60, 640, 3)
    assert frame.dtype == np.uint8
    assert frame.any()
    assert np.array_equal(frame, _reference(
        "target=2.00x  inst=1.75x  actual=2.00x  wall=  1.500s  rec=  3.000s",
    ))
