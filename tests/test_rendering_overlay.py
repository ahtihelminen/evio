"""Verdict-coloured detection boxes drawn over a rendered tile."""

from __future__ import annotations

import numpy as np
import pytest

from evio.detection.types import Classification, Detection
from evio.rendering.overlay import (
    CLUTTER_COLOR,
    DRONE_COLOR,
    UNKNOWN_COLOR,
    box_style,
    draw_detection_boxes,
    make_box_draw_fn,
)

pytestmark = pytest.mark.t1


def _detection(
    classification: Classification | None = None,
    x0: int = 20, y0: int = 30, x1: int = 60, y1: int = 70,
) -> Detection:
    return Detection(
        x0=x0, y0=y0, x1=x1, y1=y1, vx_px_s=0.0, vy_px_s=0.0, m_coh=0.5,
        best_psr=5.0, event_count=10, bg_class=0, row_start=0, row_end=0,
        col_start=0, col_end=0, classification=classification,
        track_id="track-0001",
    )


def _blank_frame() -> np.ndarray:
    return np.zeros((160, 240, 3), np.uint8)


def test_box_style_distinguishes_every_verdict() -> None:
    unknown = box_style(_detection())
    drone = box_style(_detection(Classification("drone", 0.9)))
    clutter = box_style(_detection(Classification("clutter", 0.8)))

    assert unknown == (UNKNOWN_COLOR, "?")
    assert drone[0] == DRONE_COLOR
    assert clutter[0] == CLUTTER_COLOR
    assert len({unknown[0], drone[0], clutter[0]}) == 3
    assert drone[1] and clutter[1]
    assert len({unknown[1], drone[1], clutter[1]}) == 3


def test_draw_detection_boxes_marks_bbox_edges_only() -> None:
    frame = _blank_frame()
    detection = _detection(Classification("drone", 0.9))

    drawn = draw_detection_boxes(frame, [detection])

    assert drawn is frame
    assert frame[50, 20].tolist() == list(DRONE_COLOR)  # left edge
    assert frame[50, 60].tolist() == list(DRONE_COLOR)  # right edge
    assert frame[70, 40].tolist() == list(DRONE_COLOR)  # bottom edge
    assert not frame[100:, :].any()  # below the box
    assert not frame[:, 180:].any()  # right of the box and its label


def test_draw_detection_boxes_empty_list_is_a_noop() -> None:
    frame = np.random.default_rng(0).integers(0, 256, (160, 240, 3), np.uint8)
    before = frame.tobytes()

    draw_detection_boxes(frame, [])

    assert frame.tobytes() == before


def test_box_draw_fn_draws_snapshot_detections() -> None:
    detection = _detection(Classification("clutter", 0.8))
    snapshots = {"cam-a": (None, [detection])}
    draw_fn = make_box_draw_fn(snapshots.get)
    expected = draw_detection_boxes(_blank_frame(), [detection])

    drawn = draw_fn("cam-a", _blank_frame())

    assert np.array_equal(drawn, expected)
    assert drawn[50, 20].tolist() == list(CLUTTER_COLOR)


def test_box_draw_fn_without_snapshot_leaves_frame_unchanged() -> None:
    draw_fn = make_box_draw_fn(lambda _serial: None)
    frame = _blank_frame()

    drawn = draw_fn("cam-a", frame)

    assert drawn is frame
    assert not frame.any()
