"""Detection-box overlay drawing shared by the RTSP panorama and the web UI.

One palette and one box-drawing routine for the verdict-coloured bounding
boxes the operator sees, whether composited onto the panorama RTSP frame or
the per-camera MJPEG. The verdict label is read straight off
``Detection.classification`` — the persistence gate has already committed it
upstream in the detection authority (the OPS appliance's
loop attaches it).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from evio.core.events import EventsBatch
    from evio.detection.types import Detection

    FrameSnapshot = tuple[EventsBatch, list[Detection]]

# One tile's overlay pass: `draw_fn(serial, frame) -> frame`. Named because
# the panorama composite, the MJPEG route and the app's frame_draw_fn hook
# all pass this same shape around.
DrawFn = Callable[[str, np.ndarray], np.ndarray]

# BGR. Pure green/red are taken by ON/OFF event pixels in the renderer.
DRONE_COLOR = (0, 255, 255)  # yellow — classifier says drone
CLUTTER_COLOR = (0, 200, 255)  # amber — classifier says not-drone
UNKNOWN_COLOR = (170, 170, 170)  # grey — tracked, not yet classified


def box_style(detection: Detection) -> tuple[tuple[int, int, int], str]:
    """Verdict colour + label tag for one detection's classification."""
    cls = detection.classification
    if cls is None:
        return UNKNOWN_COLOR, "?"
    color = DRONE_COLOR if cls.label == "drone" else CLUTTER_COLOR
    return color, f"{cls.label} {cls.confidence:.2f}"


def draw_detection_boxes(
    frame: np.ndarray, detections: list[Detection],
) -> np.ndarray:
    """Draw verdict-coloured bboxes + labels onto ``frame`` (mutated)."""
    for det in detections:
        color, tag = box_style(det)
        cv2.rectangle(frame, (det.x0, det.y0), (det.x1, det.y1), color, 2)
        label = f"{det.track_id[-4:]} {tag}" if det.track_id else tag
        cv2.putText(
            frame, label, (det.x0, max(12, det.y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return frame


def make_box_draw_fn(
    latest_getter: Callable[[str], FrameSnapshot | None],
) -> DrawFn:
    """Build a per-tile ``draw_fn(serial, frame)`` for ``ordered_frame_views``.

    Reads the latest ``(batch, tracked)`` for the camera (the same snapshot
    the web UI MJPEG draws) and overlays the tracked detections' verdict boxes
    on the worker-rendered tile, in sensor coordinates. ``ordered_frame_views``
    applies the 180-degree flip afterwards so boxes track flipped tiles.
    """

    def draw_fn(serial: str, frame: np.ndarray) -> np.ndarray:
        snapshot = latest_getter(serial)
        if snapshot is None:
            return frame
        _batch, detections = snapshot
        return draw_detection_boxes(frame, detections)

    return draw_fn


__all__ = [
    "CLUTTER_COLOR",
    "DRONE_COLOR",
    "UNKNOWN_COLOR",
    "box_style",
    "draw_detection_boxes",
    "make_box_draw_fn",
]
