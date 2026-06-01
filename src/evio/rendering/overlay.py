"""Draw live per-track drone verdicts onto a panorama frame.

Takes panorama-pixel boxes plus the verdicts produced by
``evio.detection.live_transformer.LiveTrackClassifier.classify`` and renders
one labelled rectangle per track. The box -> panorama-pixel mapping is the
caller's responsibility (wired to the live ring geometry at field bring-up);
this module only draws what it is handed.
"""

from __future__ import annotations

import cv2
import numpy as np

from evio.detection.live_transformer import TrackVerdict

DRONE_COLOR = (0, 200, 0)  # BGR green
OTHER_COLOR = (160, 160, 160)  # BGR gray

Box = tuple[int, int, int, int]


def _draw_label(
    frame: np.ndarray, text: str, anchor: Box, color: tuple[int, int, int],
) -> None:
    x0, y0, _, _ = anchor
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    top = y0 - th - base - 2
    if top < 0:  # box hugs the frame top -> drop the label inside
        top = y0 + 2
    bottom = top + th + base + 2
    cv2.rectangle(frame, (x0, top), (x0 + tw + 2, bottom), color, -1)
    cv2.putText(
        frame, text, (x0 + 1, top + th + 1), font, scale,
        (0, 0, 0), thick, cv2.LINE_AA,
    )


def draw_track_verdicts(
    frame: np.ndarray,
    boxes: dict[str, Box],
    verdicts: dict[str, TrackVerdict],
    *,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a labelled box per track that has both a box and a verdict.

    ``boxes`` maps track_id -> (x0, y0, x1, y1) in panorama pixels;
    ``verdicts`` is the dict returned by ``classify()``. Drawn in place;
    the frame is also returned for chaining. Tracks missing from either dict
    are skipped (a track warming up has a box but no verdict yet).
    """
    for track_id, verdict in verdicts.items():
        box = boxes.get(track_id)
        if box is None:
            continue
        color = DRONE_COLOR if verdict.is_drone else OTHER_COLOR
        x0, y0, x1, y1 = box
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)
        kind = "drone" if verdict.is_drone else "not"
        _draw_label(frame, f"{kind} {verdict.drone_prob:.2f}", box, color)
    return frame


__all__ = ["DRONE_COLOR", "OTHER_COLOR", "draw_track_verdicts"]
