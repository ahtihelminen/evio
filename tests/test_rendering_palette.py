"""Event palette: green = ON/brighter, red = OFF/dimmer, black = still."""

from __future__ import annotations

import numpy as np

from evio.rendering.frame import get_frame
from evio.rendering.numba import NumbaEventRenderer

BLACK = [0, 0, 0]
GREEN = [0, 255, 0]
RED = [0, 0, 255]


def test_get_frame_palette() -> None:
    x = np.array([1, 2], np.int32)
    y = np.array([3, 4], np.int32)
    polarity_on = np.array([True, False])
    frame = get_frame((x, y, polarity_on), width=8, height=8)
    assert frame[0, 0].tolist() == BLACK
    assert frame[3, 1].tolist() == GREEN
    assert frame[4, 2].tolist() == RED


def test_numba_renderer_palette() -> None:
    renderer = NumbaEventRenderer(8, 8)
    frame = renderer.render(
        np.array([1, 2], np.int32),
        np.array([3, 4], np.int32),
        np.array([True, False]),
    )
    assert frame[0, 0].tolist() == BLACK
    assert frame[3, 1].tolist() == GREEN
    assert frame[4, 2].tolist() == RED
