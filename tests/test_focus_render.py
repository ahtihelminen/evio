"""Rendering: layout, decimation, canvas composition."""

from __future__ import annotations

import numpy as np

from evio.focus.metric import FocusResult
from evio.focus.render import (
    RenderConfig,
    _pool_to_width,
    compose_canvas,
    grid_layout,
)
from evio.focus.sources import PaneSnapshot


def _snap(serial: str) -> PaneSnapshot:
    return PaneSnapshot(
        serial=serial,
        status="synthetic",
        view=np.random.default_rng(0)
        .random((180, 320))
        .astype(np.float32),
        roi=(448, 168, 384, 384),
        raw=np.linspace(0, 0.2, 500, dtype=np.float32),
        ema=np.linspace(0, 0.18, 500, dtype=np.float32),
        last=FocusResult(hf=0.18, roi_rate=25_000.0, frame_rate=40_000.0),
        ema_now=0.18,
        peak_ema=0.21,
        die_temp_c=41.0,
        windows=1234,
    )


def test_grid_layout() -> None:
    assert grid_layout(1) == (1, 1)
    assert grid_layout(2) == (1, 2)
    assert grid_layout(3) == (1, 3)
    assert grid_layout(4) == (2, 3)
    assert grid_layout(6) == (2, 3)


def test_pool_to_width_keeps_peaks() -> None:
    values = np.zeros(1000, dtype=np.float32)
    values[123] = 7.0
    pooled = _pool_to_width(values, 400)
    assert pooled.size <= 400
    assert pooled.max() == 7.0
    short = np.ones(10, dtype=np.float32)
    assert _pool_to_width(short, 400) is short


def test_compose_canvas_shapes() -> None:
    cfg = RenderConfig()
    two = compose_canvas(
        [_snap("a"), _snap("b")], cfg, selected_idx=0, footer="f",
    )
    assert two.shape == (cfg.pane_h + 22, 2 * cfg.view_w, 3)
    assert two.dtype == np.uint8
    six = compose_canvas(
        [_snap(str(i)) for i in range(6)], cfg, selected_idx=5,
        footer="f",
    )
    assert six.shape == (2 * cfg.pane_h + 22, 3 * cfg.view_w, 3)
    assert six.any()
