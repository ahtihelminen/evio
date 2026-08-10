"""Grid rendering for the focus meter UI (pure functions, testable)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from evio.focus.sources import SENSOR_H, SENSOR_W, PaneSnapshot

_TEXT = (235, 235, 235)
_DIM = (110, 110, 110)
_ACCENT = (80, 220, 255)  # EMA line + score readout (BGR)
_PEAK = (90, 90, 255)
_ROI = (0, 200, 0)
_SELECTED = (60, 180, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class RenderConfig:
    view_w: int = 400
    graph_h: int = 96
    text_h: int = 38

    @property
    def view_h(self) -> int:
        return self.view_w * SENSOR_H // SENSOR_W

    @property
    def pane_h(self) -> int:
        return self.view_h + self.graph_h + self.text_h

    @property
    def graph_samples(self) -> int:
        return self.view_w * 4  # ~16 s of 100 Hz windows per pane width


def grid_layout(n_panes: int) -> tuple[int, int]:
    """(rows, cols): single row up to 3 cameras, 2x3 beyond."""
    if n_panes <= 3:
        return 1, max(1, n_panes)
    return 2, 3


def _normalize_view(view: NDArray[np.float32]) -> NDArray[np.uint8]:
    peak = float(view.max())
    if peak <= 0.0:
        return np.zeros(view.shape, dtype=np.uint8)
    scale = np.quantile(view[view > 0], 0.99) if (view > 0).any() else peak
    return np.clip(view * (255.0 / max(scale, 1e-6)), 0, 255).astype(
        np.uint8,
    )


def _pool_to_width(
    values: NDArray[np.float32], width: int,
) -> NDArray[np.float32]:
    """Max-pool a series down to ``width`` samples (peaks survive)."""
    if values.size <= width:
        return values
    kernel = int(np.ceil(values.size / width))
    padded = np.full(
        ((values.size + kernel - 1) // kernel) * kernel,
        -np.inf,
        dtype=np.float32,
    )
    padded[: values.size] = values
    return padded.reshape(-1, kernel).max(axis=1)


def _draw_series(
    strip: NDArray[np.uint8],
    values: NDArray[np.float32],
    v_max: float,
    color: tuple[int, int, int],
) -> None:
    if values.size < 2 or v_max <= 0.0:
        return
    height, width = strip.shape[:2]
    xs = np.linspace(0, width - 1, values.size).astype(np.int32)
    ys = np.clip(
        height - 1 - (values / v_max) * (height - 4), 0, height - 1,
    ).astype(np.int32)
    points = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(strip, [points], False, color, 1, cv2.LINE_AA)


def render_pane(
    snap: PaneSnapshot, cfg: RenderConfig, *, selected: bool,
) -> NDArray[np.uint8]:
    pane = np.zeros((cfg.pane_h, cfg.view_w, 3), dtype=np.uint8)

    # --- event view + ROI ------------------------------------------------
    view_rect = pane[: cfg.view_h]
    if snap.view is not None:
        gray = _normalize_view(snap.view)
        resized = cv2.resize(
            gray, (cfg.view_w, cfg.view_h),
            interpolation=cv2.INTER_NEAREST,
        )
        view_rect[:] = resized[:, :, None]
    sx = cfg.view_w / SENSOR_W
    sy = cfg.view_h / SENSOR_H
    rx, ry, rw, rh = snap.roi
    cv2.rectangle(
        view_rect,
        (int(rx * sx), int(ry * sy)),
        (int((rx + rw) * sx), int((ry + rh) * sy)),
        _ROI,
        1,
    )

    # --- score graph (raw dim, EMA bright, peak line) ---------------------
    strip = pane[cfg.view_h : cfg.view_h + cfg.graph_h]
    strip[:] = 18
    v_max = max(snap.peak_ema * 1.1, 1e-4)
    raw = _pool_to_width(snap.raw, cfg.view_w)
    ema = _pool_to_width(snap.ema, cfg.view_w)
    _draw_series(strip, raw, v_max, _DIM)
    _draw_series(strip, ema, v_max, _ACCENT)
    peak_y = int(
        np.clip(
            strip.shape[0] - 1
            - (snap.peak_ema / v_max) * (strip.shape[0] - 4),
            0,
            strip.shape[0] - 1,
        ),
    )
    for x0 in range(0, cfg.view_w, 12):
        cv2.line(strip, (x0, peak_y), (x0 + 5, peak_y), _PEAK, 1)

    # --- text block --------------------------------------------------------
    text = pane[cfg.view_h + cfg.graph_h :]
    line1 = f"{snap.serial}  {snap.status}"
    cv2.putText(text, line1, (6, 14), _FONT, 0.42, _TEXT, 1, cv2.LINE_AA)
    if snap.last is not None:
        pct_of_peak = (
            100.0 * snap.ema_now / snap.peak_ema
            if snap.peak_ema > 0
            else 0.0
        )
        line2 = (
            f"HF {snap.ema_now:.4f}  peak {snap.peak_ema:.4f}"
            f" ({pct_of_peak:3.0f}%)  roi {snap.last.roi_rate / 1e3:.0f}k"
            f" ev/s"
        )
        if snap.die_temp_c is not None:
            line2 += f"  {snap.die_temp_c:.0f}C"
        cv2.putText(
            text, line2, (6, 31), _FONT, 0.42, _ACCENT, 1, cv2.LINE_AA,
        )
    if selected:
        cv2.rectangle(
            pane, (0, 0), (cfg.view_w - 1, cfg.pane_h - 1), _SELECTED, 2,
        )
    return pane


def compose_canvas(
    snaps: list[PaneSnapshot],
    cfg: RenderConfig,
    selected_idx: int,
    footer: str,
) -> NDArray[np.uint8]:
    rows, cols = grid_layout(len(snaps))
    footer_h = 22
    canvas = np.zeros(
        (rows * cfg.pane_h + footer_h, cols * cfg.view_w, 3),
        dtype=np.uint8,
    )
    for idx, snap in enumerate(snaps):
        row, col = divmod(idx, cols)
        pane = render_pane(snap, cfg, selected=idx == selected_idx)
        canvas[
            row * cfg.pane_h : (row + 1) * cfg.pane_h,
            col * cfg.view_w : (col + 1) * cfg.view_w,
        ] = pane
    cv2.putText(
        canvas,
        footer,
        (6, canvas.shape[0] - 7),
        _FONT,
        0.4,
        _DIM,
        1,
        cv2.LINE_AA,
    )
    return canvas
