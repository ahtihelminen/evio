"""Pane workers and shared state: synthetic feed, ROI ops, masking."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from evio.focus.sources import (
    SENSOR_H,
    SENSOR_W,
    PaneConfig,
    PaneState,
    SyntheticScene,
    SyntheticWorker,
    _mask_from_coords,
)

pytestmark = pytest.mark.t2


def test_synthetic_scene_is_deterministic() -> None:
    xs_a, ys_a = SyntheticScene(seed=7).window_events(0.01, 0.01)
    xs_b, ys_b = SyntheticScene(seed=7).window_events(0.01, 0.01)
    assert np.array_equal(xs_a, xs_b)
    assert np.array_equal(ys_a, ys_b)


def test_synthetic_worker_populates_pane() -> None:
    config = PaneConfig(window_ms=10.0, boot_scan_s=0.0)
    pane = PaneState("synt-test", config)
    stop = threading.Event()
    worker = SyntheticWorker(pane, stop)
    worker.start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        with pane.lock:
            if pane.windows >= 5 and pane.view is not None:
                break
        time.sleep(0.02)
    stop.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    snap = pane.snapshot(graph_samples=64)
    assert snap.windows >= 5
    assert snap.view is not None and snap.view.shape == (
        SENSOR_H // config.view_ds,
        SENSOR_W // config.view_ds,
    )
    assert snap.last is not None
    assert snap.raw.size > 0


def test_roi_move_and_resize_stay_in_bounds() -> None:
    pane = PaneState("x", PaneConfig(roi_size=384))
    for _ in range(200):
        pane.move_roi(-64, -64)
    assert pane.roi[:2] == (0, 0)
    for _ in range(200):
        pane.move_roi(64, 64)
    rx, ry, rw, rh = pane.roi
    assert rx + rw == SENSOR_W
    assert ry + rh == SENSOR_H
    for _ in range(50):
        pane.resize_roi(128)
    assert pane.roi[3] <= SENSOR_H
    for _ in range(50):
        pane.resize_roi(-128)
    assert pane.roi[2] >= 64


def test_center_roi_on_activity_finds_hot_block() -> None:
    config = PaneConfig(roi_size=256)
    pane = PaneState("x", config)
    view = np.zeros(
        (SENSOR_H // config.view_ds, SENSOR_W // config.view_ds),
        dtype=np.float32,
    )
    view[60, 100] = 100.0  # sensor coords ~ (400, 240), clamp-free
    with pane.lock:
        pane.view = view
    pane.center_roi_on_activity()
    rx, ry, rw, rh = pane.roi
    assert abs((rx + rw // 2) - 100 * config.view_ds) <= config.view_ds
    assert abs((ry + rh // 2) - 60 * config.view_ds) <= config.view_ds


def test_mask_from_coords() -> None:
    assert _mask_from_coords([]) is None
    mask = _mask_from_coords([(5, 7), (1279, 719)])
    assert mask is not None
    assert mask[7, 5] and mask[719, 1279]
    assert mask.sum() == 2


def test_consume_survives_out_of_bounds_coordinates() -> None:
    pane = PaneState("x", PaneConfig())
    worker = SyntheticWorker(pane, threading.Event())
    worker._consume(
        np.array([100, 1300, -1, 5000], dtype=np.int32),
        np.array([100, 100, 100, 900], dtype=np.int32),
    )
    snap = pane.snapshot(graph_samples=8)
    assert snap.windows == 1  # did not raise; window still scored


def test_roi_change_resets_peak_hold() -> None:
    pane = PaneState("x", PaneConfig())
    with pane.lock:
        pane.history.push(0.5)
        pane.history.push(0.1)
    assert pane.history.peak_ema > pane.history.ema
    pane.move_roi(16, 0)
    assert pane.history.peak_ema == pane.history.ema
    # a clamped no-op move must NOT reset an established peak
    pane.move_roi(-10_000, 0)  # park at the left edge
    with pane.lock:
        pane.history.push(0.5)
        pane.history.push(0.1)
    peak = pane.history.peak_ema
    assert peak > pane.history.ema
    pane.move_roi(-16, 0)  # clamps to the same position = no-op
    assert pane.history.peak_ema == peak


def test_snapshot_view_does_not_alias_worker_accumulator() -> None:
    pane = PaneState("x", PaneConfig())
    worker = SyntheticWorker(pane, threading.Event())
    xs = np.full(100, 640, dtype=np.int32)
    ys = np.full(100, 360, dtype=np.int32)
    worker._consume(xs, ys)
    snap = pane.snapshot(graph_samples=8)
    before = snap.view.copy()
    worker._consume(xs, ys)
    assert np.array_equal(snap.view, before)


def test_view_excludes_stuck_pixels() -> None:
    """Auto-ROI must never center on a masked stuck pixel."""
    pane = PaneState("x", PaneConfig())
    worker = SyntheticWorker(pane, threading.Event())
    worker.stuck_mask = _mask_from_coords([(200, 300)])
    xs = np.concatenate([
        np.full(500, 200, dtype=np.int32),  # stuck pixel flood
        np.full(20, 900, dtype=np.int32),  # real activity
    ])
    ys = np.concatenate([
        np.full(500, 300, dtype=np.int32),
        np.full(20, 400, dtype=np.int32),
    ])
    worker._consume(xs, ys)
    pane.center_roi_on_activity()
    rx, ry, rw, rh = pane.roi
    assert abs((rx + rw // 2) - 900) <= pane.config.view_ds
    assert abs((ry + rh // 2) - 400) <= pane.config.view_ds
