"""Pane workers and shared state: synthetic feed, ROI ops, masking."""

from __future__ import annotations

import threading
import time

import numpy as np

from evio.focus.sources import (
    SENSOR_H,
    SENSOR_W,
    PaneConfig,
    PaneState,
    SyntheticScene,
    SyntheticWorker,
    _mask_from_coords,
)


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
