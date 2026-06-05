"""CameraDetector: the torch-free detect -> track -> patch/tokenize stage.

These exercise the worker-runnable detector in isolation (no torch, no
metavision_hal): the ready-track patch stacking, the v3 event tokenizer,
ring bookkeeping, and the best-effort ``safe_update`` wrapper.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from evio.core.events import EventsBatch
from evio.detection.live_camera import (
    EVENT_MAX,
    PATCH_SIZE,
    SEQ_LEN,
    CameraDetectionResult,
    CameraDetector,
    _concat_batches,
    safe_update,
)


class _Det:
    """Minimal stand-in for a tracked Detection (only track_id is read)."""

    def __init__(self, track_id: str) -> None:
        self.track_id = track_id


def _make_batch(t0_us: int, x: int, y: int, count: int = 200) -> EventsBatch:
    """Build a synthetic 10 ms batch worth of events at one screen point."""
    return EventsBatch(
        timestamps=(t0_us + np.arange(count, dtype=np.int64) * 50).astype(
            np.int64,
        ),
        x_coords=np.full(count, x, dtype=np.int16),
        y_coords=np.full(count, y, dtype=np.int16),
        polarities=np.tile(np.array([1, -1], dtype=np.int8), count // 2),
    )


def test_concat_batches_preserves_order() -> None:
    batches = tuple(
        _make_batch(i * 10_000, 100, 100, count=10) for i in range(3)
    )
    merged = _concat_batches(batches)
    assert merged.timestamps.size == 30
    assert np.all(np.diff(merged.timestamps) >= 0)


def test_update_returns_wellformed_result() -> None:
    det = CameraDetector(serial="camA")
    result = det.update(_make_batch(0, 100, 100))
    assert isinstance(result, CameraDetectionResult)
    assert isinstance(result.live_ids, frozenset)
    assert det._tick == 1
    assert result.patches is None  # no ring can be full after one window


def test_v3_update_grows_batch_ring() -> None:
    det = CameraDetector(serial="camA", is_v3=True)
    det.update(_make_batch(0, 100, 100, count=50))
    det.update(_make_batch(10_000, 100, 100, count=50))
    assert len(det._batch_ring) == 2


def test_classify_inputs_stacks_ready_patches() -> None:
    det = CameraDetector(serial="camA")
    patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    det._rings["t1"] = deque([patch] * SEQ_LEN, maxlen=SEQ_LEN)
    ready, patches, events, mask = det._classify_inputs([_Det("t1")])
    assert ready == ["t1"]
    assert patches is not None
    assert patches.shape == (1, SEQ_LEN, PATCH_SIZE, PATCH_SIZE, 3)
    assert patches.dtype == np.uint8
    assert events is None
    assert mask is None


def test_classify_inputs_skips_unfilled_ring() -> None:
    det = CameraDetector(serial="camA")
    patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    det._rings["t1"] = deque([patch] * (SEQ_LEN - 1), maxlen=SEQ_LEN)
    ready, patches, _events, _mask = det._classify_inputs([_Det("t1")])
    assert ready == []
    assert patches is None


def test_build_event_batch_tokenizes_per_ready_track() -> None:
    det = CameraDetector(serial="camA", is_v3=True, window_us=10_000)
    # SEQ_LEN consecutive 10 ms windows, all spiking at (100, 100) so the
    # bbox-cropped tokenizer keeps every event.
    for i in range(SEQ_LEN):
        det._batch_ring.append(_make_batch(i * 10_000, 100, 100, count=20))
    det._bbox_rings["track-1"] = deque(
        [((90, 90, 110, 110), i * 10_000 + 5_000) for i in range(SEQ_LEN)],
        maxlen=SEQ_LEN,
    )
    events, mask = det._build_event_batch(["track-1"])
    assert events.shape == (1, EVENT_MAX, 4)
    assert mask.shape == (1, EVENT_MAX)
    assert mask[0].any()  # at least some valid event tokens
    # t_norm / x / y normalised to [0, 1], polarity in {-1, 1}.
    valid = events[0, mask[0]]
    assert (valid[:, 0] >= 0).all() and (valid[:, 0] <= 1).all()
    assert (valid[:, 1] >= 0).all() and (valid[:, 1] <= 1).all()
    assert (valid[:, 2] >= 0).all() and (valid[:, 2] <= 1).all()
    assert set(np.unique(valid[:, 3]).tolist()).issubset({-1.0, 1.0})


def test_build_event_batch_returns_none_when_ring_empty() -> None:
    det = CameraDetector(serial="camA", is_v3=True)
    events, mask = det._build_event_batch(["track-1"])
    assert events is None
    assert mask is None


def test_safe_update_swallows_failure() -> None:
    det = CameraDetector(serial="camA")

    def _boom(_batch: object) -> CameraDetectionResult:
        msg = "synthetic detector failure"
        raise RuntimeError(msg)

    det.update = _boom  # type: ignore[method-assign]
    assert safe_update(det, _make_batch(0, 100, 100)) is None
