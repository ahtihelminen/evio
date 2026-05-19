"""Track-manager behaviour: persistent IDs, Kalman gating, miss-evict."""

from __future__ import annotations

import numpy as np

from evio.detection.tracker import TrackManager, TrackManagerConfig
from evio.detection.types import Detection


def _det(
    cx: float,
    cy: float,
    *,
    half: int = 10,
    vx: float = 0.0,
    vy: float = 0.0,
    m_coh: float = 0.8,
    patch: np.ndarray | None = None,
) -> Detection:
    return Detection(
        x0=int(cx - half), y0=int(cy - half),
        x1=int(cx + half), y1=int(cy + half),
        vx_px_s=vx, vy_px_s=vy,
        m_coh=m_coh, best_psr=1.0, event_count=100,
        bg_class=0, row_start=0, row_end=1, col_start=0, col_end=1,
        z_score=3.0, camera_id="cam",
        patch=patch,
    )


def _patch(seed: int) -> np.ndarray:
    """A deterministic uint8 BGR patch."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)


def test_first_window_spawns_fresh_tracks() -> None:
    tracker = TrackManager(TrackManagerConfig(confirm_after_hits=1))
    a = _det(100, 100)
    b = _det(400, 400)
    tracked = tracker.update([a, b], now_us=0)
    assert len(tracked) == 2
    assert {t.track_id for t in tracked} == {t.track_id for t in tracked}
    assert all(t.track_id for t in tracked)
    assert tracker.active_track_count == 2


def test_stationary_object_keeps_track_id_across_frames() -> None:
    tracker = TrackManager(TrackManagerConfig(confirm_after_hits=1))
    cx, cy = 200.0, 200.0
    first = tracker.update([_det(cx, cy)], now_us=0)
    second = tracker.update([_det(cx, cy)], now_us=10_000)
    third = tracker.update([_det(cx, cy)], now_us=20_000)
    assert first[0].track_id == second[0].track_id == third[0].track_id
    assert tracker.active_track_count == 1


def test_moving_object_keeps_track_id_when_velocity_consistent() -> None:
    tracker = TrackManager(TrackManagerConfig(confirm_after_hits=1))
    vx, vy = 100.0, 0.0  # px/s
    dt = 0.05  # 50 ms between frames
    cx = 100.0
    ids: list[str] = []
    for frame in range(5):
        det = _det(cx, 100.0, vx=vx, vy=vy)
        out = tracker.update([det], now_us=int(frame * dt * 1_000_000))
        ids.append(out[0].track_id)
        cx += vx * dt
    assert len(set(ids)) == 1
    assert tracker.active_track_count == 1


def test_distant_detection_spawns_new_track_not_associates() -> None:
    tracker = TrackManager(TrackManagerConfig(confirm_after_hits=1))
    a = tracker.update([_det(100, 100)], now_us=0)
    b = tracker.update([_det(600, 600)], now_us=10_000)
    assert a[0].track_id != b[0].track_id
    assert tracker.active_track_count == 2


def test_miss_count_evicts_after_max_misses() -> None:
    tracker = TrackManager(TrackManagerConfig(max_miss_frames=2))
    tracker.update([_det(100, 100)], now_us=0)
    assert tracker.active_track_count == 1
    for frame in range(1, 5):
        tracker.update([], now_us=frame * 10_000)
    assert tracker.active_track_count == 0


def test_kalman_grade_reduces_confidence_on_inconsistent_velocity() -> None:
    """Same position, but the new velocity is wildly different — should fit
    poorly and drop m_coh below the input."""
    tracker = TrackManager(TrackManagerConfig(confirm_after_hits=1))
    first = tracker.update(
        [_det(200, 200, vx=50.0, vy=0.0, m_coh=0.9)], now_us=0,
    )
    # After dt, the track expects ~ same velocity. Feed a 90° turn here.
    second = tracker.update(
        [_det(200, 200, vx=0.0, vy=500.0, m_coh=0.9)], now_us=20_000,
    )
    assert first[0].track_id == second[0].track_id  # IoU still matches
    assert second[0].m_coh < 0.9


def test_unconfirmed_tracks_do_not_emit() -> None:
    """confirm_after_hits=3: first two sightings produce no output, third
    sighting is the first time we see this object's track_id."""
    tracker = TrackManager(TrackManagerConfig(confirm_after_hits=3))
    first = tracker.update([_det(200, 200)], now_us=0)
    second = tracker.update([_det(200, 200)], now_us=10_000)
    third = tracker.update([_det(200, 200)], now_us=20_000)
    assert first == []
    assert second == []
    assert len(third) == 1


def test_one_off_spike_never_emits() -> None:
    """A single-frame noise spike is dropped — never reaches confirmation."""
    tracker = TrackManager(TrackManagerConfig(confirm_after_hits=3))
    tracker.update([_det(50, 50)], now_us=0)
    out = tracker.update([], now_us=10_000)
    out2 = tracker.update([], now_us=20_000)
    assert out == [] and out2 == []
    assert tracker.active_track_count == 0


def test_consistent_motion_preserves_confidence() -> None:
    tracker = TrackManager(TrackManagerConfig(confirm_after_hits=1))
    first = tracker.update(
        [_det(200, 200, vx=50.0, vy=0.0, m_coh=0.7)], now_us=0,
    )
    # 20 ms later, position advanced by 50*0.02 = 1 px. Same velocity.
    second = tracker.update(
        [_det(201, 200, vx=50.0, vy=0.0, m_coh=0.7)], now_us=20_000,
    )
    assert first[0].track_id == second[0].track_id
    assert second[0].m_coh >= 0.5


def test_reid_disabled_evicts_to_nothing() -> None:
    """With reid_enabled=False the dormant pool stays empty (legacy path)."""
    tracker = TrackManager(TrackManagerConfig(
        confirm_after_hits=1, max_miss_frames=1, reid_enabled=False,
    ))
    tracker.update([_det(100, 100, patch=_patch(0))], now_us=0)
    tracker.update([_det(100, 100, patch=_patch(0))], now_us=10_000)
    tracker.update([], now_us=20_000)
    tracker.update([], now_us=30_000)
    assert tracker.active_track_count == 0
    assert tracker.dormant_track_count == 0


def test_reid_moves_evicted_track_to_dormant_pool() -> None:
    tracker = TrackManager(TrackManagerConfig(
        confirm_after_hits=1, max_miss_frames=1, reid_enabled=True,
    ))
    sig_patch = _patch(7)
    tracker.update([_det(100, 100, patch=sig_patch)], now_us=0)
    tracker.update([_det(100, 100, patch=sig_patch)], now_us=10_000)
    tracker.update([], now_us=20_000)
    tracker.update([], now_us=30_000)
    assert tracker.active_track_count == 0
    assert tracker.dormant_track_count == 1


def test_reid_resurrects_track_id_on_similar_patch() -> None:
    """Drone disappears → reappears with a near-identical event-density
    patch. Same track_id should be re-used."""
    tracker = TrackManager(TrackManagerConfig(
        confirm_after_hits=1, max_miss_frames=1,
        reid_enabled=True, reid_similarity_threshold=0.5,
    ))
    sig_patch = _patch(13)
    first = tracker.update(
        [_det(100, 100, patch=sig_patch)], now_us=0,
    )
    tracker.update([_det(100, 100, patch=sig_patch)], now_us=10_000)
    original_id = first[0].track_id

    tracker.update([], now_us=20_000)
    tracker.update([], now_us=30_000)
    tracker.update([], now_us=40_000)
    assert tracker.dormant_track_count == 1

    out = tracker.update(
        [_det(500, 500, patch=sig_patch)], now_us=50_000,
    )
    assert len(out) == 1
    assert out[0].track_id == original_id
    assert tracker.dormant_track_count == 0


def test_reid_spawns_new_id_on_different_patch() -> None:
    tracker = TrackManager(TrackManagerConfig(
        confirm_after_hits=1, max_miss_frames=1,
        reid_enabled=True, reid_similarity_threshold=0.95,
    ))
    first = tracker.update(
        [_det(100, 100, patch=_patch(1))], now_us=0,
    )
    tracker.update([_det(100, 100, patch=_patch(1))], now_us=10_000)
    tracker.update([], now_us=20_000)
    tracker.update([], now_us=30_000)

    out = tracker.update(
        [_det(500, 500, patch=_patch(99))], now_us=40_000,
    )
    assert out[0].track_id != first[0].track_id


def test_reid_signature_fallback_reattaches_when_iou_fails() -> None:
    """Drone darts far enough that IoU misses, but its appearance matches
    the still-active track. We should attach by signature, not spawn."""
    tracker = TrackManager(TrackManagerConfig(
        confirm_after_hits=1, match_iou_threshold=0.2,
        reid_enabled=True, reid_similarity_threshold=0.5,
    ))
    sig_patch = _patch(21)
    first = tracker.update(
        [_det(100, 100, patch=sig_patch)], now_us=0,
    )
    # Same drone, identical patch, but 500 px away — IoU = 0.
    second = tracker.update(
        [_det(600, 600, patch=sig_patch)], now_us=10_000,
    )
    assert first[0].track_id == second[0].track_id
    assert tracker.active_track_count == 1


def test_reid_signature_fallback_off_when_disabled() -> None:
    tracker = TrackManager(TrackManagerConfig(
        confirm_after_hits=1, match_iou_threshold=0.2,
        reid_enabled=False,
    ))
    first = tracker.update(
        [_det(100, 100, patch=_patch(21))], now_us=0,
    )
    second = tracker.update(
        [_det(600, 600, patch=_patch(21))], now_us=10_000,
    )
    assert first[0].track_id != second[0].track_id
    assert tracker.active_track_count == 2


def test_reid_dormant_timeout_drops_old_tracks() -> None:
    tracker = TrackManager(TrackManagerConfig(
        confirm_after_hits=1, max_miss_frames=1,
        reid_enabled=True, reid_dormant_timeout_s=0.01,
    ))
    sig_patch = _patch(5)
    tracker.update([_det(100, 100, patch=sig_patch)], now_us=0)
    tracker.update([_det(100, 100, patch=sig_patch)], now_us=5_000)
    tracker.update([], now_us=10_000)
    tracker.update([], now_us=15_000)
    assert tracker.dormant_track_count == 1
    tracker.update([], now_us=200_000)
    assert tracker.dormant_track_count == 0
