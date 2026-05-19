"""F1 persistence-gate + Mahalanobis-tracker behaviour tests."""

from __future__ import annotations

import numpy as np

from evio.detection.persistence import PersistenceGateConfig, PersistenceMap
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
) -> Detection:
    return Detection(
        x0=int(cx - half), y0=int(cy - half),
        x1=int(cx + half), y1=int(cy + half),
        vx_px_s=vx, vy_px_s=vy,
        m_coh=m_coh, best_psr=1.0, event_count=100,
        bg_class=0, row_start=0, row_end=1, col_start=0, col_end=1,
        z_score=3.0, camera_id="cam",
    )


def test_persistence_map_disabled_passes_everything_through() -> None:
    pm = PersistenceMap(
        PersistenceGateConfig(enabled=False),
        width=1280, height=720,
    )
    dets = [_det(100, 100, vx=0.0, vy=0.0)]
    for _ in range(60):
        pm.observe([((d.x0 + d.x1) / 2.0, (d.y0 + d.y1) / 2.0, 0.0)
                    for d in dets])
    assert pm.filter(dets) == dets


def test_persistence_map_flags_stationary_repeat_offender() -> None:
    pm = PersistenceMap(
        PersistenceGateConfig(
            grid_rows=64, grid_cols=64,
            window_history=100, hit_rate_threshold=0.30,
            max_velocity_px_s=5.0, warmup_windows=10, enabled=True,
        ),
        width=1280, height=720,
    )
    cx, cy = 100.0, 100.0
    for _ in range(60):
        pm.observe([(cx, cy, 0.0)])
    survivors = pm.filter([_det(cx, cy, vx=0.0, vy=0.0)])
    assert survivors == []


def test_persistence_map_passes_fast_movers_through_uncluttered_cells() -> None:
    """A high-speed trajectory through a cell that's never been seen
    stationary should pass — the gate must not blanket-flag every cell."""
    pm = PersistenceMap(
        PersistenceGateConfig(
            window_history=100, hit_rate_threshold=0.30,
            max_velocity_px_s=5.0, warmup_windows=10, enabled=True,
        ),
        width=1280, height=720,
    )
    # 60 windows of stationary clutter at (100, 100). A different cell
    # (600, 600) only ever saw the fast drone — that cell shouldn't be
    # flagged.
    for _ in range(60):
        pm.observe([(100.0, 100.0, 0.0)])
    pm.observe([(600.0, 600.0, 200.0)])
    fast_det = _det(600, 600, vx=200.0, vy=0.0)
    assert pm.filter([fast_det]) == [fast_det]


def test_tracker_persistence_gate_drops_static_clutter() -> None:
    tracker = TrackManager(TrackManagerConfig(
        persistence_gate_enabled=True,
        persistence_warmup_windows=5,
        persistence_hit_rate_threshold=0.20,
        persistence_max_velocity_px_s=5.0,
        confirm_after_hits=1,
    ))
    for frame in range(40):
        tracker.update(
            [_det(100, 100, vx=0.0, vy=0.0)],
            now_us=frame * 10_000,
        )
    # By now the cell at (100, 100) has fired in every window with zero
    # speed — it's clutter. A fresh detection there should be dropped.
    out = tracker.update(
        [_det(100, 100, vx=0.0, vy=0.0)],
        now_us=41 * 10_000,
    )
    assert out == []
    assert tracker.persistence_drops >= 1


def test_mahalanobis_matching_keeps_track_for_consistent_motion() -> None:
    """Constant-velocity drone should hold its track_id under the
    Mahalanobis-first gate just like under IoU matching."""
    tracker = TrackManager(TrackManagerConfig(
        mahalanobis_matching_enabled=True,
        mahalanobis_chi2_cutoff=20.0,
        confirm_after_hits=1,
    ))
    vx = 100.0
    dt = 0.05
    cx = 100.0
    ids: list[str] = []
    for frame in range(5):
        det = _det(cx, 100.0, vx=vx, vy=0.0)
        out = tracker.update([det], now_us=int(frame * dt * 1_000_000))
        ids.append(out[0].track_id)
        cx += vx * dt
    assert len(set(ids)) == 1


def test_mahalanobis_matching_rejects_distant_unrelated_detection() -> None:
    """A detection well outside any track's covariance ellipse should
    spawn a fresh track rather than steal an existing one."""
    tracker = TrackManager(TrackManagerConfig(
        mahalanobis_matching_enabled=True,
        mahalanobis_chi2_cutoff=9.49,
        confirm_after_hits=1,
    ))
    first = tracker.update([_det(100, 100)], now_us=0)
    second = tracker.update([_det(600, 600)], now_us=10_000)
    assert first[0].track_id != second[0].track_id


def test_velocity_coherence_blocks_static_confirmation() -> None:
    """A stationary candidate never satisfies v_min_confirm — never emits."""
    tracker = TrackManager(TrackManagerConfig(
        velocity_coherence_enabled=True,
        velocity_min_confirm_px_s=25.0,
        confirm_after_hits=3,
    ))
    for frame in range(10):
        out = tracker.update(
            [_det(100, 100, vx=0.0, vy=0.0)],
            now_us=frame * 10_000,
        )
        assert out == []


def test_velocity_coherence_passes_straight_fast_track() -> None:
    """A coherent constant-velocity track satisfies the rule and emits."""
    tracker = TrackManager(TrackManagerConfig(
        velocity_coherence_enabled=True,
        velocity_min_confirm_px_s=25.0,
        velocity_straightness_min=0.3,
        velocity_cv_max=0.6,
        confirm_after_hits=3,
    ))
    vx = 100.0
    dt = 0.01
    cx = 100.0
    emitted = False
    for frame in range(20):
        det = _det(cx, 100.0, vx=vx, vy=0.0)
        out = tracker.update([det], now_us=int(frame * dt * 1_000_000))
        if out:
            emitted = True
        cx += vx * dt
    assert emitted is True


def test_existing_tests_unaffected_when_f1_flags_off() -> None:
    """Sanity check: leaving F1 flags default keeps the SORT path intact."""
    tracker = TrackManager(TrackManagerConfig(confirm_after_hits=1))
    out_a = tracker.update([_det(100, 100)], now_us=0)
    out_b = tracker.update([_det(100, 100)], now_us=10_000)
    assert out_a[0].track_id == out_b[0].track_id


def test_persistence_map_snapshot_shape() -> None:
    pm = PersistenceMap(
        PersistenceGateConfig(enabled=True, grid_rows=8, grid_cols=8),
        width=1280, height=720,
    )
    for _ in range(3):
        pm.observe([(100.0, 100.0, 0.0)])
    snap = pm.snapshot()
    assert np.asarray(snap["hit_rate"]).shape == (8, 8)
    assert np.asarray(snap["mean_speed"]).shape == (8, 8)
    assert snap["windows_seen"] == [3]
