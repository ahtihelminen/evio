"""Unit tests for the F1.5 spawn-density suppressor."""

from __future__ import annotations

from evio.detection.spawn_density import SpawnDensityConfig, SpawnDensityMap


def _make_map(**overrides: object) -> SpawnDensityMap:
    cfg = SpawnDensityConfig(
        grid_rows=10, grid_cols=10,
        window_s=1.0, max_spawns=3, cooldown_s=1.0, enabled=True,
        **overrides,  # type: ignore[arg-type]
    )
    return SpawnDensityMap(cfg, width=100, height=100)


def test_disabled_never_suppresses() -> None:
    m = SpawnDensityMap(
        SpawnDensityConfig(enabled=False),
        width=100, height=100,
    )
    for _ in range(100):
        m.register_spawn(50.0, 50.0, now_us=0)
    assert m.should_suppress(50.0, 50.0, now_us=0) is False


def test_under_threshold_passes() -> None:
    m = _make_map()
    for i in range(2):
        m.register_spawn(50.0, 50.0, now_us=i * 100_000)
    assert m.should_suppress(50.0, 50.0, now_us=200_000) is False


def test_over_threshold_suppresses() -> None:
    m = _make_map()
    for i in range(3):
        m.register_spawn(50.0, 50.0, now_us=i * 100_000)
    assert m.should_suppress(50.0, 50.0, now_us=300_000) is True


def test_suppression_resets_after_window() -> None:
    m = _make_map()
    for i in range(3):
        m.register_spawn(50.0, 50.0, now_us=i * 100_000)
    # 3 spawns in 200ms is enough — flagged.
    assert m.should_suppress(50.0, 50.0, now_us=300_000) is True
    # After cooldown (1s) + window (1s) + slack, no recent spawns remain.
    later = 300_000 + 2_500_000
    assert m.should_suppress(50.0, 50.0, now_us=later) is False


def test_other_cells_not_suppressed() -> None:
    m = _make_map()
    for i in range(5):
        m.register_spawn(10.0, 10.0, now_us=i * 100_000)
    # Trigger suppression of (1,1).
    assert m.should_suppress(10.0, 10.0, now_us=500_000) is True
    # Different cell remains free.
    assert m.should_suppress(90.0, 90.0, now_us=500_000) is False


def test_old_spawns_drop_out_of_window() -> None:
    m = _make_map()
    # Two spawns now, then jump past the window.
    m.register_spawn(50.0, 50.0, now_us=0)
    m.register_spawn(50.0, 50.0, now_us=100_000)
    # 2 seconds later, those two have rolled off; a fresh 2 spawns isn't
    # enough to trigger.
    m.register_spawn(50.0, 50.0, now_us=2_000_000)
    m.register_spawn(50.0, 50.0, now_us=2_050_000)
    assert m.should_suppress(50.0, 50.0, now_us=2_100_000) is False


def test_total_suppressed_counter_increments() -> None:
    m = _make_map()
    for i in range(3):
        m.register_spawn(50.0, 50.0, now_us=i * 100_000)
    assert m.total_suppressed == 0
    m.should_suppress(50.0, 50.0, now_us=300_000)
    m.should_suppress(50.0, 50.0, now_us=400_000)
    assert m.total_suppressed == 2


def test_snapshot_shape() -> None:
    m = _make_map()
    snap = m.snapshot(now_us=0)
    assert len(snap["suppressed"]) == 10
    assert len(snap["suppressed"][0]) == 10
    assert len(snap["spawn_counts"]) == 10
    assert snap["total_suppressed"] == [0]
