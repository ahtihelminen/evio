"""Behaviour of the stuck-pixel scan, DEM masking and its persistence (SN-603).

Runs with no camera: `FakeImx636Camera` serves scripted per-pixel counts and
moves a fake clock one poll at a time, and `fake_imx636_module` stands in for
the lazily imported driver bindings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fakes.clock import FakeClock
from fakes.imx636_camera import FakeImx636Camera, fake_imx636_module

from evio.streaming import stuck_pixel_scan
from evio.streaming.stuck_pixel_scan import (
    DEM_MAX_SLOTS,
    ScanResult,
    StuckCandidate,
    apply_masks,
    boot_candidates,
    build_inventory,
    scan,
    set_afk_notch,
    write_applied_masks,
    write_inventory,
)

pytestmark = pytest.mark.t1

WIDTH, HEIGHT = 8, 6
POLL_S = 0.01  # one poll per 10 ms: one event per poll is a 100 ev/s rest


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock(now=100.0)
    monkeypatch.setattr(stuck_pixel_scan, "time", clock)
    return clock


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "imx636", fake_imx636_module())


def _quiet_scene() -> tuple[np.ndarray, np.ndarray]:
    """One event per pixel per poll, polarity in a checkerboard."""
    checker = (np.indices((HEIGHT, WIDTH)).sum(axis=0) % 2).astype(np.int64)
    return checker, 1 - checker


def _camera(
    clock: FakeClock, on: np.ndarray, off: np.ndarray,
) -> FakeImx636Camera:
    return FakeImx636Camera(on, off, clock=clock, poll_s=POLL_S)


def _ranked(n: int) -> list[StuckCandidate]:
    """`n` ON-stuck pixels over a quiet floor, ranked by `boot_candidates`."""
    on = np.full(64 * 64, 10, dtype=np.int64)
    off = np.zeros_like(on)
    on[:n] = 3_000 + 100 * np.arange(n)
    candidates, _ = boot_candidates(on, off, 1.0, 64, max_masks=4096)
    return candidates


def test_scan_nominates_stuck_pixels_with_dominant_polarity(
    clock: FakeClock,
) -> None:
    on, off = _quiet_scene()
    on[1, 2], off[1, 2] = 50, 0  # 5 kev/s, all ON
    on[4, 5], off[4, 5] = 0, 40  # 4 kev/s, all OFF

    result = scan(_camera(clock, on, off), 1.0)

    assert result.complete
    assert [(c.x, c.y, c.polarity) for c in result.candidates] == [
        (2, 1, 1), (5, 4, 0),
    ]
    assert result.candidates[0].rate_eps == pytest.approx(5_000, rel=0.05)


def test_scan_nominates_nothing_on_a_quiet_scene(clock: FakeClock) -> None:
    on, off = _quiet_scene()

    result = scan(_camera(clock, on, off), 1.0)

    assert result.total_events > 0
    assert result.candidates == []


def test_boot_candidates_ranks_by_rate_and_respects_max_masks() -> None:
    width = 10
    on = np.full(width * width, 10, dtype=np.int64)
    off = np.zeros_like(on)
    for index, count in {3: 5_000, 17: 9_000, 42: 3_000, 88: 7_000}.items():
        on[index] = count

    every, _ = boot_candidates(on, off, 1.0, width, max_masks=DEM_MAX_SLOTS)
    top, _ = boot_candidates(on, off, 1.0, width, max_masks=3)

    assert [c.rate_eps for c in every] == [9_000, 7_000, 5_000, 3_000]
    assert [(c.x, c.y) for c in top] == [(7, 1), (8, 8), (3, 0)]


def test_apply_masks_programs_top_rate_candidates_only(driver: None) -> None:
    ranked = _ranked(DEM_MAX_SLOTS + 6)
    cam = FakeImx636Camera.idle(64, 64)

    applied = apply_masks(cam, ranked)

    assert applied == ranked[:DEM_MAX_SLOTS]
    assert sorted(cam.pixel_masks) == list(range(DEM_MAX_SLOTS))
    programmed = [cam.pixel_masks[slot] for slot in range(DEM_MAX_SLOTS)]
    assert [(m.x, m.y, m.enabled) for m in programmed] == [
        (c.x, c.y, True) for c in applied
    ]


def test_set_afk_notch_reports_driver_acceptance(driver: None) -> None:
    cam = FakeImx636Camera.idle(WIDTH, HEIGHT)

    assert set_afk_notch(cam) is True
    assert cam.antiflicker.enabled and cam.antiflicker.band_stop
    assert (cam.antiflicker.low_freq, cam.antiflicker.high_freq) == (31, 55)


def test_set_afk_notch_reports_driver_rejection(driver: None) -> None:
    cam = FakeImx636Camera.idle(WIDTH, HEIGHT, rejects_antiflicker=True)

    assert set_afk_notch(cam) is False
    assert cam.antiflicker is None


def test_write_applied_masks_roundtrips_schema_v3(tmp_path: Path) -> None:
    applied = _ranked(5)
    path = tmp_path / "dem_masks" / "SN123.json"

    write_applied_masks(path, "SN123", applied, "2026-10-07T00:00:00Z")
    doc = json.loads(path.read_text())

    assert doc["schema_version"] == 3
    assert doc["serial"] == "SN123"
    assert [
        (m["slot"], m["x"], m["y"], m["dominant_polarity"], m["rate_eps"])
        for m in doc["masks"]
    ] == [
        (slot, c.x, c.y, "ON", round(c.rate_eps))
        for slot, c in enumerate(applied)
    ]
    assert {m["source"] for m in doc["masks"]} == {"factory"}
    assert doc["soft_masks"] == []
    assert not path.with_suffix(".json.tmp").exists()


def test_inventory_is_valid_json_with_serial_and_candidates(
    tmp_path: Path,
) -> None:
    candidates = _ranked(3)
    result = ScanResult(
        candidates=candidates, elapsed_s=30.0, total_events=90_000,
        total_eps=3_000.0, complete=True,
    )
    capture = {
        "captured_at": "2026-10-07T00:00:00Z", "host": "laptop",
        "producer": {"git_sha": "abc123"}, "scene": "lens cap",
        "scene_controlled": True, "duration_s": 30.0,
    }
    path = tmp_path / "inventory" / "SN123.json"

    write_inventory(path, build_inventory("SN123", result, {}, capture))
    doc = json.loads(path.read_text())

    assert doc["serial"] == "SN123"
    assert [(d["x"], d["y"], d["event_count"]) for d in doc["defects"]] == [
        (c.x, c.y, c.event_count) for c in candidates
    ]
