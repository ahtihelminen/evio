"""Tracker-layer probe for hand-wake ghost suppression.

The live engine builds ``TrackManager(TrackManagerConfig(reid_enabled=True))``
— every F1/F1.5 anti-clutter gate OFF. This probe feeds three streams
through the tracker and measures, per gate configuration, how many windows
each stream emits a confirmed box:

* ghost  — co-located, jittery, near-zero net displacement, 10 windows
* drone  — straight constant-velocity crossing (fast), 40 windows
* faint  — straight but SLOW (8 px/s), 40 windows — evio's #1 priority

A good fix drives ghost emits -> ~0 while keeping drone AND faint alive.
"""

from __future__ import annotations

import numpy as np

from evio.detection.tracker import TrackManager, TrackManagerConfig
from evio.detection.types import Detection

_WINDOW_US = 50_000  # matches the field test's --window-ms 50
_N = 40


def _det(
    cx: float, cy: float, vx: float, vy: float, *, half: int = 16,
) -> Detection:
    return Detection(
        x0=int(cx - half), y0=int(cy - half),
        x1=int(cx + half), y1=int(cy + half),
        vx_px_s=vx, vy_px_s=vy, m_coh=80.0, best_psr=20.0,
        event_count=50, bg_class=0,
        row_start=0, row_end=1, col_start=0, col_end=1,
        cx=float(cx), cy=float(cy),
    )


def _mid_x(det: Detection) -> float:
    return (det.x0 + det.x1) / 2.0


def run(cfg: TrackManagerConfig, *, seed: int = 0) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    tm = TrackManager(cfg)
    emits = {"ghost": 0, "drone": 0, "faint": 0}
    for w in range(_N):
        t = w * _WINDOW_US / 1e6
        dets: list[Detection] = []
        if w < 10:  # ghost: jittery, in place near x=300
            jx, jy = rng.uniform(-3.0, 3.0, 2)
            vx, vy = rng.uniform(-12.0, 12.0, 2)
            dets.append(_det(300 + jx, 200 + jy, vx, vy))
        # fast drone near y=360, crossing right at 150 px/s, x in [600, 900]
        dets.append(_det(600 + 150.0 * t, 360.0, 150.0, 0.0))
        # faint slow drone near y=520, crossing right at 8 px/s
        dets.append(_det(950 + 8.0 * t, 520.0, 8.0, 0.0))
        out = tm.update(dets, now_us=(w + 1) * _WINDOW_US)
        for d in out:
            mid = _mid_x(d)
            if mid < 500:
                emits["ghost"] += 1
            elif mid < 920:
                emits["drone"] += 1
            else:
                emits["faint"] += 1
    return emits


def test_gate_configurations() -> None:
    base = {"reid_enabled": True}
    configs = {
        "live (current)": TrackManagerConfig(**base),
        "+persistence": TrackManagerConfig(
            **base, persistence_gate_enabled=True,
        ),
        "+spawn_density": TrackManagerConfig(
            **base, spawn_density_enabled=True,
        ),
        "+velocity_coh": TrackManagerConfig(
            **base, velocity_coherence_enabled=True,
        ),
        "+vel(floor=5)": TrackManagerConfig(
            **base, velocity_coherence_enabled=True,
            velocity_min_confirm_px_s=5.0,
        ),
        # Drop the speed floor so a slow faraway drone survives, and lean on
        # straightness/CV to reject the jittery in-place ghost instead.
        "+vel(floor=1)": TrackManagerConfig(
            **base, velocity_coherence_enabled=True,
            velocity_min_confirm_px_s=1.0,
        ),
        "+vel(floor=1,straight=0.5)": TrackManagerConfig(
            **base, velocity_coherence_enabled=True,
            velocity_min_confirm_px_s=1.0,
            velocity_straightness_min=0.5,
        ),
        "+vel(floor=1,straight=0.7)": TrackManagerConfig(
            **base, velocity_coherence_enabled=True,
            velocity_min_confirm_px_s=1.0,
            velocity_straightness_min=0.7,
        ),
        "+vel+persist+spawn": TrackManagerConfig(
            **base, velocity_coherence_enabled=True,
            velocity_min_confirm_px_s=5.0,
            persistence_gate_enabled=True, spawn_density_enabled=True,
        ),
    }
    print(f"\n{'config':<22} {'ghost':>6} {'drone':>6} {'faint':>6}")
    for name, cfg in configs.items():
        e = run(cfg)
        print(f"{name:<22} {e['ghost']:>6} {e['drone']:>6} {e['faint']:>6}")
