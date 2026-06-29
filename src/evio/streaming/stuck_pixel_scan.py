"""Driver-side stuck-pixel scan + DEM masking (the P0 manual stopgap).

Pulls a static-scene capture off one IMX636, counts the per-pixel ON/OFF
rate, and nominates the fixed hot oscillators that flood the field cameras
(1-10 pixels at 54-400 kHz = 99.8-99.9 % of all events). A pixel is a
candidate iff, over the capture, it is (1) at/above ``STUCK_MASK_EPS``,
(2) spatially isolated (every 8-neighbour an order of magnitude quieter),
and (3) single-polarity (>= ``STUCK_POLARITY_FRACTION`` one polarity) — the
same Path-A gates the live autotuner uses, run conservatively here so an
operator vets each coordinate before any hardware mask is applied.

DEM is on-sensor (upstream of MIPI/FX3/USB), 64 per-pixel slots, live (no
shadow trigger). Applied masks are persisted into the camera's own config
via ``save_config`` so the driver re-applies them on the next open — this is
the warm-seed for the live autotuner. ``diff`` is NEVER written (hard rule).
The ``imx636`` driver is imported lazily so the module stays import-safe on a
dev box without the camera HAL.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    import imx636

    Camera = imx636.IMX636

# Scan thresholds — mirror evio.streaming.bias.constants, kept LOCAL so this
# driver-side diagnostic stands alone: it imports nothing from the autotuner,
# so it drops onto a rig as just this file + the CLI wrapper.
STUCK_MASK_EPS = 2_000.0  # per-pixel rate floor (ev/s) to be a candidate
STUCK_POLARITY_FRACTION = 0.95  # >= this share single-polarity (Path A)
DEM_MAX_SLOTS = 64  # = imx636.PIXEL_MASK_SLOTS
# An isolated stuck pixel's neighbours read at least this many times lower.
ISOLATION_FACTOR = 10.0
# AFK notch around the mains harmonic (31-55 Hz is valid: periods 252 != 142,
# it will not "collapse"). NOT 50-120 Hz, which eats low-RPM blade-pass combs.
AFK_NOTCH_LOW_HZ = 31
AFK_NOTCH_HIGH_HZ = 55


@dataclass(frozen=True)
class StuckCandidate:
    """One nominated pixel; ``polarity`` is 1 (ON-dominant) or 0 (OFF)."""

    x: int
    y: int
    rate_eps: float
    on_fraction: float
    polarity: int


def scan(
    cam: Camera, duration_s: float, *, poll_timeout_ms: int = 2,
) -> list[StuckCandidate]:
    """Count per-pixel rates over a static-scene capture; return candidates."""
    width, height = cam.width(), cam.height()
    n_px = width * height
    on = np.zeros(n_px, dtype=np.int64)
    off = np.zeros(n_px, dtype=np.int64)
    cam.start()
    start = time.monotonic()
    while time.monotonic() - start < duration_s:
        status, cds, _ = cam.poll_events(poll_timeout_ms)
        if status < 0:
            msg = "USB error during stuck-pixel scan"
            raise RuntimeError(msg)
        if cds.size == 0:
            continue
        flat = cds["y"].astype(np.intp) * width + cds["x"].astype(np.intp)
        is_on = cds["p"] > 0
        on += np.bincount(flat[is_on], minlength=n_px)
        off += np.bincount(flat[~is_on], minlength=n_px)
    elapsed = max(time.monotonic() - start, 1e-6)
    return _candidates(on, off, elapsed, width, height)


def _candidates(
    on: np.ndarray, off: np.ndarray, elapsed: float, width: int, height: int,
) -> list[StuckCandidate]:
    on2d = on.reshape((height, width))
    off2d = off.reshape((height, width))
    rate = (on2d + off2d) / elapsed
    floor = STUCK_MASK_EPS / ISOLATION_FACTOR
    out: list[StuckCandidate] = []
    ys, xs = np.nonzero(rate >= STUCK_MASK_EPS)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        block = rate[max(0, y - 1):y + 2, max(0, x - 1):x + 2].copy()
        block[min(y, 1), min(x, 1)] = 0.0  # exclude self
        if float(block.max()) >= floor:
            continue  # has an active neighbour: not an isolated oscillator
        on_count, off_count = int(on2d[y, x]), int(off2d[y, x])
        fraction = max(on_count, off_count) / max(on_count + off_count, 1)
        if fraction < STUCK_POLARITY_FRACTION:
            continue  # balanced: Path A does not mask it (operator review)
        out.append(StuckCandidate(
            x=x, y=y, rate_eps=float(rate[y, x]), on_fraction=fraction,
            polarity=1 if on_count >= off_count else 0,
        ))
    out.sort(key=lambda c: c.rate_eps, reverse=True)
    return out


def apply_masks(
    cam: Camera, candidates: list[StuckCandidate],
) -> list[StuckCandidate]:
    """Program the top-rate candidates into DEM slots; return those applied."""
    import imx636
    applied = candidates[:DEM_MAX_SLOTS]
    for slot, candidate in enumerate(applied):
        cam.set_pixel_mask(
            slot, imx636.PixelMask(candidate.x, candidate.y, True),
        )
    return applied


def set_afk_notch(
    cam: Camera,
    low_hz: int = AFK_NOTCH_LOW_HZ,
    high_hz: int = AFK_NOTCH_HIGH_HZ,
) -> bool:
    """Enable the mains band-stop notch; False if the driver rejects it."""
    import imx636
    cfg = imx636.AntiFlickerConfig()
    cfg.enabled = True
    cfg.band_stop = True
    cfg.low_freq = low_hz
    cfg.high_freq = high_hz
    try:
        cam.set_antiflicker(cfg)
    except (RuntimeError, ValueError):
        return False
    return True


def bandwidth_test(
    cam: Camera, candidate: StuckCandidate, *,
    windows: int = 20, settle_s: float = 0.5,
) -> dict:
    """usb_mbps before vs after masking the highest-rate stuck pixel.

    Decisive only on a STATIC/DARK scene: ``usb_mbps`` is a ~0.5 s
    whole-stream EMA, so a null delta against a live scene is inconclusive.
    Confirms three things: the slot reads back, the decoded stream loses the
    pixel, and the wire byte-rate drops.
    """
    import imx636

    def avg_mbps() -> float:
        cam.reset_diag()
        samples = []
        for _ in range(windows):
            time.sleep(settle_s)
            samples.append(cam.diag_stats()["usb_mbps"])
        return float(np.mean(samples))

    before = avg_mbps()
    cam.set_pixel_mask(0, imx636.PixelMask(candidate.x, candidate.y, True))
    readback = cam.get_pixel_masks()
    after = avg_mbps()
    return {
        "pixel": (candidate.x, candidate.y),
        "before_mbps": before,
        "after_mbps": after,
        "delta_mbps": before - after,
        "slot_reads_back": any(
            m.x == candidate.x and m.y == candidate.y for _, m in readback
        ),
    }


def write_inventory(
    path: Path, serial: str, duration_s: float,
    candidates: list[StuckCandidate],
) -> None:
    """Persist the per-camera ``(serial, x, y, ...)`` inventory as JSON."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "serial": serial,
        "duration_s": duration_s,
        "candidates": [
            {
                "slot": slot, "x": c.x, "y": c.y,
                "rate_eps": c.rate_eps, "on_fraction": c.on_fraction,
                "polarity": c.polarity,
            }
            for slot, c in enumerate(candidates[:DEM_MAX_SLOTS])
        ],
    }, indent=2))


__all__ = [
    "StuckCandidate",
    "apply_masks",
    "bandwidth_test",
    "scan",
    "set_afk_notch",
    "write_inventory",
]
