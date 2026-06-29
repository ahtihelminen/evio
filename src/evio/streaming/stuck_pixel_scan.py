"""Driver-side stuck-pixel scan + DEM masking (the P0 manual stopgap).

Pulls a static-scene capture off one IMX636, counts the per-pixel ON/OFF
rate, and nominates the fixed hot oscillators that flood the field cameras.
A pixel is a candidate iff, over the capture, it is (1) at/above
``STUCK_MASK_EPS``, (2) spatially isolated (every 8-neighbour an order of
magnitude quieter), and (3) single-polarity (>= ``STUCK_POLARITY_FRACTION``
one polarity) -- the same Path-A gates the live autotuner uses, run
conservatively here so an operator vets each coordinate before any hardware
mask is applied.

Every candidate carries its raw ``event_count`` and the scan's measured
``elapsed_s``; the true rate is ``event_count / elapsed_s``. Recording both
(not a bare "rate") makes the figure self-verifying and lets the catalog
compare a 30 s scan against a 5 min one. ``write_inventory`` emits a rich,
self-describing record (provenance + scene + detector params + active masks)
that ``event-camera-statistics`` ingests directly; a CLEAN camera still writes
an empty-defect record (the load-bearing healthy datapoint).

DEM is on-sensor (upstream of MIPI/FX3/USB), 64 per-pixel slots, live (no
shadow trigger). The ``imx636`` driver is imported lazily so the module stays
import-safe on a dev box without the camera HAL. ``diff`` is NEVER written.
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

# Scan thresholds -- mirror evio.streaming.bias.constants, kept LOCAL so this
# driver-side diagnostic stands alone: it imports nothing from the autotuner,
# so it drops onto a rig as just this file + the CLI wrapper.
STUCK_MASK_EPS = 2_000.0  # per-pixel rate floor (ev/s) to be a candidate
STUCK_POLARITY_FRACTION = 0.95  # >= this share single-polarity (Path A)
DEM_MAX_SLOTS = 64  # = imx636.PIXEL_MASK_SLOTS
ISOLATION_FACTOR = 10.0  # an isolated pixel's neighbours read >= this lower
# AFK notch around the mains harmonic (31-55 Hz is valid; NOT 50-120 Hz, which
# eats low-RPM blade-pass combs).
AFK_NOTCH_LOW_HZ = 31
AFK_NOTCH_HIGH_HZ = 55


@dataclass(frozen=True)
class StuckCandidate:
    """One nominated pixel; ``polarity`` is 1 (ON-dominant) or 0 (OFF)."""

    x: int
    y: int
    event_count: int  # raw events over the scan (ON+OFF for this pixel)
    rate_eps: float  # event_count / elapsed_s -- the true per-second rate
    polarity_fraction: float  # share of events that are the dominant polarity
    polarity: int


@dataclass(frozen=True)
class ScanResult:
    """A whole-camera scan: nominated pixels + the measured window + load."""

    candidates: list[StuckCandidate]
    elapsed_s: float
    total_events: int
    total_eps: float
    complete: bool  # False if a USB error cut the capture short


def scan(
    cam: Camera, duration_s: float, *, poll_timeout_ms: int = 2,
) -> ScanResult:
    """Count per-pixel rates over a static-scene capture; return candidates."""
    width, height = cam.width(), cam.height()
    n_px = width * height
    on = np.zeros(n_px, dtype=np.int64)
    off = np.zeros(n_px, dtype=np.int64)
    cam.start()
    # Drain any buffered backlog accumulated before the timed window opens, so
    # a first-poll dump can't inflate the count against a short elapsed.
    drain_until = time.monotonic() + 0.2
    while time.monotonic() < drain_until:
        cam.poll_events(poll_timeout_ms)
    complete = True
    start = time.monotonic()
    while time.monotonic() - start < duration_s:
        status, cds, _ = cam.poll_events(poll_timeout_ms)
        if status < 0:
            complete = False
            break
        if cds.size == 0:
            continue
        flat = cds["y"].astype(np.intp) * width + cds["x"].astype(np.intp)
        is_on = cds["p"] > 0
        on += np.bincount(flat[is_on], minlength=n_px)
        off += np.bincount(flat[~is_on], minlength=n_px)
    elapsed = max(time.monotonic() - start, 1e-6)
    total = int(on.sum() + off.sum())
    return ScanResult(
        candidates=_candidates(on, off, elapsed, width, height),
        elapsed_s=elapsed,
        total_events=total,
        total_eps=total / elapsed,
        complete=complete,
    )


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
        total = on_count + off_count
        fraction = max(on_count, off_count) / max(total, 1)
        if fraction < STUCK_POLARITY_FRACTION:
            continue  # balanced: Path A does not mask it (operator review)
        out.append(StuckCandidate(
            x=x, y=y, event_count=total, rate_eps=float(rate[y, x]),
            polarity_fraction=fraction,
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


def active_masks(cam: Camera) -> list[dict]:
    """DEM masks already programmed when the scan opened (for provenance)."""
    try:
        return [{"x": m.x, "y": m.y} for _, m in cam.get_pixel_masks()]
    except (RuntimeError, AttributeError):
        return []


def build_inventory(
    serial: str, result: ScanResult, sensor: dict, capture: dict,
) -> dict:
    """The rich raw inventory the event-camera-statistics catalog ingests.

    ``capture`` carries the operator-supplied / rig-only context (scene, host,
    producer git_sha, operator, batch_id, detector_params, active_masks).
    """
    return {
        "inventory_version": 1,
        "serial": serial,
        "captured_at": capture["captured_at"],
        "host": capture["host"],
        "producer": capture["producer"],
        "sensor": sensor,
        "scene": capture["scene"],
        "scene_controlled": capture["scene_controlled"],
        "scan_complete": result.complete,
        "duration_s": capture["duration_s"],
        "elapsed_s": round(result.elapsed_s, 3),
        "operator": capture.get("operator"),
        "batch_id": capture.get("batch_id"),
        "detector_params": {
            "stuck_mask_eps": STUCK_MASK_EPS,
            "isolation_factor": ISOLATION_FACTOR,
            "polarity_fraction": STUCK_POLARITY_FRACTION,
            "dem_max_slots": DEM_MAX_SLOTS,
        },
        "active_masks_at_scan": capture.get("active_masks", []),
        "metrics": {
            "usb_mbps": capture.get("usb_mbps"),
            "total_eps": round(result.total_eps),
            "erc": capture.get("erc"),
            "bandwidth_test_ran": capture.get("bandwidth_test_ran", False),
        },
        "defects": [
            {
                "x": c.x, "y": c.y, "event_count": c.event_count,
                "polarity_fraction": round(c.polarity_fraction, 4),
                "dominant_polarity": "ON" if c.polarity else "OFF",
            }
            for c in result.candidates
        ],
    }


def write_inventory(path: Path, inventory: dict) -> None:
    """Persist the per-camera raw inventory as JSON."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory, indent=2) + "\n")


def write_applied_masks(
    path: Path, serial: str, applied: list[StuckCandidate], applied_at: str,
) -> None:
    """Persist the operator-vetted DEM masks so the live run re-applies them.

    The on-sensor DEM is volatile (lost when the scan closes the camera), so
    ``--apply`` alone never reaches ``run_ops``; ``Imx636CameraSource`` reloads
    THIS manifest at open. Distinct from the raw inventory of candidates.
    """
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "serial": serial,
        "applied_at": applied_at,
        "masks": [
            {"slot": slot, "x": c.x, "y": c.y,
             "dominant_polarity": "ON" if c.polarity else "OFF",
             "rate_eps": round(c.rate_eps)}
            for slot, c in enumerate(applied)
        ],
    }, indent=2) + "\n")


__all__ = [
    "ScanResult",
    "StuckCandidate",
    "active_masks",
    "apply_masks",
    "bandwidth_test",
    "build_inventory",
    "scan",
    "set_afk_notch",
    "write_applied_masks",
    "write_inventory",
]
