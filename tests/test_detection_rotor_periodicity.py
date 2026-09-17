"""Rotor-periodicity estimator: binning, comb recovery, gates, lifecycle.

The synthetic rotor reproduces the polarity-cancellation physics: ON events
are rate-modulated by ``1 + cos(2π·bpf·t)``, OFF anti-phase, so ON+OFF is
flat and only the ON - OFF series carries the comb (the FRED 22x11 px
finding this module exists to exploit).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evio.core.events import EventsBatch

if TYPE_CHECKING:
    from collections.abc import Iterator
import pytest

from evio.detection.rotor_periodicity import (
    RotorPeriodicityEstimator,
    bin_events_polarity,
)
from evio.detection.types import Detection

pytestmark = pytest.mark.t1

_BBOX = (100, 100, 130, 130)


def _det(track_id: str = "t1") -> Detection:
    x0, y0, x1, y1 = _BBOX
    return Detection(
        x0=x0, y0=y0, x1=x1, y1=y1, vx_px_s=0.0, vy_px_s=0.0,
        m_coh=0.5, best_psr=3.0, event_count=100, bg_class=0,
        row_start=0, row_end=1, col_start=0, col_end=1,
        track_id=track_id,
    )


def _thin_poisson(
    rng: np.random.Generator, t_us: np.ndarray, rate_mod: np.ndarray,
) -> np.ndarray:
    """Thin a homogeneous candidate stream by the [0, 1] rate modulation."""
    return t_us[rng.random(t_us.size) < rate_mod]


def _synth_rotor_events(
    *,
    bpf_hz: float,
    duration_s: float,
    mod_rate_hz: float = 40_000.0,
    bg_rate_hz: float = 5_000.0,
    seed: int = 0,
    t0_us: int = 0,
) -> EventsBatch:
    """ON/OFF anti-phase rotor modulation + flat background, inside _BBOX."""
    rng = np.random.default_rng(seed)

    def poisson_train(rate_hz: float) -> np.ndarray:
        n = rng.poisson(rate_hz * duration_s)
        return np.sort(rng.uniform(0, duration_s * 1e6, n)).astype(np.int64)

    cand_on = poisson_train(mod_rate_hz)
    cand_off = poisson_train(mod_rate_hz)
    phase_on = 0.5 * (1.0 + np.cos(
        2 * np.pi * bpf_hz * cand_on / 1e6))
    phase_off = 0.5 * (1.0 + np.cos(
        2 * np.pi * bpf_hz * cand_off / 1e6 + np.pi))
    on = _thin_poisson(rng, cand_on, phase_on)
    off = _thin_poisson(rng, cand_off, phase_off)
    bg_on = poisson_train(bg_rate_hz)
    bg_off = poisson_train(bg_rate_hz)
    ts = np.concatenate([on, bg_on, off, bg_off])
    pol = np.concatenate([
        np.ones(on.size + bg_on.size, dtype=np.int8),
        np.zeros(off.size + bg_off.size, dtype=np.int8),
    ])
    order = np.argsort(ts, kind="stable")
    ts = ts[order] + t0_us
    pol = pol[order]
    x0, y0, x1, y1 = _BBOX
    xs = np.random.default_rng(seed + 1).integers(
        x0, x1 + 1, ts.size).astype(np.int16)
    ys = np.random.default_rng(seed + 2).integers(
        y0, y1 + 1, ts.size).astype(np.int16)
    return EventsBatch(
        timestamps=ts, x_coords=xs, y_coords=ys, polarities=pol,
    )


def _windows(
    batch: EventsBatch, window_us: int = 10_000,
) -> Iterator[EventsBatch]:
    t0 = int(batch.timestamps[0])
    t1 = int(batch.timestamps[-1])
    for lo in range(t0, t1, window_us):
        m = (batch.timestamps >= lo) & (batch.timestamps < lo + window_us)
        if m.any():
            yield EventsBatch(
                timestamps=batch.timestamps[m], x_coords=batch.x_coords[m],
                y_coords=batch.y_coords[m], polarities=batch.polarities[m],
            )


def _run(
    est: RotorPeriodicityEstimator, batch: EventsBatch,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for window in _windows(batch):
        out = est.update("cam", [_det()], window)
    return out


def test_binning_exact() -> None:
    ts = np.array([0, 400, 600, 1_400, 1_600], dtype=np.int64)
    pol = np.array([1, 0, 1, 1, 0], dtype=np.int8)
    on, off = bin_events_polarity(
        ts, pol, start_bin=0, n_bins=4, bin_us=500,
    )
    assert on.tolist() == [1, 1, 1, 0]
    assert off.tolist() == [1, 0, 0, 1]


def test_comb_recovery_280hz() -> None:
    batch = _synth_rotor_events(bpf_hz=280.0, duration_s=0.6, seed=3)
    est = RotorPeriodicityEstimator()
    out = _run(est, batch)
    assert "t1" in out
    e = out["t1"]
    assert e.score > 0.5
    assert e.rotor_hz is not None
    assert abs(e.rotor_hz - 280.0) < 6.0


def test_polarity_cancellation_is_load_bearing() -> None:
    """Scoring ON+OFF must lose the comb the ON-OFF series recovers."""
    batch = _synth_rotor_events(bpf_hz=280.0, duration_s=0.6, seed=4)
    est = RotorPeriodicityEstimator()
    out = _run(est, batch)
    assert out["t1"].score > 0.5  # ON-OFF finds it
    assert abs(out["t1"].rotor_hz - 280.0) < 6.0
    # Flip every OFF event to ON: the anti-phase halves now sum to a flat
    # rate — the BPF must NOT be recoverable from the combined series (the
    # FRED finding), and whatever chance peak remains scores lower.
    flat = EventsBatch(
        timestamps=batch.timestamps, x_coords=batch.x_coords,
        y_coords=batch.y_coords,
        polarities=np.ones_like(batch.polarities),
    )
    est2 = RotorPeriodicityEstimator()
    out2 = _run(est2, flat)
    combined = out2["t1"]
    assert combined.rotor_hz is None or abs(combined.rotor_hz - 280.0) > 10.0
    assert combined.score < out["t1"].score * 0.7


def test_poisson_null_scores_low() -> None:
    for seed in range(5):
        batch = _synth_rotor_events(
            bpf_hz=280.0, duration_s=0.6, mod_rate_hz=0.0,
            bg_rate_hz=30_000.0, seed=seed,
        )
        est = RotorPeriodicityEstimator()
        out = _run(est, batch)
        if "t1" in out:
            assert out["t1"].score < 0.5, f"seed {seed}"


def test_gates_insufficient_data() -> None:
    est = RotorPeriodicityEstimator()
    batch = _synth_rotor_events(bpf_hz=280.0, duration_s=0.1, seed=5)
    out = _run(est, batch)
    # 100 ms < min_ring_bins x 0.5 ms = 320 ms -> no verdict or zero score.
    assert not out or out["t1"].score == 0.0


def test_blanking_on_bias_write() -> None:
    batch = _synth_rotor_events(bpf_hz=280.0, duration_s=0.6, seed=6)
    est = RotorPeriodicityEstimator()
    _run(est, batch)
    t_end = int(batch.timestamps[-1])
    est.notify_bias_write("cam", t_end)
    # Within the 1.5 s blank window nothing accumulates or returns.
    tail = _synth_rotor_events(
        bpf_hz=280.0, duration_s=0.2, seed=7, t0_us=t_end + 10_000,
    )
    for window in _windows(tail):
        assert est.update("cam", [_det()], window) == {}


def test_sticky_between_cadences_and_prune() -> None:
    batch = _synth_rotor_events(bpf_hz=280.0, duration_s=0.6, seed=8)
    est = RotorPeriodicityEstimator(stale_ticks=5)
    out = _run(est, batch)
    score = out["t1"].score
    # One more window returns the sticky estimate (no fresh PSD).
    extra = _synth_rotor_events(
        bpf_hz=280.0, duration_s=0.01, seed=9,
        t0_us=int(batch.timestamps[-1]) + 1,
    )
    for window in _windows(extra):
        out = est.update("cam", [_det()], window)
    assert out["t1"].score == score
    # Ticks without the track age it out.
    for i in range(8):
        est.update("cam", [], EventsBatch(
            timestamps=np.array([i], dtype=np.int64),
            x_coords=np.array([0], dtype=np.int16),
            y_coords=np.array([0], dtype=np.int16),
            polarities=np.array([1], dtype=np.int8),
        ))
    assert est._rings == {}


def test_tracker_dropout_resets_ring_no_false_comb() -> None:
    """Unobserved windows must RESET the ring, not zero-fill it.

    A polarity-imbalanced source observed on alternating windows would
    otherwise get the window pacing written into the ON-OFF series as a
    false in-band comb (skeptic finding, night 2).
    """
    batch = _synth_rotor_events(
        bpf_hz=280.0, duration_s=1.2, mod_rate_hz=0.0,
        bg_rate_hz=40_000.0, seed=11,
    )
    # Make the source strongly polarity-imbalanced: 80% ON.
    pol = (np.random.default_rng(1).random(batch.timestamps.size) < 0.8)
    imbalanced = EventsBatch(
        timestamps=batch.timestamps, x_coords=batch.x_coords,
        y_coords=batch.y_coords, polarities=pol.astype(np.int8),
    )
    est = RotorPeriodicityEstimator()
    out: dict[str, object] = {}
    for i, window in enumerate(_windows(imbalanced)):
        # Track visible only on alternating windows (flapping confirmation).
        tracked = [_det()] if i % 2 == 0 else []
        result = est.update("cam", tracked, window)
        if result:
            out = result
    # The ring never survives two consecutive observations, so no verdict
    # (or a scoreless one) — critically, NO high-score false comb.
    assert not out or out["t1"].score < 0.3


def test_fresh_flag_marks_new_verdicts_only() -> None:
    batch = _synth_rotor_events(bpf_hz=280.0, duration_s=0.6, seed=12)
    est = RotorPeriodicityEstimator()
    fresh_flags = []
    for window in _windows(batch):
        out = est.update("cam", [_det()], window)
        if "t1" in out:
            fresh_flags.append(out["t1"].fresh)
    assert any(fresh_flags)          # PSD verdicts happen
    assert not all(fresh_flags)      # sticky repeats are marked stale
    # A verdict window is followed by sticky (False) until the next cadence.
    first_fresh = fresh_flags.index(True)
    assert fresh_flags[first_fresh + 1] is False


def test_warmup_produces_no_estimate() -> None:
    """Insufficient data = NOT evaluated (no sticky 0.0 verdicts)."""
    batch = _synth_rotor_events(bpf_hz=280.0, duration_s=0.25, seed=13)
    est = RotorPeriodicityEstimator()
    for window in _windows(batch):
        out = est.update("cam", [_det()], window)
    assert out == {}  # ring below min_ring_bins -> nothing reported
