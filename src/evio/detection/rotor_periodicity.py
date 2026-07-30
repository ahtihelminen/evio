"""Per-track rotor-periodicity estimator (SN-31 Stage 1, main-side).

Optical micro-Doppler: a lit rotor modulates a track's event rate at the
blade-pass frequency (BPF). Per (camera, track) this keeps rings of 0.5 ms
polarity-binned event counts (~2 KB/track — never raw events) and, every
``eval_cadence_windows``, runs a Hann periodogram over the **ON - OFF**
series — the load-bearing choice: rotational symmetry cancels the comb in the
combined-count series (the FRED 22x11 px finding), only the polarity
difference survives. A Nyquist-folded harmonic-comb score plus a spectral
flatness gate turns the peak into ``periodicity_score`` ∈ [0, 1).

One-sided evidence by design (plan §0): presence confirms drone; absence
proves nothing (edge-on/unlit rotor, night physics). Scores ride the fusion
LLR bus via ``Detection.periodicity_score`` — never ``m_coh``. Rings blank for
1.5 s after a bias write (a threshold step is a broadband splash).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from evio.core.events import EventsBatch
    from evio.detection.types import Detection


@dataclass(frozen=True, slots=True)
class RotorEstimate:
    """One periodicity verdict for a track (sticky between eval cadences).

    ``fresh`` is True only on the window whose PSD produced this verdict —
    consumers adding the score as SPRT evidence must use fresh estimates only,
    else one verdict is re-counted every window until the next cadence.
    """

    rotor_hz: float | None
    score: float
    sfm: float = 1.0
    n_bins: int = 0
    n_events: int = 0
    fresh: bool = False


@dataclass(slots=True)
class _TrackRing:
    on: deque[int]
    off: deque[int]
    next_bin: int = -1  # first bin not yet CLOSED; -1 = empty
    total_events: int = 0
    windows_since_eval: int = 0
    seen_tick: int = 0
    # Events belonging to the still-open bin (a window rarely ends on a bin
    # boundary; closing the partial bin early imprints the 100 Hz window
    # pacing onto the series as a spurious comb).
    carry_ts: np.ndarray | None = None
    carry_pol: np.ndarray | None = None


def bin_events_polarity(
    ts_us: np.ndarray,
    polarities: np.ndarray,
    *,
    start_bin: int,
    n_bins: int,
    bin_us: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin event timestamps into per-polarity 0.5 ms counts.

    Bins are absolute-time-anchored (``bin = ts_us // bin_us``) so gaps
    between windows zero-fill instead of silently compressing time.
    """
    rel = (ts_us // bin_us).astype(np.int64) - start_bin
    keep = (rel >= 0) & (rel < n_bins)
    rel = rel[keep]
    pol = polarities[keep]
    on = np.bincount(rel[pol > 0], minlength=n_bins).astype(np.int64)
    off = np.bincount(rel[pol <= 0], minlength=n_bins).astype(np.int64)
    return on, off


def _spectral_flatness(psd_band: np.ndarray) -> float:
    """Geometric / arithmetic mean of the band PSD — 1.0 = noise-flat."""
    if psd_band.size == 0:
        return 1.0
    log_mean = float(np.mean(np.log(psd_band + 1e-20)))
    return math.exp(log_mean) / (float(np.mean(psd_band)) + 1e-20)


def _running_median(psd: np.ndarray, half_width: int, guard: int) -> np.ndarray:
    """Local median background per bin, excluding a small guard band."""
    n = psd.size
    med = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half_width), min(n, i + half_width + 1)
        window = np.concatenate(
            [psd[lo: max(lo, i - guard)], psd[min(hi, i + guard + 1): hi]],
        )
        med[i] = np.median(window) if window.size else psd[i]
    return med


def _parabolic_refine(freqs: np.ndarray, log_psd: np.ndarray, i: int) -> float:
    """Sub-bin peak-frequency refinement via a 3-point parabola."""
    if i <= 0 or i >= log_psd.size - 1:
        return float(freqs[i])
    y0, y1, y2 = log_psd[i - 1], log_psd[i], log_psd[i + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(freqs[i])
    delta = 0.5 * (y0 - y2) / denom
    step = float(freqs[1] - freqs[0])
    return float(freqs[i]) + max(-0.5, min(0.5, delta)) * step


def _fold(f: float, nyquist: float) -> float:
    """Alias a super-Nyquist harmonic back to its in-band location."""
    f = f % (2.0 * nyquist)
    return f if f <= nyquist else 2.0 * nyquist - f


def series_psd(
    series: np.ndarray, bin_us: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hann periodogram ``(freqs_hz, psd)`` of a count series (mean removed)."""
    n = series.size
    centered = series - series.mean()
    window = np.hanning(n)
    psd = np.abs(np.fft.rfft(centered * window)) ** 2 / float(
        np.sum(window * window),
    )
    freqs = np.fft.rfftfreq(n, d=bin_us * 1e-6)
    return freqs, psd


def comb_score_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    *,
    band_hz: tuple[float, float],
    harmonics: int = 3,
    sfm_max: float = 0.6,
    kappa: float = 6.0,
    fold_nyquist: float | None = None,
    n_candidates: int = 1,
) -> tuple[float | None, float, float]:
    """``(f0_hz, score, sfm)`` harmonic-comb verdict over a uniform PSD grid.

    The scoring core shared by the live per-track estimator and the offline
    comb-at-locus evaluator. ``fold_nyquist`` aliases super-Nyquist harmonics
    back in-band (binned-series callers pass ``freqs[-1]``); ``None`` (NDFT
    grids — no sampling Nyquist) evaluates harmonics at ``h*f0`` directly and
    skips ones beyond the grid. ``n_candidates > 1`` anchors the comb at each
    of the top-K prominent in-band peaks and keeps the best — a strong 2nd
    harmonic no longer steals the fundamental's comb credit.
    """
    band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not band.any():
        return None, 0.0, 1.0
    sfm = _spectral_flatness(psd[band])
    if sfm > sfm_max:
        return None, 0.0, sfm
    step = float(freqs[1] - freqs[0])
    med = _running_median(
        psd, half_width=max(1, int(40.0 / step)),
        guard=max(1, int(5.0 / step)),
    )
    prom = psd / (med + 1e-12)
    banded_prom = np.where(band, prom, 0.0)
    order = np.argsort(banded_prom)[::-1][: max(1, n_candidates)]
    peak_indices = [int(i) for i in order if banded_prom[i] > 0.0] or [
        int(np.argmax(banded_prom)),
    ]

    def prom_at(f_hz: float) -> float:
        i = round(f_hz / step)
        lo, hi = max(0, i - 1), min(prom.size, i + 2)
        return float(prom[lo:hi].max()) if hi > lo else 0.0

    weights = (1.0, 0.5, 0.25, 0.125)[:harmonics]
    guard_hz = 5.0
    grid_hi = float(freqs[-1])
    best: tuple[float | None, float] = (None, 0.0)
    for peak_idx in peak_indices:
        f0 = _parabolic_refine(freqs, np.log(psd + 1e-12), peak_idx)
        # Fundamental always counts; higher harmonics only with real support
        # (prominence >= 2) AND a location that is neither back on the
        # fundamental (3*f0 folds onto f0 near fs/4) nor below the band floor
        # (1/f clutter shoulder) — else a chance peak inherits comb credit.
        comb_snr = weights[0] * math.log1p(max(0.0, prom_at(f0) - 1.0))
        for h, w in zip(range(2, harmonics + 1), weights[1:], strict=False):
            if fold_nyquist is not None:
                f_h = _fold(h * f0, fold_nyquist)
            else:
                f_h = h * f0
                if f_h > grid_hi:
                    continue
            if f_h < band_hz[0] or abs(f_h - f0) <= guard_hz:
                continue
            p = prom_at(f_h)
            if p >= 2.0:
                comb_snr += w * math.log1p(p - 1.0)
        score = 1.0 - math.exp(-comb_snr / kappa)
        if score > best[1]:
            best = (f0, score)
    return best[0], best[1], sfm


class RotorPeriodicityEstimator:
    """Accumulate per-track ON/OFF bins and score the blade-pass comb.

    Call :meth:`update` once per window with the tracked detections + the
    window's raw batch; it returns ``{track_id: RotorEstimate}`` (sticky —
    the last verdict repeats between eval cadences). Lifecycle mirrors
    ``TrackTransformerClassifier`` (per-(serial, track) state, tick-based
    pruning). ``notify_bias_write`` blanks a camera for ``blank_us``.
    """

    def __init__(
        self,
        *,
        bin_us: int = 500,
        ring_bins: int = 960,
        eval_cadence_windows: int = 16,
        min_ring_bins: int = 640,
        min_ring_events: int = 400,
        band_hz: tuple[float, float] = (30.0, 650.0),
        harmonics: int = 3,
        sfm_max: float = 0.6,
        comb_kappa: float = 6.0,
        report_floor: float = 0.1,
        blank_us: int = 1_500_000,
        max_gap_us: int = 200_000,
        bbox_margin_px: int = 2,
        stale_ticks: int = 300,
    ) -> None:
        self._bin_us = bin_us
        self._ring_bins = ring_bins
        self._cadence = max(1, eval_cadence_windows)
        self._min_bins = min_ring_bins
        self._min_events = min_ring_events
        self._band = band_hz
        self._harmonics = harmonics
        self._sfm_max = sfm_max
        self._kappa = comb_kappa
        self._floor = report_floor
        self._blank_us = blank_us
        self._max_gap_bins = max(1, max_gap_us // bin_us)
        self._margin = bbox_margin_px
        self._stale_ticks = stale_ticks
        self._rings: dict[tuple[str, str], _TrackRing] = {}
        self._estimates: dict[tuple[str, str], RotorEstimate] = {}
        self._blank_until: dict[str, int] = {}
        # Per-serial window counters — one shared counter would age every
        # camera's rings at (n_cameras x) the intended rate.
        self._ticks: dict[str, int] = {}
        self._last_seen_us: dict[str, int] = {}

    def notify_bias_write(self, serial: str, t_us: int | None = None) -> None:
        """Blank the camera and clear its rings (post-bias-write splash).

        ``t_us`` defaults to the serial's last-seen event time, so the bias
        runner can notify without knowing the camera's event clock.
        """
        if t_us is None:
            t_us = self._last_seen_us.get(serial, 0)
        self._blank_until[serial] = t_us + self._blank_us
        for key in [k for k in self._rings if k[0] == serial]:
            del self._rings[key]
            self._estimates.pop(key, None)

    def update(
        self,
        serial: str,
        tracked: list[Detection],
        batch: EventsBatch,
    ) -> dict[str, RotorEstimate]:
        """Accumulate this window; return sticky estimates per track_id."""
        tick = self._ticks.get(serial, 0) + 1
        self._ticks[serial] = tick
        if batch.timestamps.size == 0:
            return {}
        now_us = int(batch.timestamps[-1])
        self._last_seen_us[serial] = now_us
        if now_us < self._blank_until.get(serial, 0):
            return {}
        out: dict[str, RotorEstimate] = {}
        for det in tracked:
            if not det.track_id:
                continue
            key = (serial, det.track_id)
            ring = self._rings.get(key)
            if ring is None:
                ring = _TrackRing(
                    on=deque(maxlen=self._ring_bins),
                    off=deque(maxlen=self._ring_bins),
                )
                self._rings[key] = ring
            elif ring.seen_tick != tick - 1:
                # The tracker dropped this track for >=1 window: those bins
                # were never observed, and zero-filling them writes the window
                # pacing into the ON-OFF series as a false in-band comb (a
                # polarity-imbalanced source + alternate-window dropout = a
                # ~50 Hz square wave). Unobserved != quiet: start over.
                self._reset_ring(ring)
                self._estimates.pop(key, None)
            ring.seen_tick = tick
            self._accumulate(ring, det, batch)
            ring.windows_since_eval += 1
            fresh: RotorEstimate | None = None
            if ring.windows_since_eval >= self._cadence:
                ring.windows_since_eval = 0
                fresh = self._score(ring)
                if fresh is not None:
                    self._estimates[key] = fresh
            est = self._estimates.get(key)
            if fresh is not None:
                out[det.track_id] = fresh
            elif est is not None:
                # Sticky repeat of the last verdict (fresh=False).
                out[det.track_id] = RotorEstimate(
                    est.rotor_hz, est.score, est.sfm, est.n_bins,
                    est.n_events, fresh=False,
                )
        self._prune(serial, tick)
        return out

    @staticmethod
    def _reset_ring(ring: _TrackRing) -> None:
        ring.on.clear()
        ring.off.clear()
        ring.total_events = 0
        ring.next_bin = -1
        ring.windows_since_eval = 0
        ring.carry_ts = None
        ring.carry_pol = None

    def _accumulate(
        self, ring: _TrackRing, det: Detection, batch: EventsBatch,
    ) -> None:
        m = self._margin
        in_roi = (
            (batch.x_coords >= det.x0 - m) & (batch.x_coords <= det.x1 + m)
            & (batch.y_coords >= det.y0 - m) & (batch.y_coords <= det.y1 + m)
        )
        ts = batch.timestamps[in_roi]
        pol = batch.polarities[in_roi]
        if ring.carry_ts is not None:
            ts = np.concatenate([ring.carry_ts, ts])
            pol = np.concatenate([ring.carry_pol, pol])
            ring.carry_ts = None
            ring.carry_pol = None
        # The bin containing the batch's last event is still OPEN — later
        # windows may add to it. Close only [next_bin, open_bin); events in
        # the open bin carry over so the 100 Hz window pacing never imprints
        # on the series.
        open_bin = int(batch.timestamps[-1]) // self._bin_us
        if ring.next_bin < 0:
            ring.next_bin = min(
                int(ts[0]) // self._bin_us if ts.size else open_bin,
                open_bin,
            )
        n_closed = open_bin - ring.next_bin
        if n_closed > self._max_gap_bins:
            self._reset_ring(ring)
            ring.next_bin = open_bin
        elif n_closed > 0:
            on, off = bin_events_polarity(
                ts, pol, start_bin=ring.next_bin, n_bins=n_closed,
                bin_us=self._bin_us,
            )
            ring.on.extend(on.tolist())
            ring.off.extend(off.tolist())
            ring.total_events += int(on.sum() + off.sum())
            ring.next_bin = open_bin
        in_open = (ts // self._bin_us) == open_bin
        if in_open.any():
            ring.carry_ts = ts[in_open]
            ring.carry_pol = pol[in_open]
        # Keep the event budget honest as bins age out of the deque.
        if len(ring.on) == self._ring_bins:
            ring.total_events = int(sum(ring.on) + sum(ring.off))

    def _score(self, ring: _TrackRing) -> RotorEstimate | None:
        """PSD + comb score; None = insufficient data (NOT evaluated)."""
        n = len(ring.on)
        n_events = ring.total_events
        if n < self._min_bins or n_events < self._min_events:
            return None  # warm-up != "evaluated, nothing found"
        series = np.asarray(ring.on, dtype=np.float64) - np.asarray(
            ring.off, dtype=np.float64,
        )
        freqs, psd = series_psd(series, self._bin_us)
        f0, score, sfm = comb_score_psd(
            freqs, psd, band_hz=self._band, harmonics=self._harmonics,
            sfm_max=self._sfm_max, kappa=self._kappa,
            fold_nyquist=float(freqs[-1]),
        )
        rotor_hz = f0 if score >= self._floor else None
        return RotorEstimate(rotor_hz, score, sfm, n, n_events, fresh=True)

    def _prune(self, serial: str, tick: int) -> None:
        stale = [
            key for key, ring in self._rings.items()
            if key[0] == serial
            and tick - ring.seen_tick > self._stale_ticks
        ]
        for key in stale:
            del self._rings[key]
            self._estimates.pop(key, None)


__all__ = [
    "RotorEstimate",
    "RotorPeriodicityEstimator",
    "bin_events_polarity",
    "comb_score_psd",
    "series_psd",
]
