"""Event-based focus sharpness metric.

The focus signal is the ratio of high-spatial-frequency energy to total
AC energy in the 2-D spectrum of a binned event-count image (ON+OFF
pooled, Hann-windowed). Normalizing by total AC energy makes the score
invariant to stimulus brightness and event-rate drift; it peaks when
the focus ring sits at best focus.

Hot/stuck pixels are spatial deltas — broadband HF energy that swamps
the score — so the count image is masked and clipped BEFORE the FFT.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

# Absolute SNR gate: raw AC band power must exceed the noise floor by
# this factor or the window scores 0. Measured separation: pure-noise
# windows (incl. rare leak spikes) stay <= ~1.06x floor, the weakest
# legitimate dim stimulus ~1.6x, focused scenes ~31x.
_SNR_GATE = 1.2
# Isolated-delta guard: a pixel may hold at most this multiple of its
# 3x3 neighborhood sum. Real optical structure is never a lone 1-px
# delta; an unmasked warm pixel accumulates COHERENT power the Poisson
# floor cannot model (it under-corrects by ~acc x G1/G2) and inverts
# the focus ordering on dim scenes.
_DELTA_NEIGHBOR_K = 2.0
# Keep 1-2 windows of accumulator memory even on bright scenes: a
# blinking stimulus otherwise scores raw 0 on every OFF half-cycle
# (alpha=1 hard cutover), wobbling the EMA ~34% peak-to-peak.
_ALPHA_MAX = 0.7

_HANN_CACHE: dict[tuple[int, int], NDArray[np.float32]] = {}
_BAND_CACHE: dict[
    tuple[int, int, float],
    tuple[NDArray[np.bool_], NDArray[np.bool_], int, int],
] = {}


@dataclass(frozen=True)
class FocusResult:
    """One window's focus reading."""

    # HF fraction of noise-corrected AC power; ~0..1, may slightly
    # exceed 1 when LF energy falls below the subtracted noise floor.
    hf: float
    roi_rate: float  # events/s inside the ROI (post-mask)
    frame_rate: float  # events/s over the full frame (pre-mask)


def bin_count_image(
    x_coords: NDArray[np.integer],
    y_coords: NDArray[np.integer],
    *,
    roi: tuple[int, int, int, int],
) -> NDArray[np.float32]:
    """Bin events inside ``roi`` (x, y, w, h) into a count image."""
    rx, ry, rw, rh = roi
    inside = (
        (x_coords >= rx)
        & (x_coords < rx + rw)
        & (y_coords >= ry)
        & (y_coords < ry + rh)
    )
    xs = (x_coords[inside].astype(np.intp)) - rx
    ys = (y_coords[inside].astype(np.intp)) - ry
    counts = np.bincount(ys * rw + xs, minlength=rw * rh)
    return counts.astype(np.float32).reshape(rh, rw)


def _hann2d(height: int, width: int) -> NDArray[np.float32]:
    window = _HANN_CACHE.get((height, width))
    if window is None:
        window = np.outer(
            np.hanning(height), np.hanning(width),
        ).astype(np.float32)
        _HANN_CACHE[(height, width)] = window
    return window


def _bands(
    height: int, width: int, hf_cutoff: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], int, int]:
    """(hf_mask, ac_mask, n_hf, n_ac) over rfft2; 1.0 = axis Nyquist."""
    key = (height, width, round(hf_cutoff, 6))
    bands = _BAND_CACHE.get(key)
    if bands is None:
        fy = np.abs(np.fft.fftfreq(height)) / 0.5
        fx = np.fft.rfftfreq(width) / 0.5
        radius = np.hypot(fy[:, None], fx[None, :])
        hf_mask = radius >= hf_cutoff
        ac_mask = radius > 1e-9
        bands = (
            hf_mask, ac_mask, int(hf_mask.sum()), int(ac_mask.sum()),
        )
        _BAND_CACHE[key] = bands
    return bands


def hf_ratio(
    count_image: NDArray[np.float32],
    *,
    hf_cutoff: float = 0.25,
    floor_scale: float = 1.0,
) -> float:
    """High-frequency fraction of the shot-noise-corrected AC power.

    Event-count images carry a Poisson shot-noise floor whose spectrum
    is flat — on sparse windows it dominates the high band and a naive
    HF/AC ratio then REWARDS defocus (spread-out events look like white
    noise). The floor (sum of counts x window^2, flat per bin, scaled
    by ``floor_scale`` for leaky-accumulated images) is subtracted from
    both bands with a 3-sigma margin, so pure noise scores exactly 0.

    The count image must already be hot-pixel masked/clipped; a single
    stuck pixel is a spatial delta and dominates every band.
    """
    height, width = count_image.shape
    window = _hann2d(height, width)
    windowed = (count_image - float(count_image.mean())) * window
    spectrum = np.fft.rfft2(windowed)
    power = spectrum.real**2 + spectrum.imag**2
    hf_mask, ac_mask, n_hf, n_ac = _bands(height, width, hf_cutoff)
    floor_per_bin = (
        float((count_image * window * window).sum()) * floor_scale
    )
    # ~4 correlated bins per Hann-smeared mode -> band dof ~ n/4; keep
    # bands only where they exceed the floor by 3 sigma of its own sum.
    margin_hf = 1.0 + 3.0 * float(np.sqrt(8.0 / max(n_hf, 1)))
    margin_ac = 1.0 + 3.0 * float(np.sqrt(8.0 / max(n_ac, 1)))
    raw_ac = float(power[ac_mask].sum())
    if raw_ac <= _SNR_GATE * floor_per_bin * n_ac:
        return 0.0  # no stimulus: below the absolute SNR gate
    ac_power = raw_ac - floor_per_bin * n_ac * margin_ac
    if ac_power <= 0.0:
        return 0.0
    hf_power = max(
        float(power[hf_mask].sum()) - floor_per_bin * n_hf * margin_hf,
        0.0,
    )
    return hf_power / ac_power


def score_window(
    x_coords: NDArray[np.integer],
    y_coords: NDArray[np.integer],
    *,
    roi: tuple[int, int, int, int],
    window_s: float,
    stuck_mask: NDArray[np.bool_] | None = None,
    clip: float = 10.0,
    hf_cutoff: float = 0.25,
) -> FocusResult:
    """Score one event window: mask -> clip -> Hann -> rFFT -> HF/AC."""
    image = bin_count_image(x_coords, y_coords, roi=roi)
    if stuck_mask is not None:
        rx, ry, rw, rh = roi
        image[stuck_mask[ry : ry + rh, rx : rx + rw]] = 0.0
    roi_events = float(image.sum())
    if clip > 0:
        np.minimum(image, clip, out=image)
    return FocusResult(
        hf=hf_ratio(image, hf_cutoff=hf_cutoff),
        roi_rate=roi_events / window_s,
        frame_rate=float(x_coords.size) / window_s,
    )


class WindowScorer:
    """Adaptive-SNR scorer for a live pane.

    Bright scenes are scored per window (full 100 Hz bandwidth). Dim
    scenes leak-accumulate ROI count images until ~``target_events``
    contribute, trading response time (up to ~1/alpha_min windows) for
    the SNR the spectral ratio needs — without this, sparse windows
    score as noise/noise. For the leaky sum ``acc = (1 - a) * acc +
    img`` the Poisson variance carries the SQUARED weights, so the
    exact running totals G1 = sum(g) and G2 = sum(g^2) are tracked and
    ``floor_scale = G2 / G1`` — exact through startup transients and
    alpha drift (the steady-state shortcut ``1 / (2 - a)`` leaves a
    floor residual that scores pure noise as ~the band-area ratio).
    """

    def __init__(
        self,
        *,
        window_s: float,
        clip: float = 10.0,
        hf_cutoff: float = 0.25,
        target_events: float = 4000.0,
        alpha_min: float = 0.02,
    ) -> None:
        self._window_s = window_s
        self._clip = clip
        self._hf_cutoff = hf_cutoff
        self._target = target_events
        self._alpha_min = alpha_min
        self._acc: NDArray[np.float32] | None = None
        self._roi: tuple[int, int, int, int] | None = None
        self._events_ema = 0.0
        self._g1 = 1.0
        self._g2 = 1.0

    @property
    def acc_image(self) -> NDArray[np.float32] | None:
        """Accumulated masked/clipped ROI count image — the exact
        evidence the score is computed on (callers must copy)."""
        return self._acc

    def score(
        self,
        x_coords: NDArray[np.integer],
        y_coords: NDArray[np.integer],
        *,
        roi: tuple[int, int, int, int],
        stuck_mask: NDArray[np.bool_] | None = None,
    ) -> FocusResult:
        image = bin_count_image(x_coords, y_coords, roi=roi)
        if stuck_mask is not None:
            rx, ry, rw, rh = roi
            image[stuck_mask[ry : ry + rh, rx : rx + rw]] = 0.0
        roi_events = float(image.sum())
        if self._clip > 0:
            np.minimum(image, self._clip, out=image)
        if self._acc is None or roi != self._roi:
            self._acc = image.copy()
            self._roi = roi
            self._events_ema = roi_events
            self._g1 = 1.0
            self._g2 = 1.0
        else:
            self._events_ema += 0.2 * (roi_events - self._events_ema)
            alpha = float(
                np.clip(
                    self._events_ema / self._target, self._alpha_min,
                    _ALPHA_MAX,
                ),
            )
            keep = 1.0 - alpha
            self._acc *= keep
            self._acc += image
            self._g1 = self._g1 * keep + 1.0
            self._g2 = self._g2 * keep * keep + 1.0
        return FocusResult(
            hf=hf_ratio(
                _suppress_isolated_deltas(self._acc),
                hf_cutoff=self._hf_cutoff,
                floor_scale=self._g2 / self._g1,
            ),
            roi_rate=roi_events / self._window_s,
            frame_rate=float(x_coords.size) / self._window_s,
        )


def _suppress_isolated_deltas(
    image: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Clamp every pixel to ``k x`` its 3x3 neighborhood sum.

    An unmasked warm pixel (even a few ev/window — below every
    boot-scan gate) accumulates a coherent spatial delta whose power
    grows quadratically while the Poisson floor only credits it
    linearly; on dim scenes that single pixel inverts the focus
    ordering. Isolated deltas have ~empty neighborhoods, so the clamp
    zeroes them while leaving real (extended) structure untouched.
    """
    neighborhood = cv2.boxFilter(
        image, -1, (3, 3), normalize=False,
    ) - image
    return np.minimum(image, _DELTA_NEIGHBOR_K * neighborhood)


class ScoreHistory:
    """Ring buffer of scores with EMA smoothing and peak-hold."""

    def __init__(
        self, capacity: int = 4096, ema_alpha: float = 0.15,
    ) -> None:
        self._raw = np.zeros(capacity, dtype=np.float32)
        self._ema = np.zeros(capacity, dtype=np.float32)
        self._alpha = ema_alpha
        self.count = 0
        self.ema = 0.0
        self.peak_ema = 0.0

    def push(self, value: float) -> None:
        self.ema = (
            value
            if self.count == 0
            else self.ema + self._alpha * (value - self.ema)
        )
        slot = self.count % self._raw.size
        self._raw[slot] = value
        self._ema[slot] = self.ema
        self.count += 1
        self.peak_ema = max(self.peak_ema, self.ema)

    def reset_peak(self) -> None:
        self.peak_ema = self.ema

    def recent(
        self, n_samples: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Last ``n_samples`` (raw, ema), oldest first."""
        have = min(self.count, self._raw.size, n_samples)
        if have == 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty
        end = self.count % self._raw.size
        idx = (np.arange(end - have, end)) % self._raw.size
        return self._raw[idx].copy(), self._ema[idx].copy()
