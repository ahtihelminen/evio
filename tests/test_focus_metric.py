"""Focus metric: blur ordering, rate invariance, hot-pixel immunity."""

from __future__ import annotations

import numpy as np

from evio.focus.metric import (
    ScoreHistory,
    WindowScorer,
    bin_count_image,
    hf_ratio,
    score_window,
)

ROI = (448, 168, 384, 384)  # centered 384x384 on the 1280x720 sensor


def _dot_scene_events(
    blur_px: float, n_events: int = 30_000, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Blinking-dot-grid events with gaussian defocus jitter."""
    rng = np.random.default_rng(seed)
    gx = np.linspace(520, 760, 6)
    gy = np.linspace(240, 480, 6)
    centers = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
    picks = rng.integers(0, centers.shape[0], n_events)
    pos = centers[picks] + rng.normal(
        scale=blur_px + 0.2, size=(n_events, 2),
    )
    xs = np.clip(pos[:, 0], 0, 1279).astype(np.int32)
    ys = np.clip(pos[:, 1], 0, 719).astype(np.int32)
    return xs, ys


def test_blur_ladder_orders_strictly() -> None:
    scores = [
        score_window(
            *_dot_scene_events(blur), roi=ROI, window_s=0.01, clip=0,
        ).hf
        for blur in (0.3, 1.5, 3.0, 6.0)
    ]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 1.5 * scores[-1]  # sharp >> defocused


def test_blur_ladder_survives_default_clip_at_physical_rates() -> None:
    """Real pixels fire <=~3 events per 10 ms window — the default
    clip must never invert the ordering on physically plausible scenes.
    """
    scores = [
        score_window(
            *_dot_scene_events(blur, n_events=800),
            roi=ROI, window_s=0.01,
        ).hf
        for blur in (1.0, 2.5, 5.0)
    ]
    assert scores == sorted(scores, reverse=True)


def test_score_invariant_to_event_rate() -> None:
    """Doubling the event rate must not materially move the score.

    (Exact invariance no longer holds: the shot-noise floor scales
    linearly while coherent signal power scales quadratically, so the
    corrected ratio IMPROVES slightly with more events.)
    """
    xs, ys = _dot_scene_events(0.5)
    lean = score_window(xs, ys, roi=ROI, window_s=0.01, clip=0).hf
    dense = score_window(
        np.concatenate([xs, xs]), np.concatenate([ys, ys]),
        roi=ROI, window_s=0.01, clip=0,
    ).hf
    assert lean > 0.05
    assert abs(lean - dense) < 0.1 * lean


def test_hot_pixel_swamps_unless_masked_or_clipped() -> None:
    xs, ys = _dot_scene_events(1.0, n_events=5_000)
    hot_x, hot_y = 600, 300
    xs_hot = np.concatenate([xs, np.full(500, hot_x, dtype=np.int32)])
    ys_hot = np.concatenate([ys, np.full(500, hot_y, dtype=np.int32)])

    clean = score_window(xs, ys, roi=ROI, window_s=0.01, clip=0).hf
    polluted = score_window(
        xs_hot, ys_hot, roi=ROI, window_s=0.01, clip=0,
    ).hf
    assert abs(polluted - clean) > 0.25 * clean  # the failure mode

    mask = np.zeros((720, 1280), dtype=bool)
    mask[hot_y, hot_x] = True
    masked = score_window(
        xs_hot, ys_hot, roi=ROI, window_s=0.01, stuck_mask=mask, clip=0,
    ).hf
    assert abs(masked - clean) < 0.02 * clean

    clipped = score_window(
        xs_hot, ys_hot, roi=ROI, window_s=0.01, clip=10.0,
    ).hf
    assert abs(clipped - clean) < abs(polluted - clean)


def test_bin_count_image_counts_and_roi_filtering() -> None:
    xs = np.array([10, 10, 11, 500], dtype=np.int32)
    ys = np.array([20, 20, 20, 600], dtype=np.int32)
    image = bin_count_image(xs, ys, roi=(8, 18, 8, 8))
    assert image.shape == (8, 8)
    assert image[2, 2] == 2.0  # (10,20) twice
    assert image[2, 3] == 1.0  # (11,20)
    assert image.sum() == 3.0  # (500,600) outside the ROI


def test_score_window_rates() -> None:
    xs = np.array([10, 500], dtype=np.int32)
    ys = np.array([20, 600], dtype=np.int32)
    result = score_window(xs, ys, roi=(8, 18, 8, 8), window_s=0.01)
    assert result.roi_rate == 100.0  # 1 event / 10 ms
    assert result.frame_rate == 200.0  # 2 events / 10 ms


def test_hf_ratio_empty_image_is_zero() -> None:
    assert hf_ratio(np.zeros((64, 64), dtype=np.float32)) == 0.0


def _dim_windows(
    blur: float, n_per_window: int, windows: int, warm_px_rate: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Dim blinking scene; optional unmasked warm pixel inside the ROI."""
    out = []
    for i in range(windows):
        if i % 2 == 0:  # ON half-cycle
            xs, ys = _dot_scene_events(blur, n_per_window, seed=i)
        else:
            xs = np.zeros(0, dtype=np.int32)
            ys = np.zeros(0, dtype=np.int32)
        if warm_px_rate:
            xs = np.concatenate(
                [xs, np.full(warm_px_rate, 640, dtype=np.int32)],
            )
            ys = np.concatenate(
                [ys, np.full(warm_px_rate, 360, dtype=np.int32)],
            )
        out.append((xs, ys))
    return out


def _tail_mean_score(
    feed: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    scorer = WindowScorer(window_s=0.01)
    scores = [
        scorer.score(xs, ys, roi=ROI).hf for xs, ys in feed
    ]
    tail = scores[len(scores) // 2 :]
    return float(np.mean(tail))


def test_warm_pixel_does_not_invert_ordering_in_accumulation() -> None:
    """A few-ev/window warm pixel (below every boot-scan gate) must not
    make defocus outscore focus in the dim/accumulate regime — the
    field-critical inversion the isolated-delta guard exists for."""
    focused = _tail_mean_score(_dim_windows(0.5, 200, 300, 5))
    defocused = _tail_mean_score(_dim_windows(5.0, 200, 300, 5))
    assert focused > 2.0 * defocused


def test_pure_noise_accumulation_never_spikes() -> None:
    """No-stimulus panes must stay at ~0 (no EMA/peak poisoning)."""
    rng = np.random.default_rng(3)
    scorer = WindowScorer(window_s=0.01)
    worst = 0.0
    for _ in range(300):
        score = scorer.score(
            rng.integers(0, 1280, 300).astype(np.int32),
            rng.integers(0, 720, 300).astype(np.int32),
            roi=ROI,
        ).hf
        worst = max(worst, score)
    assert worst <= 0.05


def test_score_history_wrap_peak_and_reset() -> None:
    history = ScoreHistory(capacity=8, ema_alpha=0.5)
    for value in (1.0, 5.0, 2.0):
        history.push(value)
    assert history.count == 3
    assert history.peak_ema >= history.ema > 0
    peak_before = history.peak_ema
    for _ in range(20):  # wrap the ring with low values
        history.push(0.1)
    raw, ema = history.recent(4)
    assert raw.shape == (4,) and ema.shape == (4,)
    assert np.allclose(raw, 0.1)
    assert history.peak_ema == peak_before  # peak survives the wrap
    history.reset_peak()
    assert history.peak_ema == history.ema
