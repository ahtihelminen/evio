from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.detection.event_tokens import DEFAULT_E_MAX, tokenize_events_in_track


def _batch(
    ts: Sequence[int],
    xs: Sequence[int],
    ys: Sequence[int],
    pol: Sequence[int],
) -> EventsBatch:
    return EventsBatch(
        timestamps=np.asarray(ts, dtype=np.int64),
        x_coords=np.asarray(xs, dtype=np.int16),
        y_coords=np.asarray(ys, dtype=np.int16),
        polarities=np.asarray(pol, dtype=np.int8),
    )


def test_returns_padded_zeros_for_empty_input() -> None:
    batch = _batch([], [], [], [])
    events, mask = tokenize_events_in_track(
        batch, bboxes=[(0, 0, 10, 10)], timestamps_us=[1000],
        window_us=100,
    )
    assert events.shape == (DEFAULT_E_MAX, 4)
    assert mask.shape == (DEFAULT_E_MAX,)
    assert events.dtype == np.float32
    assert mask.dtype == np.bool_
    assert not mask.any()


def test_normalises_t_x_y_into_unit_range() -> None:
    # 4 events spanning the full bbox + window. Polarity alternates.
    bbox = (10, 20, 30, 40)
    timestamps_us = [1000, 2000]  # T=2, window_us=200 -> [900, 2100]
    batch = _batch(
        ts=[900, 1500, 1500, 2099],
        xs=[10, 20, 29, 29],
        ys=[20, 30, 39, 39],
        pol=[1, -1, 1, -1],
    )
    events, mask = tokenize_events_in_track(
        batch, bboxes=[bbox, bbox], timestamps_us=timestamps_us,
        window_us=200, e_max=8,
    )
    assert mask.sum() == 4
    # x_norm and y_norm in [0, 1)
    assert (events[:4, 1] >= 0.0).all()
    assert (events[:4, 1] < 1.0).all()
    assert (events[:4, 2] >= 0.0).all()
    assert (events[:4, 2] < 1.0).all()
    # t_norm in [0, 1)
    assert (events[:4, 0] >= 0.0).all()
    assert (events[:4, 0] < 1.0).all()
    # polarity in {-1, +1}
    assert set(events[:4, 3].tolist()) == {-1.0, 1.0}
    # padding rows are zero
    assert (events[4:] == 0).all()


def test_drops_events_outside_spatial_or_temporal_window() -> None:
    bbox = (10, 10, 20, 20)
    # 3 events: in-bbox in-time, in-bbox out-of-time, out-of-bbox in-time
    batch = _batch(
        ts=[1000, 5000, 1000],
        xs=[15, 15, 100],
        ys=[15, 15, 15],
        pol=[1, 1, 1],
    )
    _events, mask = tokenize_events_in_track(
        batch, bboxes=[bbox], timestamps_us=[1000],
        window_us=200,  # window spans [900, 1100]
        e_max=8,
    )
    assert mask.sum() == 1


def test_uses_union_bbox_for_moving_track() -> None:
    # Two windows with bboxes moving from (0,0)-(10,10) to (10,10)-(20,20).
    # Union: (0,0)-(20,20).
    bboxes = [(0, 0, 10, 10), (10, 10, 20, 20)]
    timestamps_us = [1000, 2000]
    batch = _batch(
        ts=[1000, 2000],
        xs=[5, 15],    # both inside union
        ys=[5, 15],
        pol=[1, 1],
    )
    _events, mask = tokenize_events_in_track(
        batch, bboxes=bboxes, timestamps_us=timestamps_us,
        window_us=200, e_max=8,
    )
    assert mask.sum() == 2


def test_uniform_stride_subsample_when_exceeding_e_max() -> None:
    # Make events exactly fill the window so subsampling output spans [0, 1).
    bbox = (0, 0, 100, 100)
    n = 10_000
    # window centred at 50_000 with window_us=100_000 -> [0, 100_000).
    # Events at 0..99_990 us so they fill the window.
    ts = np.arange(n, dtype=np.int64) * 10
    xs = np.full(n, 50, dtype=np.int16)
    ys = np.full(n, 50, dtype=np.int16)
    pol = np.ones(n, dtype=np.int8)
    batch = EventsBatch(
        timestamps=ts, x_coords=xs, y_coords=ys, polarities=pol,
    )
    events, mask = tokenize_events_in_track(
        batch, bboxes=[bbox], timestamps_us=[50_000],
        window_us=100_000, e_max=128,
    )
    assert mask.sum() == 128
    # First t_norm near 0, last near 1 — uniform stride.
    assert events[0, 0] < 0.05
    assert events[127, 0] > 0.95


def test_e_max_validation() -> None:
    batch = _batch([0], [0], [0], [1])
    with pytest.raises(ValueError, match="e_max"):
        tokenize_events_in_track(
            batch, bboxes=[(0, 0, 10, 10)], timestamps_us=[0],
            window_us=100, e_max=0,
        )


def test_mismatched_bboxes_and_timestamps_raises() -> None:
    batch = _batch([0], [0], [0], [1])
    with pytest.raises(ValueError, match="bboxes and timestamps"):
        tokenize_events_in_track(
            batch, bboxes=[(0, 0, 10, 10)], timestamps_us=[0, 100],
            window_us=100,
        )


def test_per_pixel_cap_bounds_stuck_pixel_keeps_blob() -> None:
    import numpy as np

    from evio.core.events import EventsBatch
    from evio.detection.event_tokens import tokenize_events_in_track

    rng = np.random.default_rng(7)
    n_stuck, n_blob = 5000, 200
    ts = np.sort(rng.integers(0, 40_000, n_stuck + n_blob)).astype(np.int64)
    xs = np.full(ts.size, 50, dtype=np.int16)
    ys = np.full(ts.size, 60, dtype=np.int16)
    blob_at = rng.choice(ts.size, n_blob, replace=False)
    xs[blob_at] = 80 + rng.integers(0, 5, n_blob)
    ys[blob_at] = 90 + rng.integers(0, 5, n_blob)
    batch = EventsBatch(
        timestamps=ts, x_coords=xs, y_coords=ys,
        polarities=rng.integers(0, 2, ts.size).astype(np.int8),
    )
    kw = {"bboxes": [(40, 50, 100, 110)] * 4,
          "timestamps_us": [5_000, 15_000, 25_000, 35_000],
          "window_us": 10_000}

    tok, mask = tokenize_events_in_track(batch, e_max=1024, **kw)
    assert tok.shape[1] == 4  # default contract unchanged
    stuck_frac = ((tok[mask, 1] * 60 + 40).round() == 50).mean()
    assert stuck_frac > 0.9  # the spoof lane, demonstrated

    tok, mask = tokenize_events_in_track(
        batch, e_max=1024, per_pixel_cap=16, with_pixel_count=True, **kw)
    assert tok.shape[1] == 5
    xs_tok = (tok[mask, 1] * 60 + 40).round()
    stuck = xs_tok == 50
    assert stuck.sum() <= 16          # stuck pixel bounded
    assert (~stuck).sum() >= 190      # blob intact
    # 5th feature separates one-fast-pixel from coherent object
    assert tok[mask, 4][stuck].mean() > 3 * tok[mask, 4][~stuck].mean()


def test_pixel_count_without_cap_and_empty_paths() -> None:
    import numpy as np

    from evio.core.events import EventsBatch
    from evio.detection.event_tokens import tokenize_events_in_track

    batch = EventsBatch(
        timestamps=np.array([5_000], dtype=np.int64),
        x_coords=np.array([10], dtype=np.int16),
        y_coords=np.array([10], dtype=np.int16),
        polarities=np.array([1], dtype=np.int8),
    )
    tok, mask = tokenize_events_in_track(
        batch, bboxes=[(0, 0, 32, 32)], timestamps_us=[5_000],
        window_us=10_000, e_max=16, with_pixel_count=True)
    assert tok.shape == (16, 5) and mask.sum() == 1
    # out-of-box -> empty with the right width
    tok, mask = tokenize_events_in_track(
        batch, bboxes=[(100, 100, 132, 132)], timestamps_us=[5_000],
        window_us=10_000, e_max=16, with_pixel_count=True)
    assert tok.shape == (16, 5) and mask.sum() == 0
