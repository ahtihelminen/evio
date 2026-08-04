"""Raw-event tokenisation for the v3 temporal transformer.

The voxel attempt failed because per-event signal got split across 16
GroupNorm-noisy channels of a CNN built for BGR statistics. Going to
raw events sidesteps the encoder mismatch entirely: each event is one
token whose ``(t, x, y, p)`` feature vector is projected into the
transformer's 128-D working space and attended over alongside the
spatial frame tokens.

This module is a pure-numpy tokeniser. It takes one labelled
sequence's worth of bboxes (T windows over a track) and produces:

* ``events`` — ``(E_max, 4) float32`` normalised features:
  ``[t_norm, x_norm, y_norm, p_signed]`` with each axis in ``[0, 1]``
  except polarity which is in ``{-1, +1}``.
* ``mask`` — ``(E_max,) bool`` — ``True`` for valid event tokens,
  ``False`` for padding.

Events are kept only when they fall inside the spatial *union* of the
T per-frame bboxes AND the temporal span ``[t_first - window/2,
t_last + window/2]``. Sub-sampling is stride-uniform when count
exceeds ``E_max``, preserving the temporal distribution.

The output is meant to be stacked across the corpus and saved
alongside the BGR patches; the v3 model consumes both branches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from evio.core.events import EventsBatch

DEFAULT_E_MAX: int = 1024


def tokenize_events_in_track(
    batch: EventsBatch,
    *,
    bboxes: list[tuple[int, int, int, int]],
    timestamps_us: list[int],
    window_us: int,
    e_max: int = DEFAULT_E_MAX,
    per_pixel_cap: int | None = None,
    with_pixel_count: bool = False,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Crop + normalise + sub-sample events for one track sequence.

    ``bboxes`` and ``timestamps_us`` must have the same length (T) and
    correspond to the per-frame annotations the spatial branch sees.
    The union of bboxes defines the spatial crop; the first/last
    timestamps + ``window_us`` defines the temporal crop.
    """
    if len(bboxes) != len(timestamps_us):
        msg = (
            f"bboxes and timestamps must align: got {len(bboxes)} "
            f"bboxes vs {len(timestamps_us)} timestamps"
        )
        raise ValueError(msg)
    if e_max < 1:
        msg = f"e_max must be >= 1, got {e_max}"
        raise ValueError(msg)
    if not bboxes:
        return _empty(e_max, with_pixel_count=with_pixel_count)

    x0u = min(b[0] for b in bboxes)
    y0u = min(b[1] for b in bboxes)
    x1u = max(b[2] for b in bboxes)
    y1u = max(b[3] for b in bboxes)
    width = max(1, x1u - x0u)
    height = max(1, y1u - y0u)

    half = window_us // 2
    t_min = timestamps_us[0] - half
    t_max = timestamps_us[-1] + half
    span = max(1, t_max - t_min)

    # Time crop FIRST via searchsorted on the (assumed sorted) timestamps —
    # the spatial mask then only scans the small window slice instead of
    # the entire per-clip event array. Caller-supplied timestamps in
    # every extractor (FRED memmap cache, NeRDD npz, Ev-Flying npy,
    # eTraM h5) ship sorted, so this is safe.
    lo = int(np.searchsorted(batch.timestamps, t_min, side="left"))
    hi = int(np.searchsorted(batch.timestamps, t_max, side="right"))
    if hi <= lo:
        return _empty(e_max, with_pixel_count=with_pixel_count)
    ts_slice = batch.timestamps[lo:hi]
    xs_slice = batch.x_coords[lo:hi]
    ys_slice = batch.y_coords[lo:hi]
    pol_slice = batch.polarities[lo:hi]

    in_box = (
        (xs_slice >= x0u) & (xs_slice < x1u)
        & (ys_slice >= y0u) & (ys_slice < y1u)
    )
    local_indices = np.where(in_box)[0]
    if local_indices.size == 0:
        return _empty(e_max, with_pixel_count=with_pixel_count)

    # Per-pixel token cap (R1 spoof-lane fix): a kHz stuck pixel inside the
    # union box otherwise claims tokens proportional to its rate and forges
    # periodic structure. Capping BEFORE the stride subsample bounds any
    # single (x, y) to per_pixel_cap tokens while the pre-cap count is
    # preserved as an optional 5th feature (log-compressed), so "one pixel
    # firing fast" stays separable from "spatially coherent object".
    if with_pixel_count or per_pixel_cap is not None:
        pixel_ids = (
            ys_slice[local_indices].astype(np.int64) * 16384
            + xs_slice[local_indices].astype(np.int64)
        )
        _, inverse, pix_counts = np.unique(
            pixel_ids, return_inverse=True, return_counts=True,
        )
        counts_per_event = pix_counts[inverse]
    if per_pixel_cap is not None:
        order = np.argsort(inverse, kind="stable")
        rank = np.empty(order.size, dtype=np.int64)
        boundaries = np.flatnonzero(
            np.diff(inverse[order], prepend=inverse[order[0]] ^ 1),
        )
        rank[order] = np.arange(order.size) - np.repeat(
            boundaries, np.diff(np.append(boundaries, order.size)),
        )
        keep_cap = rank < per_pixel_cap
        local_indices = local_indices[keep_cap]
        counts_per_event = counts_per_event[keep_cap]
        if local_indices.size == 0:
            return _empty(e_max, with_pixel_count=with_pixel_count)

    if local_indices.size > e_max:
        # Uniform-stride subsample preserves temporal distribution
        # better than random — keeps the rotor-frequency periodicity
        # legible to the attention layer.
        step = local_indices.size / e_max
        keep = (np.arange(e_max) * step).astype(np.int64)
        local_indices = local_indices[keep]
        if with_pixel_count:
            counts_per_event = counts_per_event[keep]

    dims = 5 if with_pixel_count else 4
    out = np.zeros((e_max, dims), dtype=np.float32)
    n = int(local_indices.size)
    ts = ts_slice[local_indices].astype(np.float32)
    xs = xs_slice[local_indices].astype(np.float32)
    ys = ys_slice[local_indices].astype(np.float32)
    ps = (pol_slice[local_indices] > 0).astype(np.float32) * 2.0 - 1.0
    out[:n, 0] = (ts - t_min) / span
    out[:n, 1] = (xs - x0u) / width
    out[:n, 2] = (ys - y0u) / height
    out[:n, 3] = ps
    if with_pixel_count:
        # log1p of the PRE-cap per-pixel count, /10 to keep the feature
        # O(1) (log1p(50k ev/px) ~ 10.8 -> ~1.08).
        out[:n, 4] = np.log1p(
            counts_per_event.astype(np.float32)) / 10.0

    mask = np.zeros(e_max, dtype=np.bool_)
    mask[:n] = True
    return out, mask


def _empty(
    e_max: int, *, with_pixel_count: bool = False,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    return (
        np.zeros((e_max, 5 if with_pixel_count else 4), dtype=np.float32),
        np.zeros(e_max, dtype=np.bool_),
    )


__all__ = ["DEFAULT_E_MAX", "tokenize_events_in_track"]
