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
        return _empty(e_max)

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

    in_box = (
        (batch.x_coords >= x0u) & (batch.x_coords < x1u)
        & (batch.y_coords >= y0u) & (batch.y_coords < y1u)
    )
    in_time = (
        (batch.timestamps >= t_min) & (batch.timestamps < t_max)
    )
    indices = np.where(in_box & in_time)[0]
    if indices.size == 0:
        return _empty(e_max)

    if indices.size > e_max:
        # Uniform-stride subsample preserves temporal distribution
        # better than random — keeps the rotor-frequency periodicity
        # legible to the attention layer.
        step = indices.size / e_max
        keep = (np.arange(e_max) * step).astype(np.int64)
        indices = indices[keep]

    out = np.zeros((e_max, 4), dtype=np.float32)
    n = int(indices.size)
    ts = batch.timestamps[indices].astype(np.float32)
    xs = batch.x_coords[indices].astype(np.float32)
    ys = batch.y_coords[indices].astype(np.float32)
    ps = (batch.polarities[indices] > 0).astype(np.float32) * 2.0 - 1.0
    out[:n, 0] = (ts - t_min) / span
    out[:n, 1] = (xs - x0u) / width
    out[:n, 2] = (ys - y0u) / height
    out[:n, 3] = ps

    mask = np.zeros(e_max, dtype=np.bool_)
    mask[:n] = True
    return out, mask


def _empty(e_max: int) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    return (
        np.zeros((e_max, 4), dtype=np.float32),
        np.zeros(e_max, dtype=np.bool_),
    )


__all__ = ["DEFAULT_E_MAX", "tokenize_events_in_track"]
