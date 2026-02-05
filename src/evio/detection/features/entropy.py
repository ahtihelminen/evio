import numpy as np
from numpy.typing import NDArray

from evio.core.events import EventsBatch
from evio.detection.grid_state import GridConfig


def cell_shannon_entropy(
    events: EventsBatch, grid: GridConfig
) -> NDArray[np.float32]:
    """
    Shannon entropy (base-2) of the spatial event distribution inside each cell.
    For each cell: p_i = count(pixel_i)/sum(counts), H = -sum p_i log2 p_i
    """
    cell_ids = grid.cell_id(events.x_coords, events.y_coords).astype(
        np.int64, copy=False
    )

    cell_area = int(grid.cell_w) * int(grid.cell_h)
    local_x = (events.x_coords % grid.cell_w).astype(np.int64, copy=False)
    local_y = (events.y_coords % grid.cell_h).astype(np.int64, copy=False)
    local_pixel_id = local_y * int(grid.cell_w) + local_x

    pixel_in_cell_id = cell_ids * cell_area + local_pixel_id

    # Per (cell, local_pixel) counts
    pix_counts = np.bincount(
        pixel_in_cell_id, minlength=int(grid.cells) * cell_area
    ).astype(np.float32, copy=False)
    pix_counts = pix_counts.reshape(int(grid.cells), cell_area)

    totals = pix_counts.sum(axis=1)  # (cells,)
    ent = np.zeros(int(grid.cells), dtype=np.float32)

    nonzero = totals > 0
    if np.any(nonzero):
        p = pix_counts[nonzero] / totals[nonzero, None]
        # Avoid log(0) by masking
        mask = p > 0
        ent_nonzero = -(p[mask] * np.log2(p[mask])).reshape(-1)
        # Sum back per-row (cell): do it via row indices
        row_idx = np.repeat(np.arange(p.shape[0], dtype=np.int32), cell_area)[
            mask.ravel()
        ]
        ent_acc = np.bincount(
            row_idx, weights=ent_nonzero, minlength=p.shape[0]
        ).astype(np.float32, copy=False)
        ent[nonzero] = ent_acc

    return ent


def cell_bernoulli_entropy(
    events: EventsBatch, grid_cfg: GridConfig
) -> NDArray[np.float32]:
    """
    Two-state (Bernoulli) entropy per cell (base-2),
    with per-pixel state x∈{0,1}:

      x=1  if pixel fired ≥1 event in a time bin
      x=0  otherwise

    For each (cell, local_pixel), estimate:
      p = (#time_bins where pixel fired) / (#time_bins in this batch)

    Then per-cell entropy is:
      H_cell = sum_over_pixels [ -p log2 p - (1-p) log2 (1-p) ]

    NOTE: This requires a time-bin width in grid_cfg (dt_us).
    """
    if events.x_coords.size == 0:
        return np.zeros(int(grid_cfg.cells), dtype=np.float32)

    # --- time bin width (microseconds) ---
    dt_us = grid_cfg.dt_us
    if dt_us is None or dt_us <= 0:
        msg = "grid_cfg.dt_us must be set to a positive integer"
        raise ValueError(msg)

    # --- cell + local pixel ids ---
    cell_ids = grid_cfg.cell_id(events.x_coords, events.y_coords).astype(
        np.int64, copy=False
    )

    cell_w = int(grid_cfg.cell_w)
    cell_h = int(grid_cfg.cell_h)
    cell_area = cell_w * cell_h

    local_x = (events.x_coords % cell_w).astype(np.int64, copy=False)
    local_y = (events.y_coords % cell_h).astype(np.int64, copy=False)
    local_pixel_id = local_y * cell_w + local_x  # [0, cell_area)

    pixel_in_cell_id = (
        cell_ids * cell_area + local_pixel_id
    )  # [0, cells*cell_area)

    # --- time binning ---
    t = events.timestamps
    t0 = int(t.min())
    # bin index per event (non-negative)
    bin_idx = ((t - t0) // np.int64(dt_us)).astype(np.int64, copy=False)
    n_bins = int(bin_idx.max()) + 1
    if n_bins <= 0:
        return np.zeros(int(grid_cfg.cells), dtype=np.float32)

    # We want: for each (pixel_in_cell_id, bin_idx), mark presence once
    # Build a composite id and unique it.
    # composite = pixel_in_cell_id * n_bins + bin_idx
    composite = pixel_in_cell_id * np.int64(n_bins) + bin_idx
    composite_unique = np.unique(composite)

    # Recover pixel_in_cell_id from composite ids, then count how many bins each
    # pixel fired in.
    pixel_ids_unique = (composite_unique // np.int64(n_bins)).astype(
        np.int64, copy=False
    )

    fired_bins_per_pixel = np.bincount(
        pixel_ids_unique,
        minlength=int(grid_cfg.cells) * cell_area,
    ).astype(np.float32, copy=False)

    p = fired_bins_per_pixel / float(
        n_bins
    )  # firing probability per pixel within this batch

    # Bernoulli entropy per pixel (base-2), safely handling p=0 or 1
    eps = np.float32(1e-6)  # float32-safe near 1.0
    p_clip = np.clip(p, eps, np.float32(1.0) - eps)

    h = -(
        p_clip * np.log2(p_clip)
        + (np.float32(1.0) - p_clip) * np.log2(np.float32(1.0) - p_clip)
    ).astype(np.float32, copy=False)
    h[(p <= 0.0) | (p >= 1.0)] = 0.0

    # Sum per cell
    h = h.reshape(int(grid_cfg.cells), cell_area)
    return h.sum(axis=1).astype(np.float32, copy=False)
