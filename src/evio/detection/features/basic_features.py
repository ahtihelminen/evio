import numpy as np
from numpy.typing import NDArray

from evio.core.events import EventsBatch
from evio.detection.grid import GridConfig


def cell_event_count(
    events: EventsBatch, grid: GridConfig
) -> NDArray[np.int32]:
    cell_ids = grid.cell_id(events.x_coords, events.y_coords)
    return np.bincount(cell_ids, minlength=grid.cells).astype(
        np.int32, copy=False
    )


def cell_active_pixels(
    events: EventsBatch, grid: GridConfig
) -> NDArray[np.int32]:
    # Count unique (cell, pixel-within-cell) pairs
    cell_ids = grid.cell_id(events.x_coords, events.y_coords).astype(
        np.int64, copy=False
    )

    cell_area = int(grid.cell_w) * int(grid.cell_h)
    local_x = (events.x_coords % grid.cell_w).astype(np.int64, copy=False)
    local_y = (events.y_coords % grid.cell_h).astype(np.int64, copy=False)
    local_pixel_id = local_y * int(grid.cell_w) + local_x

    pixel_in_cell_id = cell_ids * cell_area + local_pixel_id
    unique_pixels = np.unique(pixel_in_cell_id)

    unique_cell_ids = (unique_pixels // cell_area).astype(np.int32, copy=False)
    return np.bincount(unique_cell_ids, minlength=grid.cells).astype(
        np.int32, copy=False
    )


def cell_fano_factor(
    events: EventsBatch, grid: GridConfig
) -> NDArray[np.float32]:
    """
    Per-cell Fano factor of per-pixel event counts:
      F = variance(counts) / mean(counts)
    Returns 0 when the mean count is 0.
    """
    if events.x_coords.size == 0:
        return np.zeros(int(grid.cells), dtype=np.float32)

    cell_w = int(grid.cell_w)
    cell_h = int(grid.cell_h)
    cell_area = cell_w * cell_h
    if cell_area <= 0:
        return np.zeros(int(grid.cells), dtype=np.float32)

    cells = int(grid.cells)
    cell_ids = grid.cell_id(events.x_coords, events.y_coords).astype(
        np.int64, copy=False
    )
    local_x = (events.x_coords % cell_w).astype(np.int64, copy=False)
    local_y = (events.y_coords % cell_h).astype(np.int64, copy=False)
    local_pixel_id = local_y * cell_w + local_x

    pixel_in_cell_id = cell_ids * cell_area + local_pixel_id
    counts = np.bincount(pixel_in_cell_id, minlength=cells * cell_area).astype(
        np.float32, copy=False
    )
    counts = counts.reshape(cells, cell_area)

    sum_counts = counts.sum(axis=1, dtype=np.float32)
    mean = sum_counts / np.float32(cell_area)
    sum_sq = (counts * counts).sum(axis=1, dtype=np.float32)
    var = sum_sq / np.float32(cell_area) - mean * mean
    var = np.maximum(var, np.float32(0.0))

    fano = np.ones(cells, dtype=np.float32)
    nz = mean > 0
    fano[nz] = var[nz] / mean[nz]
    return fano


def cell_8_coactivity(
    events: EventsBatch, grid: GridConfig
) -> NDArray[np.int32]:
    """
    Sum of 8-neighborhood active pixels around active pixels within each cell.
    Returns per-cell counts in flattened order (grid.cells,).
    """
    if events.x_coords.size == 0:
        return np.zeros(int(grid.cells), dtype=np.int32)

    cell_w = int(grid.cell_w)
    cell_h = int(grid.cell_h)
    cell_area = cell_w * cell_h

    cell_ids = grid.cell_id(events.x_coords, events.y_coords).astype(
        np.int64, copy=False
    )
    local_x = (events.x_coords % cell_w).astype(np.int64, copy=False)
    local_y = (events.y_coords % cell_h).astype(np.int64, copy=False)
    local_pixel_id = local_y * cell_w + local_x

    pixel_in_cell_id = cell_ids * cell_area + local_pixel_id
    unique_pixels = np.unique(pixel_in_cell_id)

    if unique_pixels.size == 0:
        return np.zeros(int(grid.cells), dtype=np.int32)

    unique_cell_ids = (unique_pixels // cell_area).astype(np.int64, copy=False)
    unique_local_ids = (unique_pixels % cell_area).astype(np.int64, copy=False)
    local_y = (unique_local_ids // cell_w).astype(np.int64, copy=False)
    local_x = (unique_local_ids % cell_w).astype(np.int64, copy=False)

    active = np.zeros((int(grid.cells), cell_h, cell_w), dtype=np.int8)
    active[unique_cell_ids, local_y, local_x] = 1

    p = np.pad(active, ((0, 0), (1, 1), (1, 1)), mode="constant")
    co = (
        p[:, :-2, :-2]
        + p[:, :-2, 1:-1]
        + p[:, :-2, 2:]
        + p[:, 1:-1, :-2]
        + p[:, 1:-1, 2:]
        + p[:, 2:, :-2]
        + p[:, 2:, 1:-1]
        + p[:, 2:, 2:]
    ).astype(np.int32, copy=False)

    return (co * active).sum(axis=(1, 2)).astype(np.int32, copy=False)


def cell_polarity_skewness(
    events: EventsBatch, grid: GridConfig
) -> NDArray[np.float32]:
    """
    (N+ - N-) / (N+ + N-) per cell. 0 when no events.
    """
    pos_mask = events.polarities > 0
    neg_mask = events.polarities < 0

    pos_ids = grid.cell_id(events.x_coords[pos_mask], events.y_coords[pos_mask])
    neg_ids = grid.cell_id(events.x_coords[neg_mask], events.y_coords[neg_mask])

    pos = np.bincount(pos_ids, minlength=grid.cells).astype(
        np.float32, copy=False
    )
    neg = np.bincount(neg_ids, minlength=grid.cells).astype(
        np.float32, copy=False
    )

    denom = pos + neg
    skew = np.zeros(int(grid.cells), dtype=np.float32)
    nz = denom > 0
    skew[nz] = (pos[nz] - neg[nz]) / denom[nz]
    return skew
