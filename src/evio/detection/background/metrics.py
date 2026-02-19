import numpy as np

from evio.core.events import EventsBatch
from evio.detection.grid import GridConfig


def update_stats_ema(
    values: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    count: np.ndarray,
    alpha: float,
) -> None:
    if not np.any(mask):
        return
    delta = values[mask] - mean[mask]
    mean[mask] += alpha * delta
    var[mask] = (1.0 - alpha) * var[mask] + alpha * (delta * delta)
    count[mask] += 1


def update_stats_welford(
    values: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    m2: np.ndarray,
    count: np.ndarray,
) -> None:
    if not np.any(mask):
        return
    count[mask] += 1
    delta = values[mask] - mean[mask]
    mean[mask] += delta / count[mask]
    delta2 = values[mask] - mean[mask]
    m2[mask] += delta * delta2
    var_vals = np.zeros_like(mean[mask])
    np.divide(
        m2[mask],
        count[mask] - 1,
        out=var_vals,
        where=count[mask] > 1,
    )
    var[mask] = var_vals


def compute_coactivity(
    microstate: np.ndarray,
    *,
    sub_rows: int,
    sub_cols: int,
    time_bins: int,
) -> np.ndarray:
    """
    Binary spatial coactivity with time collapsed.

    For each region:
    1) Collapse time bins and mark each subcell as active/inactive.
    2) Sum 8-neighbor active flags around each active subcell.
    3) Accumulate neighbor counts over active subcells.
    """
    cells = microstate.shape[0]
    active = (
        microstate.reshape(cells, sub_rows, sub_cols, time_bins).sum(axis=3)
        > 0
    )
    active_i = active.astype(np.int8, copy=False)
    padded = np.pad(active_i, ((0, 0), (1, 1), (1, 1)), mode="constant")
    neighbors = (
        padded[:, :-2, :-2]
        + padded[:, :-2, 1:-1]
        + padded[:, :-2, 2:]
        + padded[:, 1:-1, :-2]
        + padded[:, 1:-1, 2:]
        + padded[:, 2:, :-2]
        + padded[:, 2:, 1:-1]
        + padded[:, 2:, 2:]
    )
    return (neighbors * active_i).sum(axis=(1, 2)).astype(np.float32)


def compute_coactivity_pixels(
    events: EventsBatch,
    *,
    region_grid: GridConfig,
) -> np.ndarray:
    """
    Pixel-wise binary coactivity with time collapsed.

    For each region:
    1) Mark each pixel active if it fired at least once in the window.
    2) Sum 8-neighbor active flags around each active pixel.
    3) Accumulate neighbor counts over active pixels.
    """
    cell_count = region_grid.cells
    if events.timestamps.size == 0:
        return np.zeros(cell_count, dtype=np.float32)

    cell_w = region_grid.cell_w
    cell_h = region_grid.cell_h
    features_per_region = cell_h * cell_w

    cell_ids = region_grid.cell_id(events.x_coords, events.y_coords).astype(
        np.int64,
        copy=False,
    )
    region_rows = cell_ids // region_grid.cols
    region_cols = cell_ids % region_grid.cols

    local_x = (
        events.x_coords.astype(np.int32, copy=False) - region_cols * cell_w
    )
    local_y = (
        events.y_coords.astype(np.int32, copy=False) - region_rows * cell_h
    )
    local_x = np.clip(local_x, 0, cell_w - 1)
    local_y = np.clip(local_y, 0, cell_h - 1)

    pixel_id = (local_y.astype(np.int64) * cell_w) + local_x.astype(np.int64)
    composite = cell_ids * features_per_region + pixel_id
    total_bins = cell_count * features_per_region

    pixel_hits = np.bincount(composite, minlength=total_bins)
    active = (pixel_hits > 0).astype(np.int8, copy=False).reshape(
        cell_count,
        cell_h,
        cell_w,
    )

    padded = np.pad(active, ((0, 0), (1, 1), (1, 1)), mode="constant")
    neighbors = (
        padded[:, :-2, :-2]
        + padded[:, :-2, 1:-1]
        + padded[:, :-2, 2:]
        + padded[:, 1:-1, :-2]
        + padded[:, 1:-1, 2:]
        + padded[:, 2:, :-2]
        + padded[:, 2:, 1:-1]
        + padded[:, 2:, 2:]
    )
    return (neighbors * active).sum(axis=(1, 2)).astype(np.float32)
