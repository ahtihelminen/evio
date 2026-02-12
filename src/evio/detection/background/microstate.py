import numpy as np
from numpy.typing import NDArray

from evio.core.events import EventsBatch
from evio.detection.voxel import GridConfig


def microstate_from_events(
    events: EventsBatch,
    grid: GridConfig,
    *,
    include_polarity: bool = False,
) -> NDArray[np.int32]:
    """
    Build a microstate vector by counting events per (row, col, time_bin).

    The grid partitions the sensor into (rows, cols) cells and the window
    into grid.time_bins time bins. The output is flattened in row-major order.
    """
    rows = grid.rows
    cols = grid.cols
    time_bins = grid.time_bins

    if include_polarity:
        counts = np.zeros((rows, cols, time_bins, 2), dtype=np.int32)
    else:
        counts = np.zeros((rows, cols, time_bins), dtype=np.int32)

    cell_ids = grid.cell_id(events.x_coords, events.y_coords)
    row_indices = (cell_ids // cols).astype(np.int32, copy=False)
    col_indices = (cell_ids % cols).astype(np.int32, copy=False)

    window_start_us = events.timestamps.min()
    time_offsets = (
        events.timestamps.astype(np.int64, copy=False) - window_start_us
    )
    time_indices = (time_offsets // grid.dt_us).astype(np.int32, copy=False)
    time_indices = np.clip(time_indices, 0, time_bins - 1)

    if include_polarity:
        polarity_indices = (events.polarities > 0).astype(np.int32, copy=False)
        np.add.at(
            counts,
            (row_indices, col_indices, time_indices, polarity_indices),
            1,
        )
    else:
        np.add.at(counts, (row_indices, col_indices, time_indices), 1)

    return counts.reshape(-1)


def microstate_from_event_tensor(
    event_tensor: NDArray[np.int8],
    *,
    rows: int,
    cols: int,
    time_bins: int,
    include_polarity: bool = False,
) -> NDArray[np.int32]:
    """
    Build a microstate vector from a dense event tensor of shape (x, y, t).
    """
    width, height, duration = event_tensor.shape
    cell_width = (width + cols - 1) // cols
    cell_height = (height + rows - 1) // rows
    time_bin_width = (duration + time_bins - 1) // time_bins

    event_positions = np.nonzero(event_tensor)
    if len(event_positions[0]) == 0:
        output_shape = (
            (rows, cols, time_bins, 2)
            if include_polarity
            else (
                rows,
                cols,
                time_bins,
            )
        )
        return np.zeros(output_shape, dtype=np.int32).reshape(-1)

    x_coords, y_coords, t_coords = event_positions
    row_indices = np.clip(y_coords // cell_height, 0, rows - 1)
    col_indices = np.clip(x_coords // cell_width, 0, cols - 1)
    time_indices = np.clip(t_coords // time_bin_width, 0, time_bins - 1)

    if include_polarity:
        counts = np.zeros((rows, cols, time_bins, 2), dtype=np.int32)
        polarities = event_tensor[x_coords, y_coords, t_coords]
        polarity_indices = (polarities > 0).astype(np.int32, copy=False)
        np.add.at(
            counts,
            (row_indices, col_indices, time_indices, polarity_indices),
            1,
        )
    else:
        counts = np.zeros((rows, cols, time_bins), dtype=np.int32)
        np.add.at(counts, (row_indices, col_indices, time_indices), 1)

    return counts.reshape(-1)
