from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, types
from numba.typed import Dict
from numpy.typing import NDArray

from evio.core.events import EventsBatch
from evio.detection.features.basic_features import (
    cell_8_coactivity,
    cell_active_pixels,
    cell_event_count,
    cell_fano_factor,
    cell_polarity_skewness,
)
from evio.detection.grid import GridConfig

_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class BgClassesResult:
    calibrating: bool
    class_map: NDArray[np.int32] | None
    calibration_elapsed_us: int
    calibration_target_us: int


class BackgroundClassesClassifier:
    def __init__(
        self,
        *,
        grid: GridConfig,
        calibration_time_us: int,
        random_seed: int = 7,
        n_init: int = 24,
        max_iters: int = 100,
    ) -> None:
        self._grid = grid
        self._calibration_time_us = max(0, int(calibration_time_us))
        self._random_seed = int(random_seed)
        self._n_init = max(1, int(n_init))
        self._max_iters = max(1, int(max_iters))

        self._calibration_start_ts_us: int | None = None
        self._calibration_elapsed_us = 0
        self._calibration_windows: list[NDArray[np.float32]] = []

        self._feature_mean: NDArray[np.float32] | None = None
        self._feature_std: NDArray[np.float32] | None = None
        self._cluster_centers: NDArray[np.float32] | None = None
        self._class_map: NDArray[np.int32] | None = None

        # Trigger compilation up front with a tiny array, the same way
        # NumbaEventRenderer.__init__ does, so JIT cost isn't paid on
        # the first real window inside systemd's startup budget.
        _fused_feature_kernel(
            np.zeros(1, dtype=np.int16),
            np.zeros(1, dtype=np.int16),
            np.zeros(1, dtype=np.int8),
            1,
            1,
            1,
            1,
        )

    def update(self, events: EventsBatch) -> BgClassesResult:
        if self._class_map is not None:
            return BgClassesResult(
                calibrating=False,
                class_map=self._class_map,
                calibration_elapsed_us=self._calibration_elapsed_us,
                calibration_target_us=self._calibration_time_us,
            )

        prepared = _prepare_feature_matrix(
            _compute_feature_matrix(events, self._grid), self._grid
        )
        self._collect_calibration_window(prepared, events)
        if self._calibration_done():
            self._fit_clusters()
        return BgClassesResult(
            calibrating=True,
            class_map=self._class_map,
            calibration_elapsed_us=self._calibration_elapsed_us,
            calibration_target_us=self._calibration_time_us,
        )

    def _collect_calibration_window(
        self,
        prepared_features: NDArray[np.float32],
        events: EventsBatch,
    ) -> None:
        self._calibration_windows.append(prepared_features)

        if events.timestamps.size > 0:
            batch_start_ts_us = int(events.timestamps[0])
            batch_end_ts_us = int(events.timestamps[-1])
            if self._calibration_start_ts_us is None:
                self._calibration_start_ts_us = batch_start_ts_us
            self._calibration_elapsed_us = max(
                self._calibration_elapsed_us,
                batch_end_ts_us - self._calibration_start_ts_us,
            )
            return

        self._calibration_elapsed_us += int(self._grid.window_us)

    def _calibration_done(self) -> bool:
        if not self._calibration_windows:
            return False
        return self._calibration_elapsed_us >= self._calibration_time_us

    def _fit_clusters(self) -> None:
        if not self._calibration_windows:
            return

        calibration_stack = np.stack(
            self._calibration_windows,
            axis=0,
        )
        calibration_per_cell = calibration_stack.mean(axis=0)
        feature_mean = calibration_per_cell.mean(axis=0, keepdims=True)
        feature_std = calibration_per_cell.std(axis=0, keepdims=True)
        feature_std = np.where(feature_std < _EPS, 1.0, feature_std)
        standardized = _standardize_matrix(
            calibration_per_cell,
            feature_mean=feature_mean.astype(np.float32, copy=False),
            feature_std=feature_std.astype(np.float32, copy=False),
        )
        _, centers = _kmeans_two_clusters(
            standardized,
            seed=self._random_seed,
            n_init=self._n_init,
            max_iters=self._max_iters,
        )
        centers = _order_cluster_centers(centers)

        self._feature_mean = feature_mean.astype(np.float32, copy=False)
        self._feature_std = feature_std.astype(np.float32, copy=False)
        self._cluster_centers = centers.astype(np.float32, copy=False)
        self._calibration_windows.clear()

        labels = _assign_clusters(standardized, self._cluster_centers)
        self._class_map = labels.reshape(
            self._grid.rows, self._grid.cols
        ).astype(np.int32)


def _compute_feature_matrix_reference(
    events: EventsBatch,
    grid: GridConfig,
) -> NDArray[np.float32]:
    """Readable numpy reference: five whole-array passes over `events`.

    Kept as the spec `_compute_feature_matrix`'s fused numba kernel is
    checked against — see the equivalence test.
    """
    feature_functions = (
        cell_event_count,
        cell_active_pixels,
        cell_fano_factor,
        cell_8_coactivity,
        cell_polarity_skewness,
    )
    matrix = np.zeros((grid.cells, len(feature_functions)), dtype=np.float32)
    for index, feature_fn in enumerate(feature_functions):
        values = np.asarray(feature_fn(events, grid), dtype=np.float32).reshape(
            grid.cells
        )
        matrix[:, index] = values
    return matrix


@njit(cache=True)
def _fused_feature_kernel(
    x_coords: NDArray[np.int16],
    y_coords: NDArray[np.int16],
    polarities: NDArray[np.int8],
    rows: int,
    cols: int,
    cell_w: int,
    cell_h: int,
) -> NDArray[np.float32]:
    """One pass over events, one pass over the touched-pixel list.

    Computes the same five columns as `_compute_feature_matrix_reference`
    (event count, active pixels, fano factor, 8-coactivity, polarity
    skewness) without materializing a dense per-pixel volume: touched
    (cell, local-pixel) pairs live in a hash map built while walking the
    events once, then a single pass over that sparse map derives the
    per-cell aggregates, using the map itself for the 8-neighbor lookups
    `cell_8_coactivity` needs.
    """
    cells = rows * cols
    cell_area = cell_w * cell_h

    event_count = np.zeros(cells, dtype=np.int64)
    pos_count = np.zeros(cells, dtype=np.int64)
    neg_count = np.zeros(cells, dtype=np.int64)
    touched = Dict.empty(key_type=types.int64, value_type=types.int64)

    for i in range(x_coords.size):
        x_coord = int(x_coords[i])
        y_coord = int(y_coords[i])

        col = x_coord // cell_w
        col = min(max(col, 0), cols - 1)
        row = y_coord // cell_h
        row = min(max(row, 0), rows - 1)
        cell_id = row * cols + col
        event_count[cell_id] += 1

        polarity = int(polarities[i])
        if polarity > 0:
            pos_count[cell_id] += 1
        elif polarity < 0:
            neg_count[cell_id] += 1

        local_x = x_coord % cell_w
        local_y = y_coord % cell_h
        pixel_id = cell_id * cell_area + local_y * cell_w + local_x
        if pixel_id in touched:
            touched[pixel_id] += 1
        else:
            touched[pixel_id] = 1

    active_pixels = np.zeros(cells, dtype=np.int64)
    sum_sq = np.zeros(cells, dtype=np.int64)
    coactivity = np.zeros(cells, dtype=np.int64)

    for pixel_id in touched:
        count = touched[pixel_id]
        cell_id = pixel_id // cell_area
        local_id = pixel_id % cell_area
        local_y = local_id // cell_w
        local_x = local_id % cell_w

        active_pixels[cell_id] += 1
        sum_sq[cell_id] += count * count

        neighbor_sum = 0
        for d_row in (-1, 0, 1):
            n_row = local_y + d_row
            if n_row < 0 or n_row >= cell_h:
                continue
            for d_col in (-1, 0, 1):
                if d_row == 0 and d_col == 0:
                    continue
                n_col = local_x + d_col
                if n_col < 0 or n_col >= cell_w:
                    continue
                neighbor_id = cell_id * cell_area + n_row * cell_w + n_col
                if neighbor_id in touched:
                    neighbor_sum += 1
        coactivity[cell_id] += neighbor_sum

    matrix = np.zeros((cells, 5), dtype=np.float32)
    for cell_id in range(cells):
        count_f = np.float32(event_count[cell_id])
        matrix[cell_id, 0] = count_f
        matrix[cell_id, 1] = np.float32(active_pixels[cell_id])

        mean = count_f / np.float32(cell_area)
        var = np.float32(sum_sq[cell_id]) / np.float32(cell_area)
        var -= mean * mean
        var = max(var, np.float32(0.0))
        fano = np.float32(1.0)
        if mean > np.float32(0.0):
            fano = var / mean
        matrix[cell_id, 2] = fano

        matrix[cell_id, 3] = np.float32(coactivity[cell_id])

        pos_f = np.float32(pos_count[cell_id])
        neg_f = np.float32(neg_count[cell_id])
        denom = pos_f + neg_f
        skew = np.float32(0.0)
        if denom > np.float32(0.0):
            skew = (pos_f - neg_f) / denom
        matrix[cell_id, 4] = skew

    return matrix


def _compute_feature_matrix(
    events: EventsBatch,
    grid: GridConfig,
) -> NDArray[np.float32]:
    """Fused numba path; falls back to the reference on empty batches.

    `_compute_feature_matrix_reference`'s per-function empty-batch
    early-returns are quirky enough (e.g. `cell_fano_factor` returns
    zeros rather than the per-cell default of 1.0) that delegating for
    the rare empty case is simpler than replicating them in the kernel.
    """
    if events.x_coords.size == 0:
        return _compute_feature_matrix_reference(events, grid)
    return _fused_feature_kernel(
        events.x_coords,
        events.y_coords,
        events.polarities,
        grid.rows,
        grid.cols,
        grid.cell_w,
        grid.cell_h,
    )


def _prepare_feature_matrix(
    matrix: NDArray[np.float32],
    grid: GridConfig,
) -> NDArray[np.float32]:
    prepared = matrix.copy()
    prepared[:, 0] = np.log1p(prepared[:, 0])
    prepared[:, 1] = prepared[:, 1] / float(grid.cell_area)
    prepared[:, 3] = prepared[:, 3] / float(grid.cell_area * 8)
    return prepared


def _standardize_matrix(
    matrix: NDArray[np.float32],
    *,
    feature_mean: NDArray[np.float32],
    feature_std: NDArray[np.float32],
) -> NDArray[np.float32]:
    return (matrix - feature_mean) / feature_std


def _kmeans_two_clusters(
    matrix: NDArray[np.float32],
    *,
    seed: int,
    n_init: int,
    max_iters: int,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    if matrix.shape[0] < 2:
        msg = "Need at least two cells for two-cluster k-means."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    best_labels = np.zeros(matrix.shape[0], dtype=np.int32)
    best_centers = np.zeros((2, matrix.shape[1]), dtype=np.float32)
    best_inertia = np.inf

    for _ in range(n_init):
        center_ids = rng.choice(matrix.shape[0], size=2, replace=False)
        centers = matrix[center_ids].copy()
        labels = np.zeros(matrix.shape[0], dtype=np.int32)

        for _ in range(max_iters):
            squared_distances = (
                (matrix[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2
            ).sum(axis=2)
            new_labels = np.argmin(squared_distances, axis=1).astype(np.int32)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

            for cluster_id in range(2):
                members = matrix[labels == cluster_id]
                if members.size == 0:
                    random_index = int(rng.integers(0, matrix.shape[0]))
                    centers[cluster_id] = matrix[random_index]
                else:
                    centers[cluster_id] = members.mean(axis=0)

        inertia = float(((matrix - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers


def _order_cluster_centers(
    centers: NDArray[np.float32],
) -> NDArray[np.float32]:
    if centers.shape[0] != 2:
        return centers
    if centers[0, 0] <= centers[1, 0]:
        return centers
    return centers[[1, 0]].copy()


def _assign_clusters(
    matrix: NDArray[np.float32],
    centers: NDArray[np.float32],
) -> NDArray[np.int32]:
    squared_distances = (
        (matrix[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2
    ).sum(axis=2)
    return np.argmin(squared_distances, axis=1).astype(np.int32)
