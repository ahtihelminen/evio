from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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

    def update(self, events: EventsBatch) -> BgClassesResult:
        feature_matrix = _compute_feature_matrix(events, self._grid)
        prepared = _prepare_feature_matrix(feature_matrix, self._grid)

        if self._cluster_centers is None:
            self._collect_calibration_window(prepared, events)
            if self._calibration_done():
                self._fit_clusters()
            if self._cluster_centers is None:
                return BgClassesResult(
                    calibrating=True,
                    class_map=None,
                    calibration_elapsed_us=self._calibration_elapsed_us,
                    calibration_target_us=self._calibration_time_us,
                )

        feature_mean = self._feature_mean
        feature_std = self._feature_std
        centers = self._cluster_centers
        if feature_mean is None or feature_std is None or centers is None:
            msg = "Cluster model is not initialized."
            raise RuntimeError(msg)

        standardized = _standardize_matrix(
            prepared,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
        labels = _assign_clusters(standardized, centers)
        class_map = labels.reshape(self._grid.rows, self._grid.cols)
        return BgClassesResult(
            calibrating=False,
            class_map=class_map,
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


def _compute_feature_matrix(
    events: EventsBatch,
    grid: GridConfig,
) -> NDArray[np.float32]:
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
