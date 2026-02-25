"""Per-class background abnormality scoring with separate sky/not-sky models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from evio.core.events import EventsBatch
from evio.detection.background import microstate_from_events_regions
from evio.detection.background.metrics import update_stats_welford
from evio.detection.grid import GridConfig

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ClassSpecificConfig:
    """Configuration for a single background class."""

    z_threshold: float = 5.0
    update_threshold: float = 5.0
    calibration_windows: int = 10
    count_weight: float = 1.0
    multi_weight: float = 1.0
    coactivity_weight: float = 1.0
    count_std_floor: float = 10.0
    multi_std_floor: float = 10.0

    def __post_init__(self) -> None:
        if self.z_threshold <= 0:
            msg = "z_threshold must be > 0"
            raise ValueError(msg)
        if self.update_threshold <= 0:
            msg = "update_threshold must be > 0"
            raise ValueError(msg)
        if self.calibration_windows < 0:
            msg = "calibration_windows must be >= 0"
            raise ValueError(msg)
        if self.count_weight < 0:
            msg = "count_weight must be >= 0"
            raise ValueError(msg)
        if self.multi_weight < 0:
            msg = "multi_weight must be >= 0"
            raise ValueError(msg)
        if self.coactivity_weight < 0:
            msg = "coactivity_weight must be >= 0"
            raise ValueError(msg)
        if self.count_std_floor <= 0:
            msg = "count_std_floor must be > 0"
            raise ValueError(msg)
        if self.multi_std_floor <= 0:
            msg = "multi_std_floor must be > 0"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DualBackgroundConfig:
    """Configuration for dual background models (sky/not-sky)."""

    sky: ClassSpecificConfig = ClassSpecificConfig(
        z_threshold=3.0,
        update_threshold=3.0,
        calibration_windows=20,
        count_weight=1.5,
        coactivity_weight=0.5,
    )
    not_sky: ClassSpecificConfig = ClassSpecificConfig(
        z_threshold=4.0,
        update_threshold=4.0,
        calibration_windows=10,
        count_weight=1.0,
        coactivity_weight=1.0,
    )
    sub_rows: int = 4
    sub_cols: int = 4
    time_bins: int = 6
    alpha: float = 1.0
    ema_alpha: float | None = 0.05
    min_events: int = 1
    model_update_weight: float = 1.0
    coactivity_std_floor: float = 1.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.sub_rows <= 0:
            msg = "sub_rows must be > 0"
            raise ValueError(msg)
        if self.sub_cols <= 0:
            msg = "sub_cols must be > 0"
            raise ValueError(msg)
        if self.time_bins <= 0:
            msg = "time_bins must be > 0"
            raise ValueError(msg)
        if self.alpha <= 0:
            msg = "alpha must be > 0"
            raise ValueError(msg)
        if self.min_events < 0:
            msg = "min_events must be >= 0"
            raise ValueError(msg)
        if self.model_update_weight <= 0:
            msg = "model_update_weight must be > 0"
            raise ValueError(msg)
        if self.coactivity_std_floor <= 0:
            msg = "coactivity_std_floor must be > 0"
            raise ValueError(msg)
        if self.epsilon <= 0:
            msg = "epsilon must be > 0"
            raise ValueError(msg)


@dataclass(slots=True)
class DualBackgroundResult:
    """Result from dual background scoring."""

    window_index: int
    calibrating: bool
    z_map: FloatArray
    z_count_map: FloatArray
    z_multi_map: FloatArray
    z_coactivity_map: FloatArray
    update_mask: BoolArray
    event_totals: FloatArray


class _PerClassModel:
    """Internal model state for a single background class."""

    def __init__(
        self,
        cell_count: int,
        feature_size: int,
        config: ClassSpecificConfig,
        shared_config: DualBackgroundConfig,
    ) -> None:
        self.config = config
        self.shared = shared_config

        self._counts = np.zeros((cell_count, feature_size), dtype=np.float32)
        self._total = np.zeros(cell_count, dtype=np.float32)
        self._window_counts = np.zeros(cell_count, dtype=np.float32)

        self._probabilities = np.full(
            (cell_count, feature_size), 1.0 / feature_size, dtype=np.float32
        )
        self._lambdas = -np.log(self._probabilities)

        self._count_mean = np.zeros(cell_count, dtype=np.float32)
        self._count_var = np.zeros(cell_count, dtype=np.float32)
        self._count_m2 = np.zeros(cell_count, dtype=np.float32)
        self._count_count = np.zeros(cell_count, dtype=np.int32)

        self._multi_mean = np.zeros(cell_count, dtype=np.float32)
        self._multi_var = np.zeros(cell_count, dtype=np.float32)
        self._multi_m2 = np.zeros(cell_count, dtype=np.float32)
        self._multi_count = np.zeros(cell_count, dtype=np.int32)

        self._coact_mean = np.zeros(cell_count, dtype=np.float32)
        self._coact_var = np.zeros(cell_count, dtype=np.float32)
        self._coact_m2 = np.zeros(cell_count, dtype=np.float32)
        self._coact_count = np.zeros(cell_count, dtype=np.int32)

        self._coactivity_mean = np.zeros(cell_count, dtype=np.float32)
        self._coactivity_count = np.zeros(cell_count, dtype=np.int32)

        self._log_factorial_lookup = np.zeros(1, dtype=np.float32)
        self._windows_seen = 0

    @property
    def calibrating(self) -> bool:
        return self._windows_seen < self.config.calibration_windows

    def update(
        self,
        microstate: FloatArray,
        event_totals: FloatArray,
        coactivity: FloatArray,
        class_mask: BoolArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, BoolArray]:
        """Update model and return z-scores for cells in this class."""
        if not np.any(class_mask):
            z_count = np.zeros_like(event_totals)
            z_multi = np.zeros_like(event_totals)
            z_coact = np.zeros_like(event_totals)
            z_flat = np.zeros_like(event_totals)
            update_mask = np.zeros_like(class_mask)
            return z_count, z_multi, z_coact, z_flat, update_mask

        max_count = int(microstate.max(initial=0))
        max_total = int(event_totals.max(initial=0))
        self._ensure_log_factorials(max(max_count, max_total))

        log_factorial_total = self._log_factorial_lookup[
            event_totals.astype(np.int32, copy=False)
        ]
        log_factorial_microstate_sum = self._log_factorial_lookup[
            microstate.astype(np.int32, copy=False)
        ].sum(axis=1, dtype=np.float32)

        count_energy = self._score_count_energy(
            event_totals, log_factorial_total
        )
        multi_energy = self._score_multi_energy(
            microstate, log_factorial_total, log_factorial_microstate_sum
        )
        coactivity_term = self._coactivity_term(coactivity)

        count_energy_w = np.float32(self.config.count_weight) * count_energy
        multi_energy_w = np.float32(self.config.multi_weight) * multi_energy
        coactivity_energy_w = (
            np.float32(self.config.coactivity_weight) * coactivity_term
        )

        if self.calibrating:
            self._update_calibration(
                microstate,
                event_totals,
                coactivity,
                class_mask,
                count_energy_w,
                multi_energy_w,
                coactivity_energy_w,
            )
            z_count = np.zeros_like(event_totals)
            z_multi = np.zeros_like(event_totals)
            z_coact = np.zeros_like(event_totals)
            z_flat = np.zeros_like(event_totals)
            update_mask = class_mask.copy()
        else:
            (
                z_count,
                z_multi,
                z_coact,
                z_flat,
                update_mask,
            ) = self._update_running(
                microstate,
                event_totals,
                coactivity,
                class_mask,
                count_energy_w,
                multi_energy_w,
                coactivity_energy_w,
            )

        self._windows_seen += 1
        return z_count, z_multi, z_coact, z_flat, update_mask

    def _update_calibration(
        self,
        microstate: FloatArray,
        event_totals: FloatArray,
        coactivity: FloatArray,
        class_mask: BoolArray,
        count_energy: FloatArray,
        multi_energy: FloatArray,
        coactivity_energy: FloatArray,
    ) -> None:
        update_weight = np.float32(self.shared.model_update_weight)
        active = class_mask

        self._counts[active] += update_weight * microstate[active]
        self._total[active] += update_weight * event_totals[active]
        self._window_counts[active] += update_weight

        counts = self._counts[active]
        total = self._total[active][:, None]
        probabilities = (counts + self.shared.alpha) / (
            total + self.shared.alpha * self._counts.shape[1]
        )
        self._probabilities[active] = probabilities
        self._lambdas[active] = -np.log(probabilities)

        stats_mask = active & (event_totals >= self.shared.min_events)
        self._update_coactivity_mean(coactivity, stats_mask)
        self._update_score_stats(
            count_energy, multi_energy, coactivity_energy, stats_mask
        )

    def _update_running(
        self,
        microstate: FloatArray,
        event_totals: FloatArray,
        coactivity: FloatArray,
        class_mask: BoolArray,
        count_energy: FloatArray,
        multi_energy: FloatArray,
        coactivity_energy: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, BoolArray]:
        valid_count = (
            class_mask
            & (event_totals >= self.shared.min_events)
            & (self._count_count >= 2)
        )
        valid_multi = (
            class_mask
            & (event_totals >= self.shared.min_events)
            & (self._multi_count >= 2)
        )
        valid_coact = (
            class_mask
            & (event_totals >= self.shared.min_events)
            & (self._coact_count >= 2)
        )

        z_count = np.zeros_like(event_totals)
        z_multi = np.zeros_like(event_totals)
        z_coact = np.zeros_like(event_totals)

        if np.any(valid_count):
            std_count = np.maximum(
                np.sqrt(self._count_var[valid_count]),
                np.float32(self.config.count_std_floor),
            )
            z_count[valid_count] = (
                count_energy[valid_count] - self._count_mean[valid_count]
            ) / (std_count + self.shared.epsilon)

        if np.any(valid_multi):
            std_multi = np.maximum(
                np.sqrt(self._multi_var[valid_multi]),
                np.float32(self.config.multi_std_floor),
            )
            z_multi[valid_multi] = (
                multi_energy[valid_multi] - self._multi_mean[valid_multi]
            ) / (std_multi + self.shared.epsilon)

        if np.any(valid_coact):
            std_coact = np.maximum(
                np.sqrt(self._coact_var[valid_coact]),
                np.float32(self.shared.coactivity_std_floor),
            )
            z_coact[valid_coact] = (
                coactivity_energy[valid_coact] - self._coact_mean[valid_coact]
            ) / (std_coact + self.shared.epsilon)

        z_flat = z_count + z_multi + z_coact

        update_mask = class_mask & (
            (~valid_count) | (z_flat <= self.config.update_threshold)
        )

        if np.any(update_mask):
            update_weight = np.float32(self.shared.model_update_weight)
            self._counts[update_mask] += update_weight * microstate[update_mask]
            self._total[update_mask] += (
                update_weight * event_totals[update_mask]
            )
            self._window_counts[update_mask] += update_weight
            counts = self._counts[update_mask]
            total = self._total[update_mask][:, None]
            probabilities = (counts + self.shared.alpha) / (
                total + self.shared.alpha * self._counts.shape[1]
            )
            self._lambdas[update_mask] = -np.log(probabilities)

        stats_mask = update_mask & (event_totals >= self.shared.min_events)
        self._update_coactivity_mean(coactivity, stats_mask)
        self._update_score_stats(
            count_energy, multi_energy, coactivity_energy, stats_mask
        )

        return z_count, z_multi, z_coact, z_flat, update_mask

    def _ensure_log_factorials(self, max_value: int) -> None:
        current_max = self._log_factorial_lookup.size - 1
        if max_value <= current_max:
            return
        start = current_max + 1
        values = np.arange(start, max_value + 1, dtype=np.float64)
        increments = np.log(values)
        cumulative = np.cumsum(increments, dtype=np.float64)
        base = float(self._log_factorial_lookup[current_max])
        new_values = (base + cumulative).astype(np.float32)
        self._log_factorial_lookup = np.concatenate(
            (self._log_factorial_lookup, new_values)
        )

    def _score_count_energy(
        self,
        event_totals: FloatArray,
        log_factorial_total: FloatArray,
    ) -> FloatArray:
        rates = np.empty_like(event_totals)
        seen_mask = self._window_counts > 0
        unseen_mask = ~seen_mask

        if np.any(seen_mask):
            rates[seen_mask] = self._total[seen_mask] / self._window_counts[
                seen_mask
            ].astype(np.float32)
        if np.any(unseen_mask):
            rates[unseen_mask] = np.maximum(
                event_totals[unseen_mask], np.float32(1.0)
            )
        rates = np.maximum(rates, np.float32(self.shared.epsilon))

        return rates - event_totals * np.log(rates) + log_factorial_total

    def _score_multi_energy(
        self,
        microstate: FloatArray,
        log_factorial_total: FloatArray,
        log_factorial_microstate_sum: FloatArray,
    ) -> FloatArray:
        multinomial_energy = (microstate * self._lambdas).sum(axis=1)
        return (
            -log_factorial_total
            + log_factorial_microstate_sum
            + multinomial_energy
        )

    def _coactivity_term(self, coactivity: FloatArray) -> FloatArray:
        coactivity_scale = np.maximum(self._coactivity_mean, np.float32(1.0))
        return (coactivity / (coactivity_scale + self.shared.epsilon)).astype(
            np.float32, copy=False
        )

    def _update_coactivity_mean(
        self, coactivity: FloatArray, mask: BoolArray
    ) -> None:
        if not np.any(mask):
            return
        if self.shared.ema_alpha is None:
            self._coactivity_count[mask] += 1
            delta = coactivity[mask] - self._coactivity_mean[mask]
            self._coactivity_mean[mask] += delta / self._coactivity_count[mask]
            return
        self._coactivity_mean[mask] = (
            1.0 - float(self.shared.ema_alpha)
        ) * self._coactivity_mean[mask] + float(self.shared.ema_alpha) * (
            coactivity[mask]
        )

    def _update_score_stats(
        self,
        count_energy: FloatArray,
        multi_energy: FloatArray,
        coactivity_term: FloatArray,
        mask: BoolArray,
    ) -> None:
        update_stats_welford(
            count_energy,
            mask,
            self._count_mean,
            self._count_var,
            self._count_m2,
            self._count_count,
        )
        update_stats_welford(
            multi_energy,
            mask,
            self._multi_mean,
            self._multi_var,
            self._multi_m2,
            self._multi_count,
        )
        update_stats_welford(
            coactivity_term,
            mask,
            self._coact_mean,
            self._coact_var,
            self._coact_m2,
            self._coact_count,
        )


class DualBackgroundScorer:
    """Per-class background abnormality scoring with sky/not-sky models."""

    def __init__(self, grid: GridConfig, config: DualBackgroundConfig) -> None:
        self.grid = grid
        self.config = config
        self._window_index = 0

        feature_size = config.sub_rows * config.sub_cols * config.time_bins
        cell_count = grid.cells

        self._feature_size = feature_size
        self._cell_count = cell_count

        self._sky_model = _PerClassModel(
            cell_count, feature_size, config.sky, config
        )
        self._not_sky_model = _PerClassModel(
            cell_count, feature_size, config.not_sky, config
        )

    @property
    def window_index(self) -> int:
        return self._window_index

    def update(
        self,
        batch: EventsBatch,
        class_map: NDArray[np.int32],
    ) -> DualBackgroundResult:
        """Update background models and return per-class z-scores."""
        microstate = self._compute_microstate(batch)
        microstate_counts = microstate.astype(np.int32, copy=False)
        event_totals = microstate_counts.sum(axis=1, dtype=np.float32)

        coactivity = self._compute_coactivity(batch)

        class_flat = class_map.reshape(self._cell_count)
        sky_mask = class_flat == 0
        not_sky_mask = ~sky_mask

        (
            z_count_sky,
            z_multi_sky,
            z_coact_sky,
            z_sky,
            update_sky,
        ) = self._sky_model.update(
            microstate, event_totals, coactivity, sky_mask
        )

        (
            z_count_not_sky,
            z_multi_not_sky,
            z_coact_not_sky,
            z_not_sky,
            update_not_sky,
        ) = self._not_sky_model.update(
            microstate, event_totals, coactivity, not_sky_mask
        )

        z_count = np.where(sky_mask, z_count_sky, z_count_not_sky)
        z_multi = np.where(sky_mask, z_multi_sky, z_multi_not_sky)
        z_coact = np.where(sky_mask, z_coact_sky, z_coact_not_sky)
        z_flat = np.where(sky_mask, z_sky, z_not_sky)
        update_mask = sky_mask & update_sky | not_sky_mask & update_not_sky

        calibrating = (
            self._sky_model.calibrating or self._not_sky_model.calibrating
        )

        result = DualBackgroundResult(
            window_index=self._window_index,
            calibrating=calibrating,
            z_map=z_flat.reshape(self.grid.rows, self.grid.cols),
            z_count_map=z_count.reshape(self.grid.rows, self.grid.cols),
            z_multi_map=z_multi.reshape(self.grid.rows, self.grid.cols),
            z_coactivity_map=z_coact.reshape(self.grid.rows, self.grid.cols),
            update_mask=update_mask.reshape(self.grid.rows, self.grid.cols),
            event_totals=event_totals.reshape(self.grid.rows, self.grid.cols),
        )
        self._window_index += 1
        return result

    def _compute_microstate(self, batch: EventsBatch) -> FloatArray:
        microstate_flat = microstate_from_events_regions(
            batch,
            self.grid,
            sub_rows=self.config.sub_rows,
            sub_cols=self.config.sub_cols,
            time_bins=self.config.time_bins,
            include_polarity=False,
        )
        microstate = microstate_flat.reshape(
            self._cell_count, self._feature_size
        )
        return microstate.astype(np.float32, copy=False)

    def _compute_coactivity(self, batch: EventsBatch) -> FloatArray:
        """Compute pixel-wise coactivity for each cell."""
        if batch.timestamps.size == 0:
            return np.zeros(self._cell_count, dtype=np.float32)

        cell_w = self.grid.cell_w
        cell_h = self.grid.cell_h
        features_per_region = cell_h * cell_w

        cell_ids = self.grid.cell_id(batch.x_coords, batch.y_coords).astype(
            np.int64, copy=False
        )
        region_rows = cell_ids // self.grid.cols
        region_cols = cell_ids % self.grid.cols

        local_x = (
            batch.x_coords.astype(np.int32, copy=False) - region_cols * cell_w
        )
        local_y = (
            batch.y_coords.astype(np.int32, copy=False) - region_rows * cell_h
        )
        local_x = np.clip(local_x, 0, cell_w - 1)
        local_y = np.clip(local_y, 0, cell_h - 1)

        pixel_id = local_y.astype(np.int64) * cell_w + local_x.astype(np.int64)
        composite = cell_ids * features_per_region + pixel_id
        total_bins = self._cell_count * features_per_region

        pixel_hits = np.bincount(composite, minlength=total_bins)
        active = (
            (pixel_hits > 0)
            .astype(np.int8, copy=False)
            .reshape(
                self._cell_count,
                cell_h,
                cell_w,
            )
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
