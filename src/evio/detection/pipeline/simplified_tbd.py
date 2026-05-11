"""Simplified track-before-detect stage with integrated candidate detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from evio.core.events import EventsBatch
from evio.detection.grid import GridConfig
from evio.detection.types import Detection

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class SimplifiedTbdConfig:
    """Configuration for simplified TBD stage."""

    integration_window_ms: float = 400.0
    roi_margin_px: int = 8
    velocity_min_px_s: float = -180.0
    velocity_max_px_s: float = 180.0
    velocity_step_px_s: float = 45.0
    min_events_per_roi: int = 30
    min_hist_side_px: int = 8
    baseline_warmup_samples: int = 50
    baseline_update_threshold: float = 4.0
    # Z-score threshold for validation (used instead of PSR-based coherence)
    # Higher values = fewer false positives but may miss weak detections
    # GT detections typically have z_scores 100-300, noise has 10-50
    coherence_threshold: float = 50.0
    sky_coherence_threshold: float = 30.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.integration_window_ms <= 0.0:
            msg = "integration_window_ms must be > 0"
            raise ValueError(msg)
        if self.velocity_step_px_s <= 0.0:
            msg = "velocity_step_px_s must be > 0"
            raise ValueError(msg)
        if self.velocity_max_px_s <= self.velocity_min_px_s:
            msg = "velocity_max_px_s must be > velocity_min_px_s"
            raise ValueError(msg)
        if self.min_events_per_roi <= 0:
            msg = "min_events_per_roi must be > 0"
            raise ValueError(msg)
        if self.min_hist_side_px <= 0:
            msg = "min_hist_side_px must be > 0"
            raise ValueError(msg)
        if self.baseline_warmup_samples < 0:
            msg = "baseline_warmup_samples must be >= 0"
            raise ValueError(msg)
        if self.epsilon <= 0.0:
            msg = "epsilon must be > 0"
            raise ValueError(msg)


@dataclass(slots=True)
class SimplifiedTbdResult:
    """Result from simplified TBD stage."""

    window_index: int
    integration_span_us: int
    buffered_event_count: int
    baseline_mean: float
    baseline_var: float
    baseline_count: int
    detections: list[Detection]

    @property
    def max_m_coh(self) -> float:
        if not self.detections:
            return 0.0
        return max(d.m_coh for d in self.detections)


class SimplifiedTbdStage:
    """Simplified TBD stage with integrated candidate detection from z-map."""

    def __init__(
        self, region_grid: GridConfig, config: SimplifiedTbdConfig
    ) -> None:
        self.region_grid = region_grid
        self.config = config
        self._window_index = 0
        self._buffer: deque[EventsBatch] = deque()
        self._integration_window_us = int(config.integration_window_ms * 1000.0)
        self._velocity_values = np.arange(
            config.velocity_min_px_s,
            config.velocity_max_px_s + config.velocity_step_px_s * 0.5,
            config.velocity_step_px_s,
            dtype=np.float32,
        )
        self._baseline_mean = 0.0
        self._baseline_var = 0.0
        self._baseline_m2 = 0.0
        self._baseline_count = 0

    def update(
        self,
        batch: EventsBatch,
        z_map: FloatArray,
        class_map: IntArray,
        sky_threshold: float,
        not_sky_threshold: float,
    ) -> SimplifiedTbdResult:
        """Update TBD stage and return validated detections."""
        self._append_and_prune_buffer(batch)
        buffered_events = self._stack_buffer_events()

        if buffered_events is None:
            result = SimplifiedTbdResult(
                window_index=self._window_index,
                integration_span_us=0,
                buffered_event_count=0,
                baseline_mean=self._baseline_mean,
                baseline_var=self._baseline_var,
                baseline_count=self._baseline_count,
                detections=[],
            )
            self._window_index += 1
            return result

        timestamps, x_coords, y_coords = buffered_events
        if timestamps.size == 0:
            result = SimplifiedTbdResult(
                window_index=self._window_index,
                integration_span_us=0,
                buffered_event_count=0,
                baseline_mean=self._baseline_mean,
                baseline_var=self._baseline_var,
                baseline_count=self._baseline_count,
                detections=[],
            )
            self._window_index += 1
            return result

        t_ref = int(timestamps.max())
        min_ts = t_ref - self._integration_window_us
        in_window = timestamps >= min_ts
        timestamps = timestamps[in_window]
        x_coords = x_coords[in_window]
        y_coords = y_coords[in_window]

        if timestamps.size == 0:
            result = SimplifiedTbdResult(
                window_index=self._window_index,
                integration_span_us=0,
                buffered_event_count=0,
                baseline_mean=self._baseline_mean,
                baseline_var=self._baseline_var,
                baseline_count=self._baseline_count,
                detections=[],
            )
            self._window_index += 1
            return result

        integration_span_us = int(timestamps.max() - timestamps.min())

        candidates = self._find_candidates(
            z_map, class_map, sky_threshold, not_sky_threshold
        )
        detections: list[Detection] = []

        for candidate in candidates:
            detection = self._score_candidate(
                candidate,
                timestamps=timestamps,
                x_coords=x_coords,
                y_coords=y_coords,
                t_ref=t_ref,
                class_map=class_map,
            )
            if detection is not None:
                detections.append(detection)

        result = SimplifiedTbdResult(
            window_index=self._window_index,
            integration_span_us=integration_span_us,
            buffered_event_count=int(timestamps.size),
            baseline_mean=self._baseline_mean,
            baseline_var=self._baseline_var,
            baseline_count=self._baseline_count,
            detections=detections,
        )
        self._window_index += 1
        return result

    def _append_and_prune_buffer(self, batch: EventsBatch) -> None:
        self._buffer.append(batch)
        if batch.timestamps.size == 0:
            return

        latest_ts = int(batch.timestamps[-1])
        min_allowed_ts = latest_ts - self._integration_window_us
        while self._buffer:
            oldest = self._buffer[0]
            if oldest.timestamps.size == 0:
                self._buffer.popleft()
                continue
            if int(oldest.timestamps[-1]) < min_allowed_ts:
                self._buffer.popleft()
                continue
            break

    def _stack_buffer_events(
        self,
    ) -> tuple[NDArray[np.int64], NDArray[np.int16], NDArray[np.int16]] | None:
        non_empty = [
            batch for batch in self._buffer if batch.timestamps.size > 0
        ]
        if not non_empty:
            return None

        timestamps = np.concatenate([batch.timestamps for batch in non_empty])
        x_coords = np.concatenate([batch.x_coords for batch in non_empty])
        y_coords = np.concatenate([batch.y_coords for batch in non_empty])
        return timestamps, x_coords, y_coords

    def _find_candidates(
        self,
        z_map: FloatArray,
        class_map: IntArray,
        sky_threshold: float,
        not_sky_threshold: float,
    ) -> list[tuple[int, int, int, int, int, float]]:
        """Find candidate regions from z-map threshold crossings.

        Returns list of (row_start, row_end, col_start, col_end, bg_class,
        z_score).
        """
        rows, cols = z_map.shape
        candidates: list[tuple[int, int, int, int, int, float]] = []

        threshold_mask = np.zeros((rows, cols), dtype=bool)
        sky_mask = class_map == 0
        not_sky_mask = ~sky_mask

        threshold_mask[sky_mask] = z_map[sky_mask] >= sky_threshold
        threshold_mask[not_sky_mask] = z_map[not_sky_mask] >= not_sky_threshold

        if not np.any(threshold_mask):
            return candidates

        visited = np.zeros_like(threshold_mask, dtype=bool)

        for row in range(rows):
            for col in range(cols):
                if not threshold_mask[row, col] or visited[row, col]:
                    continue

                component = self._flood_fill(threshold_mask, visited, row, col)
                if len(component) < 1:
                    continue

                comp_rows = [r for r, _ in component]
                comp_cols = [c for _, c in component]

                row_start = min(comp_rows)
                row_end = max(comp_rows) + 1
                col_start = min(comp_cols)
                col_end = max(comp_cols) + 1

                center_row = (row_start + row_end) // 2
                center_col = (col_start + col_end) // 2
                bg_class = int(class_map[center_row, center_col])

                # Use max z-score in the component
                max_z = 0.0
                for r, c in component:
                    z_val = float(z_map[r, c])
                    if z_val > max_z:
                        max_z = z_val

                candidates.append(
                    (row_start, row_end, col_start, col_end, bg_class, max_z)
                )

        return candidates

    def _flood_fill(
        self,
        mask: BoolArray,
        visited: BoolArray,
        start_row: int,
        start_col: int,
    ) -> list[tuple[int, int]]:
        """Flood fill to find connected component."""
        rows, cols = mask.shape
        stack: list[tuple[int, int]] = [(start_row, start_col)]
        visited[start_row, start_col] = True
        component: list[tuple[int, int]] = []

        while stack:
            row, col = stack.pop()
            component.append((row, col))

            for dr, dc in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                nr, nc = row + dr, col + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and not visited[nr, nc]
                    and mask[nr, nc]
                ):
                    visited[nr, nc] = True
                    stack.append((nr, nc))

        return component

    def _score_candidate(
        self,
        candidate: tuple[int, int, int, int, int, float],
        *,
        timestamps: NDArray[np.int64],
        x_coords: NDArray[np.int16],
        y_coords: NDArray[np.int16],
        t_ref: int,
        class_map: IntArray,
    ) -> Detection | None:
        row_start, row_end, col_start, col_end, bg_class, z_score = candidate
        x0, y0, x1, y1 = self._cell_to_pixel_bounds(
            row_start, row_end, col_start, col_end
        )

        roi_mask = (
            (x_coords >= x0)
            & (x_coords < x1)
            & (y_coords >= y0)
            & (y_coords < y1)
        )
        if not np.any(roi_mask):
            return None

        roi_timestamps = timestamps[roi_mask]
        roi_x = x_coords[roi_mask].astype(np.float32, copy=False)
        roi_y = y_coords[roi_mask].astype(np.float32, copy=False)

        if roi_timestamps.size < self.config.min_events_per_roi:
            return None

        best_psr = 0.0
        best_vx = 0.0
        best_vy = 0.0

        roi_width = max(self.config.min_hist_side_px, x1 - x0)
        roi_height = max(self.config.min_hist_side_px, y1 - y0)
        dt_seconds = (
            roi_timestamps.astype(np.float32, copy=False) - np.float32(t_ref)
        ) / np.float32(1_000_000.0)

        for vx in self._velocity_values:
            warped_x = (roi_x - vx * dt_seconds).astype(np.float32, copy=False)
            for vy in self._velocity_values:
                warped_y = (roi_y - vy * dt_seconds).astype(
                    np.float32, copy=False
                )
                psr = self._histogram_psr(
                    warped_x=warped_x,
                    warped_y=warped_y,
                    x0=x0,
                    y0=y0,
                    width=roi_width,
                    height=roi_height,
                )
                if psr > best_psr:
                    best_psr = psr
                    best_vx = float(vx)
                    best_vy = float(vy)

        # Use z-score directly for validation instead of PSR-based coherence
        # The z-score is already computed from the background model and
        # provides a reliable measure of anomaly
        coherence_threshold = (
            self.config.sky_coherence_threshold
            if bg_class == 0
            else self.config.coherence_threshold
        )

        # Use z_score as m_coh for validation
        m_coh = z_score

        if m_coh < coherence_threshold:
            return None

        return Detection(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            vx_px_s=best_vx,
            vy_px_s=best_vy,
            m_coh=float(m_coh),
            best_psr=float(best_psr),
            event_count=int(roi_timestamps.size),
            bg_class=bg_class,
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end,
            z_score=z_score,
        )

    def _cell_to_pixel_bounds(
        self, row_start: int, row_end: int, col_start: int, col_end: int
    ) -> tuple[int, int, int, int]:
        x0 = col_start * self.region_grid.cell_w
        y0 = row_start * self.region_grid.cell_h
        x1 = min(col_end * self.region_grid.cell_w, self.region_grid.width)
        y1 = min(row_end * self.region_grid.cell_h, self.region_grid.height)

        margin = self.config.roi_margin_px
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(self.region_grid.width, x1 + margin)
        y1 = min(self.region_grid.height, y1 + margin)
        return x0, y0, x1, y1

    def _histogram_psr(
        self,
        *,
        warped_x: FloatArray,
        warped_y: FloatArray,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> float:
        local_x = np.rint(warped_x - np.float32(x0)).astype(
            np.int32, copy=False
        )
        local_y = np.rint(warped_y - np.float32(y0)).astype(
            np.int32, copy=False
        )

        valid = (
            (local_x >= 0)
            & (local_x < width)
            & (local_y >= 0)
            & (local_y < height)
        )
        if not np.any(valid):
            return 0.0

        flat_indices = local_y[valid] * width + local_x[valid]
        histogram = np.bincount(flat_indices, minlength=width * height).astype(
            np.float32, copy=False
        )

        mean = float(histogram.mean())
        std = float(histogram.std())
        peak = float(histogram.max())
        return (peak - mean) / (std + self.config.epsilon)

    def _coherence_score(self, best_psr: float) -> float:
        if self._baseline_count < 2:
            return 0.0
        std = float(np.sqrt(max(self._baseline_var, 0.0)))
        return (best_psr - self._baseline_mean) / (std + self.config.epsilon)

    def _update_baseline(self, score: float) -> None:
        self._baseline_count += 1
        delta = score - self._baseline_mean
        self._baseline_mean += delta / self._baseline_count
        delta2 = score - self._baseline_mean
        self._baseline_m2 += delta * delta2
        if self._baseline_count > 1:
            self._baseline_var = self._baseline_m2 / (self._baseline_count - 1)
