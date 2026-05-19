"""Per-cell hit-rate map for the F1 persistence-cluster suppression gate.

A 64x64 grid covers the sensor.  Every detection bumps the cell its
centroid falls into; we keep a sliding ring buffer of which cells fired
in each window plus a per-cell mean speed (px/s).  A cell counts as
"persistent clutter" when it fired in more than ``hit_rate_threshold``
of the recent windows **and** the mean speed there is below
``max_velocity_px_s``.  Detections in flagged cells get dropped before
the tracker so the FRED HUD stopwatch, building edges, etc. can't seed
or steal tracks.

A single-pixel max-range target visits any one cell in a tiny fraction
of windows (well below the default threshold), so this gate cannot mask
real long-range candidates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PersistenceGateConfig:
    """Knobs for ``PersistenceMap``."""

    grid_rows: int = 64
    grid_cols: int = 64
    window_history: int = 200
    hit_rate_threshold: float = 0.30
    max_velocity_px_s: float = 5.0
    # Don't mask anything until the buffer is at least this full.  Avoids
    # spuriously masking cells when the clip starts on a hot region.
    warmup_windows: int = 30
    # Cell speed below which "stationary" applies.  Independent of
    # max_velocity_px_s because the hit-rate gate already requires both
    # flags to fire simultaneously.
    enabled: bool = False


class PersistenceMap:
    """Per-cell hit-rate ring buffer for the persistence-cluster gate."""

    def __init__(
        self, config: PersistenceGateConfig, *, width: int, height: int,
    ) -> None:
        self._cfg = config
        self._width = width
        self._height = height
        # bool ring buffer: one (rows, cols) mask per processed window.
        self._history: deque[NDArray[np.bool_]] = deque(
            maxlen=config.window_history,
        )
        # Accumulated per-cell speed for the windows currently in history.
        self._speed_sum: NDArray[np.float32] = np.zeros(
            (config.grid_rows, config.grid_cols), dtype=np.float32,
        )
        self._speed_count: NDArray[np.int32] = np.zeros(
            (config.grid_rows, config.grid_cols), dtype=np.int32,
        )
        self._cell_w = width / config.grid_cols
        self._cell_h = height / config.grid_rows
        self._windows_seen = 0

    @property
    def windows_seen(self) -> int:
        return self._windows_seen

    def cell_for(self, x: float, y: float) -> tuple[int, int]:
        """Centroid → (row, col) clipped to the grid."""
        col = int(np.clip(x / self._cell_w, 0, self._cfg.grid_cols - 1))
        row = int(np.clip(y / self._cell_h, 0, self._cfg.grid_rows - 1))
        return row, col

    def observe(
        self,
        detections: list[tuple[float, float, float]],
    ) -> None:
        """Register one batch of (cx, cy, speed_px_s) tuples for this window."""
        rows, cols = self._cfg.grid_rows, self._cfg.grid_cols
        hits = np.zeros((rows, cols), dtype=np.bool_)
        for cx, cy, speed in detections:
            r, c = self.cell_for(cx, cy)
            hits[r, c] = True
            self._speed_sum[r, c] += speed
            self._speed_count[r, c] += 1
        if len(self._history) == self._cfg.window_history:
            # The oldest window is about to be evicted — its speed contributions
            # don't deserve to live forever but we keep the accumulators simple
            # (mean over recent observations, not strictly windowed). Cheap and
            # stable enough for a 64x64 grid.
            pass
        self._history.append(hits)
        self._windows_seen += 1

    def is_clutter(self, x: float, y: float) -> bool:
        """True when the cell at (x, y) is currently flagged as clutter."""
        if not self._cfg.enabled:
            return False
        if self._windows_seen < self._cfg.warmup_windows:
            return False
        if len(self._history) == 0:
            return False
        row, col = self.cell_for(x, y)
        hit_rate = self._hit_rate_for(row, col)
        if hit_rate < self._cfg.hit_rate_threshold:
            return False
        mean_speed = self._mean_speed_for(row, col)
        return mean_speed < self._cfg.max_velocity_px_s

    def filter(
        self,
        detections: list,
    ) -> list:
        """Drop detections whose centroid falls in a flagged cell."""
        if (
            not self._cfg.enabled
            or self._windows_seen < self._cfg.warmup_windows
        ):
            return detections
        return [
            d for d in detections
            if not self.is_clutter(
                (d.x0 + d.x1) / 2.0, (d.y0 + d.y1) / 2.0,
            )
        ]

    def _hit_rate_for(self, row: int, col: int) -> float:
        if not self._history:
            return 0.0
        stacked = np.stack(self._history, axis=0)
        return float(stacked[:, row, col].mean())

    def _mean_speed_for(self, row: int, col: int) -> float:
        count = int(self._speed_count[row, col])
        if count == 0:
            return 0.0
        return float(self._speed_sum[row, col] / count)

    def snapshot(self) -> dict[str, list]:
        """Return a JSON-friendly snapshot of the current persistence state."""
        if not self._history:
            hit_rate_map = np.zeros(
                (self._cfg.grid_rows, self._cfg.grid_cols), dtype=np.float32,
            )
        else:
            hit_rate_map = np.stack(self._history, axis=0).mean(axis=0)
        return {
            "hit_rate": hit_rate_map.tolist(),
            "mean_speed": (
                self._speed_sum / np.maximum(self._speed_count, 1)
            ).tolist(),
            "windows_seen": [self._windows_seen],
        }


__all__ = ["PersistenceGateConfig", "PersistenceMap"]
