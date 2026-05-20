"""Track-spawn-density suppressor (Fix 1).

Where ``PersistenceMap`` counts *detections per cell per window*, this
module counts *new track_id spawns per cell over a rolling time window*.

A real drone produces one spawn per visit and then keeps its track for
the entire flight; rain bursts, snow, foliage motion, and sensor noise
produce many short-lived spawn/die cycles in the same region. A region
emitting more than ``max_spawns`` new track_ids in ``window_s`` is
almost certainly a noise source and should be suppressed.

The gate runs *before* the tracker creates the new track. It does not
touch existing confirmed tracks passing through the region — those
keep flowing as long as Mahalanobis matching keeps them alive.

The suppressor does NOT affect single-pixel max-range candidates by
construction: the spawn-count threshold is independent of event count
or m_coh, and a far-range drone that briefly visits a noisy region
trips the gate just like a raindrop — but that's a known accepted
trade-off; the user can set ``max_spawns`` high (or disable per-cell)
if max-range PD takes priority over noise rejection.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SpawnDensityConfig:
    """Knobs for ``SpawnDensityMap``."""

    grid_rows: int = 64
    grid_cols: int = 64
    window_s: float = 5.0      # rolling time window for spawn count
    max_spawns: int = 4        # >this many spawns in `window_s` → suppress
    cooldown_s: float = 5.0    # how long a flagged cell stays suppressed
    enabled: bool = False


class SpawnDensityMap:
    """Per-cell spawn-rate counter + suppression cooldown."""

    def __init__(
        self, config: SpawnDensityConfig, *, width: int, height: int,
    ) -> None:
        self._cfg = config
        self._cell_w = width / config.grid_cols
        self._cell_h = height / config.grid_rows
        # Per cell, a deque of recent spawn timestamps (microseconds).
        self._spawn_times: dict[tuple[int, int], deque[int]] = {}
        # Per cell, the timestamp until which suppression remains active.
        self._suppress_until_us: NDArray[np.int64] = np.zeros(
            (config.grid_rows, config.grid_cols), dtype=np.int64,
        )
        self._total_suppressed = 0

    @property
    def total_suppressed(self) -> int:
        return self._total_suppressed

    def cell_for(self, x: float, y: float) -> tuple[int, int]:
        col = int(np.clip(x / self._cell_w, 0, self._cfg.grid_cols - 1))
        row = int(np.clip(y / self._cell_h, 0, self._cfg.grid_rows - 1))
        return row, col

    def should_suppress(
        self, x: float, y: float, now_us: int,
    ) -> bool:
        """True when (x, y)'s cell is over the spawn-rate threshold."""
        if not self._cfg.enabled:
            return False
        row, col = self.cell_for(x, y)
        if int(self._suppress_until_us[row, col]) > now_us:
            self._total_suppressed += 1
            return True
        # Check rolling spawn count for this cell.
        deque_for_cell = self._spawn_times.get((row, col))
        if deque_for_cell is None:
            return False
        window_us = int(self._cfg.window_s * 1_000_000)
        threshold = now_us - window_us
        while deque_for_cell and deque_for_cell[0] < threshold:
            deque_for_cell.popleft()
        if len(deque_for_cell) >= self._cfg.max_spawns:
            cooldown_us = int(self._cfg.cooldown_s * 1_000_000)
            self._suppress_until_us[row, col] = now_us + cooldown_us
            self._total_suppressed += 1
            return True
        return False

    def register_spawn(self, x: float, y: float, now_us: int) -> None:
        """Record that a new track_id was just created at (x, y)."""
        row, col = self.cell_for(x, y)
        deque_for_cell = self._spawn_times.setdefault(
            (row, col), deque(),
        )
        deque_for_cell.append(now_us)
        # Bound the deque so a memory leak doesn't develop on long runs.
        max_len = self._cfg.max_spawns * 4
        while len(deque_for_cell) > max_len:
            deque_for_cell.popleft()

    def suppressed_mask(self, now_us: int) -> NDArray[np.bool_]:
        """Return a (rows, cols) bool mask of currently-suppressed cells."""
        return self._suppress_until_us > now_us

    def snapshot(self, now_us: int) -> dict[str, list]:
        """JSON-friendly snapshot for overlays / debugging."""
        mask = self.suppressed_mask(now_us)
        spawn_counts = np.zeros(
            (self._cfg.grid_rows, self._cfg.grid_cols), dtype=np.int32,
        )
        for (r, c), deque_for_cell in self._spawn_times.items():
            spawn_counts[r, c] = len(deque_for_cell)
        return {
            "suppressed": mask.tolist(),
            "spawn_counts": spawn_counts.tolist(),
            "total_suppressed": [self._total_suppressed],
        }


__all__ = ["SpawnDensityConfig", "SpawnDensityMap"]
