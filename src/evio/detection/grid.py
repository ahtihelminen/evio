from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from evio.core.events import EventsBatch


# --- Grid config ---
@dataclass(frozen=True, slots=True)
class GridConfig:
    width: int
    height: int
    cell_w: int
    cell_h: int
    dt_us: int

    @property
    def cols(self) -> int:
        return (self.width + self.cell_w - 1) // self.cell_w

    @property
    def rows(self) -> int:
        return (self.height + self.cell_h - 1) // self.cell_h

    @property
    def cells(self) -> int:
        return self.rows * self.cols

    @property
    def cell_area(self) -> int:
        return self.cell_w * self.cell_h

    def cell_id(
        self, x: NDArray[np.integer], y: NDArray[np.integer]
    ) -> NDArray[np.int32]:
        cx = (x // self.cell_w).astype(np.int32, copy=False)
        cy = (y // self.cell_h).astype(np.int32, copy=False)
        # Optionally clip to grid if events can be out-of-bounds:
        cx = np.clip(cx, 0, self.cols - 1)
        cy = np.clip(cy, 0, self.rows - 1)
        return cy * self.cols + cx

    def reshape_cells(self, flat: NDArray[np.number]) -> NDArray[np.number]:
        return flat.reshape(self.rows, self.cols)


# --- Feature function standard ---
class FeatureFn(Protocol):
    def __call__(
        self,
        events: EventsBatch,
        grid: GridConfig,
    ) -> NDArray[np.floating] | NDArray[np.integer]:
        """
        Must return per-cell feature array, either shape (grid.n_cells,)
        or shape (grid.n_rows, grid.n_cols).
        """
        ...


# --- GridState ---
class GridState:
    """
    Holds per-cell features as attributes.

    Usage:
        state = GridState(cfg, {"counts": feat_counts, "pos": feat_pos})
        state.update(events)
        print(state.counts.shape)  # (n_rows, n_cols) by default
    """

    def __init__(
        self,
        grid: GridConfig,
        features: dict[str, FeatureFn],
        *,
        store_2d: bool = True,
        dtype: np.dtype | None = None,
    ) -> None:
        self.grid = grid
        self._features = dict(features)
        self._store_2d = store_2d
        self._dtype = dtype
        self.feature_names = tuple(self._features.keys())

        # Pre-create attributes with None for IDE friendliness
        for name in self.feature_names:
            setattr(self, name, None)

    def update(self, events: EventsBatch) -> None:
        """
        Evaluate all registered features for this event batch.

        kwargs are forwarded to every feature function
        """
        for name, fn in self._features.items():
            arr = fn(events, self.grid)
            if self._dtype is not None:
                arr = np.asarray(arr, dtype=self._dtype)
            setattr(self, name, arr)
