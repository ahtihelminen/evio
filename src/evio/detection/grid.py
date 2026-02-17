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
    rows: int
    cols: int
    time_bins: int
    window_us: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            msg = "width and height must be positive"
            raise ValueError(msg)
        if self.rows <= 0 or self.cols <= 0:
            msg = "rows and cols must be positive"
            raise ValueError(msg)
        if self.time_bins <= 0:
            msg = "time_bins must be positive"
            raise ValueError(msg)
        if self.window_us <= 0:
            msg = "window_us must be positive"
            raise ValueError(msg)

    @property
    def cell_w(self) -> int:
        return (self.width + self.cols - 1) // self.cols

    @property
    def cell_h(self) -> int:
        return (self.height + self.rows - 1) // self.rows

    @property
    def dt_us(self) -> int:
        return max(1, (self.window_us + self.time_bins - 1) // self.time_bins)

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
        Must return per-cell feature array, either shape (grid.cells,)
        or shape (grid.rows, grid.cols).
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
