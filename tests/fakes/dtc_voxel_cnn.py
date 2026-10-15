"""GPU voxel CNN detector, stubbed: densest cells stand in for the network.

The real method feeds a time x height x width event voxel grid to a CNN on
the GPU. The stub builds the same grid in numpy and proposes the densest
cells, so the manifest and dependency checks can be exercised without torch.

A gate fixture for the neutrality gate, registered with the real
@method decorator on import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from evio.dtc.facts import EXTENT
from evio.dtc.manifest import Compute, Cost, Stage
from evio.dtc.protocols import Context
from evio.dtc.registry import method
from evio.dtc.types import Candidate, Candidates, Region, Step, Window

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class VoxelCnnConfig:
    time_bins: int = 4
    cell_px: int = 16
    top_k: int = 3
    min_events: int = 20


@method(
    "voxel_cnn",
    stages=(Stage.DETECT,),
    config=VoxelCnnConfig,
    compute=Compute.GPU,
    cost=Cost.HEAVY,
    dependencies=("torch",),
    requires=(EXTENT,),
    cadence=1,
)
class VoxelCnn:
    """GPU voxel CNN detector (stub: densest cells of a time-binned grid)."""

    def __init__(self, config: VoxelCnnConfig, context: Context) -> None:
        self.config = config
        width, height = context.facts[EXTENT]
        cell = config.cell_px
        self.grid_shape = (
            config.time_bins,
            -(-height // cell),
            -(-width // cell),
        )

    def reset(self) -> None:
        """Stateless: every step is computed from its own window."""

    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        items: list[Candidate] = []
        for window in step.windows.values():
            items.extend(self._detect(window))
        return Candidates(tuple(items))

    def voxel_grid(self, window: Window) -> np.ndarray:
        """Event counts per (time bin, cell row, cell column)."""
        bins, rows, cols = self.grid_shape
        batch = window.events
        span = max(window.t_end_us - window.t_start_us, 1)
        t_bin = (batch.timestamps - window.t_start_us) * bins // span
        t_bin = np.clip(t_bin, 0, bins - 1)
        row = np.clip(batch.y_coords // self.config.cell_px, 0, rows - 1)
        col = np.clip(batch.x_coords // self.config.cell_px, 0, cols - 1)
        grid = np.zeros(self.grid_shape, dtype=np.float32)
        np.add.at(grid, (t_bin, row, col), 1.0)
        return grid

    def _detect(self, window: Window) -> list[Candidate]:
        """The densest cells, each once, however many time bins it fills."""
        counts = self.voxel_grid(window).sum(axis=0)
        order = np.argsort(-counts.ravel(), kind="stable")
        cell = self.config.cell_px
        total = len(window.events.timestamps)
        found = []
        for flat in order[: self.config.top_k]:
            row, col = np.unravel_index(flat, counts.shape)
            count = float(counts[row, col])
            if count < max(self.config.min_events, 1):
                break
            box = (
                float(col * cell),
                float(row * cell),
                float((col + 1) * cell),
                float((row + 1) * cell),
            )
            found.append(
                Candidate(window.stream, Region(box=box), count / total)
            )
        return found
