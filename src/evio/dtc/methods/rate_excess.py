"""Blobs of events beyond a learned per-cell background.

The sensor is cut into 4 px cells. For each cell the detector learns, over a
warmup, how many events a step normally brings (mean and spread). A target is
a stretch of cells that together carry more events than that background
predicts: sum the observed counts over a small window of cells, subtract the
sum of the expected counts, and divide by the spread. That is a z-score with a
null that can be checked: on pure background it should rarely reach the
threshold.

The background learns everywhere except under a blob it just reported, so a
target that hovers is not slowly taught to the model as normal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from evio.core.events import EventsBatch
from evio.dtc.errors import ConfigError
from evio.dtc.facts import EXTENT
from evio.dtc.manifest import Stage
from evio.dtc.protocols import Context
from evio.dtc.registry import method
from evio.dtc.types import (
    Box,
    Candidate,
    Candidates,
    Region,
    State,
    Status,
    Step,
)

CELL_PX = 4
_MIN_EXCESS_PER_CELL = 2.0
_EXTENT_CELLS = 7
_DILATE_CELLS = 2

Map = NDArray[np.float32]


@dataclass(frozen=True)
class RateExcessConfig:
    """`window_cells` is the side of the summing window, in 4 px cells."""

    window_cells: int = 5
    z_threshold: float = 8.0
    warmup_steps: int = 25
    memory_steps: int = 50
    dispersion: float = 0.5
    max_blobs: int = 5
    extent_fraction: float = 0.05


@method(
    "rate-excess",
    stages=(Stage.DETECT,),
    config=RateExcessConfig,
    requires=(EXTENT,),
)
class RateExcess:
    """Blobs of events beyond the learned per-cell background."""

    def __init__(self, config: RateExcessConfig, context: Context) -> None:
        self.config = config
        self.stream = str(context.stream)
        self.extent = context.facts[EXTENT]
        width, height = self.extent
        self.shape = (-(-height // CELL_PX), -(-width // CELL_PX))
        if min(self.shape) < config.window_cells:
            msg = (
                f"extent {self.extent} is smaller than the "
                f"{config.window_cells}-cell window of {CELL_PX} px cells"
            )
            raise ConfigError(msg)
        self.reset()

    def reset(self) -> None:
        self._mean: Map = np.zeros(self.shape, np.float32)
        self._spread: Map = np.zeros(self.shape, np.float32)
        self._steps = 0

    @property
    def mean(self) -> Map:
        """The learned events per step of each cell, (rows, cols). Read-only."""
        return _read_only(self._mean)

    @property
    def spread(self) -> Map:
        """The learned variance of every cell, (rows, cols). Read-only."""
        return _read_only(self._spread)

    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        counts = self._count_map(step.windows[self.stream].events)
        self._steps += 1
        warmup = self.config.warmup_steps
        if self._steps <= warmup:
            self._learn(counts, 1.0 / self._steps, keep=None)
            status = Status(State.CALIBRATING, self._steps / warmup)
            return Candidates(status=status)
        found = self._find(counts)
        keep = keep_mask(found, self.shape)
        self._learn(counts, 1.0 / self.config.memory_steps, keep=keep)
        return Candidates(tuple(found))

    def _count_map(self, events: EventsBatch) -> Map:
        width, height = self.extent
        x = events.x_coords.astype(np.int64)
        y = events.y_coords.astype(np.int64)
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        rows, cols = self.shape
        cell = (y[inside] >> 2) * cols + (x[inside] >> 2)
        counts = np.bincount(cell, minlength=rows * cols)
        return counts.reshape(self.shape).astype(np.float32)

    def _learn(self, counts: Map, rate: float, keep: Map | None) -> None:
        delta = counts - self._mean
        update = np.float32(rate) if keep is None else rate * keep
        self._mean += update * delta
        self._spread += update * (delta * delta - self._spread)

    def _variance(self) -> Map:
        poisson_like = self._mean + self.config.dispersion * self._mean**2
        return np.maximum(np.maximum(self._spread, poisson_like), 0.02)

    def _z_map(self, counts: Map) -> Map:
        """The z-score of every window position, for this step's counts."""
        sums = window_sums(
            np.stack([counts - self._mean, self._variance()]),
            self.config.window_cells,
        )
        return sums[0] / np.sqrt(sums[1])

    def _find(self, counts: Map) -> list[Candidate]:
        config = self.config
        size = config.window_cells
        z = self._z_map(counts)
        found = peaks(z, size, config.z_threshold)[: config.max_blobs]
        excess = counts - self._mean
        middle = size // 2
        return [
            Candidate(
                self.stream,
                Region(
                    excess_box(
                        excess,
                        row + middle,
                        col + middle,
                        config.extent_fraction,
                    )
                ),
                float(z[row, col]),
                {"z": float(z[row, col])},
            )
            for row, col in found
        ]


def _read_only(values: Map) -> Map:
    view = values.view()
    view.setflags(write=False)
    return view


def window_sums(values: NDArray[np.float32], size: int) -> NDArray[np.float64]:
    """Sum over every size x size window fully inside the last two axes.

    Summed directly in float64, a row pass then a column pass. An integral
    image is no cheaper here and cancels badly: a hot cell can have a variance
    in the tens of thousands, which would swamp a quiet window's whole sum.
    """
    rows = _slide_sum(values.astype(np.float64), size)
    return _slide_sum(rows.swapaxes(-1, -2), size).swapaxes(-1, -2)


def _slide_sum(values: NDArray[np.float64], size: int) -> NDArray[np.float64]:
    """Sum of every run of `size` neighbours along the second-last axis."""
    runs = values.shape[-2] - size + 1
    total = values[..., :runs, :].copy()
    for shift in range(1, size):
        total += values[..., shift : shift + runs, :]
    return total


def peaks(z: Map, size: int, threshold: float) -> list[tuple[int, int]]:
    """Window positions that are the highest nearby and above threshold.

    The highest value nearby is a running maximum along the rows, then along
    the columns: a fixed cost however many positions pass the threshold.
    """
    before, after = size // 2, (size - 1) // 2
    padded = np.pad(z, ((before, after),) * 2, constant_values=-np.inf)
    highest = _slide_max(_slide_max(padded.T, size).T, size)
    rows, cols = np.nonzero((z >= threshold) & (z >= highest))
    order = np.argsort(-z[rows, cols])
    return [(int(rows[i]), int(cols[i])) for i in order]


def _slide_max(values: NDArray[np.float64], size: int) -> NDArray[np.float64]:
    """Maximum of every run of `size` neighbours along the second-last axis."""
    runs = values.shape[-2] - size + 1
    top = values[:runs].copy()
    for shift in range(1, size):
        np.maximum(top, values[shift : shift + runs], out=top)
    return top


def excess_box(excess: Map, row: int, col: int, fraction: float) -> Box:
    """The cells around a peak that carry the excess, as a pixel box."""
    top, left = max(row - _EXTENT_CELLS, 0), max(col - _EXTENT_CELLS, 0)
    bottom, right = row + _EXTENT_CELLS + 1, col + _EXTENT_CELLS + 1
    patch = excess[top:bottom, left:right]
    limit = max(_MIN_EXCESS_PER_CELL, fraction * float(patch.max()))
    rows, cols = np.nonzero(patch >= limit)
    if rows.size == 0:
        rows, cols = np.array([row - top]), np.array([col - left])
    return (
        float((left + cols.min()) * CELL_PX),
        float((top + rows.min()) * CELL_PX),
        float((left + cols.max() + 1) * CELL_PX),
        float((top + rows.max() + 1) * CELL_PX),
    )


def keep_mask(found: list[Candidate], shape: tuple[int, int]) -> Map:
    """1 everywhere except under (and just around) a reported blob."""
    keep = np.ones(shape, np.float32)
    for candidate in found:
        x0, y0, x1, y1 = candidate.region.box
        keep[
            max(int(y0) // CELL_PX - _DILATE_CELLS, 0) : int(y1) // CELL_PX
            + _DILATE_CELLS
            + 1,
            max(int(x0) // CELL_PX - _DILATE_CELLS, 0) : int(x1) // CELL_PX
            + _DILATE_CELLS
            + 1,
        ] = 0.0
    return keep
