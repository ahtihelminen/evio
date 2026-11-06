"""The rate-excess detector as first ported from evioR: the test oracle.

A frozen, unregistered copy of `evio.dtc.methods.rate_excess` before it was
optimised, kept verbatim in its math (float64 integral images, a full
sliding-window max, a dense variance map) so the equivalence test can hold the
fast detector to it. Do not optimise this file; it is the definition of the
output.

Differences from the real device:
- none known: it is the definition of the output the optimised detector is
  held to; it is slower and not registered as a method.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from evio.core.events import EventsBatch
from evio.dtc.errors import ConfigError
from evio.dtc.facts import EXTENT
from evio.dtc.methods.rate_excess import CELL_PX, RateExcessConfig
from evio.dtc.protocols import Context
from evio.dtc.types import (
    Box,
    Candidate,
    Candidates,
    Region,
    State,
    Status,
    Step,
)

_MIN_EXCESS_PER_CELL = 2.0
_EXTENT_CELLS = 7
_DILATE_CELLS = 2

Map = NDArray[np.float32]


class ReferenceRateExcess:
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
        size = self.config.window_cells
        observed = window_sums(counts, size)
        expected = window_sums(self._mean, size)
        variance = window_sums(self._variance(), size)
        return (observed - expected) / np.sqrt(variance)

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


def window_sums(values: Map, size: int) -> Map:
    """Sum over every size x size window fully inside the map.

    The integral image is float64: a hot cell can have a variance in the
    tens of thousands, and float32 sums of that size are off by more than a
    quiet window's whole variance.
    """
    rows, cols = values.shape
    total = np.zeros((rows + 1, cols + 1), np.float64)
    total[1:, 1:] = values.cumsum(0, dtype=np.float64).cumsum(1)
    return (
        total[size:, size:]
        - total[:-size, size:]
        - total[size:, :-size]
        + total[:-size, :-size]
    )


def peaks(z: Map, size: int, threshold: float) -> list[tuple[int, int]]:
    """Window positions that are the highest nearby and above threshold."""
    before, after = size // 2, (size - 1) // 2
    padded = np.pad(z, ((before, after),) * 2, constant_values=-np.inf)
    highest = sliding_window_view(padded, (size, size)).max(axis=(2, 3))
    rows, cols = np.nonzero((z >= threshold) & (z >= highest))
    order = np.argsort(-z[rows, cols])
    return [(int(rows[i]), int(cols[i])) for i in order]


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
