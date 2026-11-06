"""Single-frame detector: one 2D count image, local maxima as candidates.

A gate fixture for the neutrality gate, registered with the real
@method decorator on import.

Differences from the real device:
- it is a fixture, not a port of a production detector: its thresholds
  are defaults, not tuned against recordings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from evio.dtc.facts import EXTENT, SEAMS
from evio.dtc.manifest import Stage
from evio.dtc.registry import method
from evio.dtc.types import Candidate, Candidates, Region

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evio.dtc.protocols import Context
    from evio.dtc.types import Step


@dataclass(frozen=True)
class FrameConfig:
    min_count: int = 20
    radius: int = 4
    max_candidates: int = 8


@method(
    "frame",
    stages=(Stage.DETECT,),
    config=FrameConfig,
    requires=(EXTENT,),
)
class Frame:
    """Count image of the window; local maxima, minus any that cross a seam."""

    def __init__(self, config: FrameConfig, context: Context) -> None:
        self.config = config
        self.stream = context.stream
        self.width, self.height = context.facts[EXTENT]
        self.seams: tuple[int, ...] = tuple(context.facts.get(SEAMS, ()))

    def reset(self) -> None:
        """Stateless: nothing to return to the built state."""

    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        events = step.windows[self.stream].events
        x = events.x_coords.astype(np.int64)
        y = events.y_coords.astype(np.int64)
        inside = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        image = np.bincount(
            y[inside] * self.width + x[inside],
            minlength=self.width * self.height,
        ).reshape(self.height, self.width)
        return Candidates(tuple(self._candidates(image)))

    def _candidates(self, image: np.ndarray) -> list[Candidate]:
        radius = self.config.radius
        ys, xs = np.nonzero(image >= self.config.min_count)
        counts = image[ys, xs]
        order = np.lexsort((xs, ys, -counts))
        peaks: list[tuple[int, int, int]] = []
        for i in order:
            px, py = int(xs[i]), int(ys[i])
            if all(
                max(abs(px - qx), abs(py - qy)) > radius for qx, qy, _ in peaks
            ):
                peaks.append((px, py, int(counts[i])))
        candidates = []
        for px, py, count in peaks[: self.config.max_candidates]:
            x0, x1 = px - radius, px + radius + 1
            if not any(x0 < seam < x1 for seam in self.seams):
                box = (x0, py - radius, x1, py + radius + 1)
                candidates.append(
                    Candidate(self.stream, Region(box=box), float(count))
                )
        return candidates
