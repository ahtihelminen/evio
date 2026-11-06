"""A frame-free spiking detector: leaky integrate-and-fire cells on a hash.

Each event feeds the coarse spatial cell it lands in. A cell leaks, integrates
and tracks the centroid of what it integrated; when its potential reaches the
threshold it spikes and resets. A step emits one point-only candidate per cell
that spiked, scored by its spike count. No raster is built and no extent is
needed, so it runs on any coordinate space.

A gate fixture for the neutrality gate, registered with the real
@method decorator on import.

Differences from the real device:
- it is a fixture, not a port of a production detector: the cells are a
  hash with fixed leak and threshold, not tuned against recordings.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from evio.dtc.manifest import Stage
from evio.dtc.protocols import Context
from evio.dtc.registry import method
from evio.dtc.types import Candidate, Candidates, Point, Region, Step

Cell = tuple[int, int]
FORGET_TAUS = 10


@dataclass(frozen=True)
class PointConfig:
    cell_px: int = 16
    tau_us: float = 5000.0
    threshold: float = 8.0


@method(
    "point",
    stages=(Stage.DETECT,),
    config=PointConfig,
    geometry=("point",),
)
class SpikingPoint:
    """Leaky integrate-and-fire cells over a spatial hash; centroids only."""

    def __init__(self, config: PointConfig, context: Context) -> None:
        self.config = config
        self.stream = context.stream
        self.reset()

    def reset(self) -> None:
        # cell -> [potential, potential-weighted x, same for y, last t_us]
        self.cells: dict[Cell, list[float]] = {}

    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        config = self.config
        events = step.windows[self.stream].events
        spikes: dict[Cell, tuple[Point, int]] = {}
        for x, y, t in zip(
            events.x_coords.tolist(),
            events.y_coords.tolist(),
            events.timestamps.tolist(),
            strict=True,
        ):
            cell = (x // config.cell_px, y // config.cell_px)
            state = self.cells.setdefault(cell, [0.0, 0.0, 0.0, t])
            leak = math.exp(-max(t - state[3], 0) / config.tau_us)
            state[:3] = [v * leak for v in state[:3]]
            state[0] += 1.0
            state[1] += x
            state[2] += y
            state[3] = t
            if state[0] >= config.threshold:
                _, count = spikes.get(cell, ((0.0, 0.0), 0))
                spikes[cell] = (
                    (state[1] / state[0], state[2] / state[0]),
                    count + 1,
                )
                state[:3] = [0.0, 0.0, 0.0]
        self._forget_silent(step.t_end_us)
        return Candidates(
            tuple(
                Candidate(self.stream, Region(point=point), float(count))
                for point, count in spikes.values()
            )
        )

    def _forget_silent(self, now_us: int) -> None:
        horizon = now_us - FORGET_TAUS * self.config.tau_us
        self.cells = {c: s for c, s in self.cells.items() if s[3] >= horizon}
