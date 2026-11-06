"""Cross-stream coincidence: a joint detector over every stream of a step.

A grid cell is active in a stream when it holds at least `min_events` events.
A cell active in `min_streams` streams at the same position, after each
stream's x offset, yields a candidate in every stream where it is active. A
stream missing from the step simply cannot vote; the output is then degraded.

A gate fixture for the neutrality gate, registered with the real
@method decorator on import.

Differences from the real device:
- it is a fixture, not a port of a production detector: coincidence is a
  plain count over a grid, with no calibration or epipolar geometry beyond
  each stream's x offset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from evio.dtc.manifest import Scope, Stage
from evio.dtc.protocols import Context
from evio.dtc.registry import method
from evio.dtc.types import (
    Candidate,
    Candidates,
    Region,
    State,
    Status,
    Step,
    StreamKey,
    Window,
)

Cell = tuple[int, int]


@dataclass(frozen=True)
class JointConfig:
    cell_px: int = 16
    min_events: int = 20
    min_streams: int = 2
    x_offsets: Mapping[StreamKey, int] = field(default_factory=dict)


@method("joint", stages=(Stage.DETECT,), config=JointConfig, scope=Scope.JOINT)
class Joint:
    """Detect positions active in several streams at once."""

    def __init__(self, config: JointConfig, context: Context) -> None:
        self.config = config
        self.streams = context.streams

    def reset(self) -> None:
        """Stateless: every step is judged on its own windows."""

    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        active = {
            stream: self._active_cells(window)
            for stream, window in step.windows.items()
        }
        votes: dict[Cell, int] = {}
        for cells in active.values():
            for cell in cells:
                votes[cell] = votes.get(cell, 0) + 1
        items = tuple(
            self._candidate(stream, cell, votes[cell])
            for stream, cells in active.items()
            for cell in sorted(cells)
            if votes[cell] >= self.config.min_streams
        )
        missing = [s for s in self.streams if s not in step.windows]
        if missing:
            return Candidates(
                items, Status(State.DEGRADED, detail=f"missing {missing}")
            )
        return Candidates(items)

    def _active_cells(self, window: Window) -> set[Cell]:
        cell_px = self.config.cell_px
        offset = self.config.x_offsets.get(window.stream, 0)
        x = (window.events.x_coords.astype(np.int64) + offset) // cell_px
        y = window.events.y_coords.astype(np.int64) // cell_px
        cells, counts = np.unique(
            np.stack([x, y], axis=1), axis=0, return_counts=True
        )
        busy = cells[counts >= self.config.min_events]
        return {(int(cx), int(cy)) for cx, cy in busy}

    def _candidate(
        self, stream: StreamKey, cell: Cell, votes: int
    ) -> Candidate:
        cell_px = self.config.cell_px
        x0 = cell[0] * cell_px - self.config.x_offsets.get(stream, 0)
        y0 = cell[1] * cell_px
        box = (float(x0), float(y0), float(x0 + cell_px), float(y0 + cell_px))
        return Candidate(
            stream,
            Region(box=box),
            votes / len(self.streams),
            {"streams_active": float(votes)},
        )
