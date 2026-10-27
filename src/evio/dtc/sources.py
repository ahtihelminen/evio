"""Source-side helpers that reshape steps before any method sees them.

A mosaic is a source, not a method property: `stitch` turns several windows
into one under a new stream key, moving the tiles' cues with them, and
`seam_columns` says where the joins are so the source can advertise them as
the `seams` fact.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from evio.core.events import EventsBatch
from evio.dtc.types import Step, StreamKey, Window

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def stitch(
    step: Step, layout: Sequence[tuple[StreamKey, int, int]], stream: StreamKey
) -> Step:
    """One window under `stream`: each layout window shifted by its offset.

    Events stay time-ordered (ties keep layout order). The extent is the
    bounding box of the tiles, or None when any tile's extent is unknown.
    """
    tiles = [(step.windows[key], dx, dy) for key, dx, dy in layout]
    parts = [
        EventsBatch(
            w.events.timestamps,
            w.events.x_coords + np.int16(dx),
            w.events.y_coords + np.int16(dy),
            w.events.polarities,
        )
        for w, dx, dy in tiles
    ]
    order = np.argsort(
        np.concatenate([p.timestamps for p in parts]), kind="stable"
    )
    events = EventsBatch(
        *(
            np.concatenate([getattr(p, name) for p in parts])[order]
            for name in ("timestamps", "x_coords", "y_coords", "polarities")
        )
    )
    extents = [
        (w.extent[0] + dx, w.extent[1] + dy)
        for w, dx, dy in tiles
        if w.extent is not None
    ]
    extent = (
        (max(x for x, _ in extents), max(y for _, y in extents))
        if len(extents) == len(tiles)
        else None
    )
    mosaic = Window(stream, events, step.t_start_us, step.t_end_us, extent)
    offsets = {key: (dx, dy) for key, dx, dy in layout}
    cues = tuple(
        cue if cue.stream is None else replace(
            cue, stream=stream,
            region=cue.region and cue.region.moved(*offsets[cue.stream]),
        )
        for cue in step.cues
        if cue.stream is None or cue.stream in offsets
    )
    return Step(
        step.index, step.t_start_us, step.t_end_us, {stream: mosaic}, cues
    )


def seam_columns(
    layout: Sequence[tuple[StreamKey, int, int]],
    widths: Mapping[StreamKey, int],
) -> tuple[int, ...]:
    """The interior x positions where tiles join, for the `seams` fact."""
    edges = {x for key, x, _ in layout} | {
        x + widths[key] for key, x, _ in layout
    }
    return tuple(sorted(edges - {min(edges), max(edges)}))
