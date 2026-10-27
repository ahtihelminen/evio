"""Turn event batches into steps and drive a pipeline with them.

Replay is the same code path as live: `run` only steps a pipeline, so any
iterable of steps (a file, a camera, a test) feeds it alike.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from evio.core.events import EventsBatch
from evio.dtc.types import Cue, Step, StreamKey, Window

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from evio.dtc.result import Pipeline, StepResult


def _joined(first: EventsBatch, second: EventsBatch) -> EventsBatch:
    return EventsBatch(
        *(
            np.concatenate([getattr(first, name), getattr(second, name)])
            for name in ("timestamps", "x_coords", "y_coords", "polarities")
        )
    )


def _split(events: EventsBatch, cut_us: int) -> tuple[EventsBatch, EventsBatch]:
    """Events before `cut_us` and the rest; views, never copies."""
    cut = int(np.searchsorted(events.timestamps, cut_us, side="left"))
    return (
        EventsBatch(
            events.timestamps[:cut],
            events.x_coords[:cut],
            events.y_coords[:cut],
            events.polarities[:cut],
        ),
        EventsBatch(
            events.timestamps[cut:],
            events.x_coords[cut:],
            events.y_coords[cut:],
            events.polarities[cut:],
        ),
    )


def steps_from_batches(
    batches: Iterable[EventsBatch],
    stream: StreamKey,
    step_us: int,
    extent: tuple[int, int] | None = None,
    start_us: int | None = None,
) -> Iterator[Step]:
    """Contiguous steps of `step_us`, anchored at `start_us`.

    `start_us` defaults to the first event's time; give streams the same one
    so `zip_steps` can align them. Input must be time-sorted and not precede
    `start_us`, else `ValueError`.

    Steps are half-open `[t_start, t_end)` and indexed from 0. Unlike
    `RollingWindowBatchProcessor`, a step with no events is still emitted, so
    a gap longer than a step yields empty steps and indices stay in step with
    sensor time. A step is emitted once an event at or after its end arrives;
    when the batches run out the last step is emitted too, with its full
    nominal span, since no later event can add to it. No trailing empty step
    follows. Windows hold views of the input arrays, which are never mutated.
    """
    last_us = start_us
    pending: EventsBatch | None = None
    index = 0

    def emit(events: EventsBatch) -> Step:
        assert start_us is not None
        end_us = start_us + step_us
        window = Window(stream, events, start_us, end_us, extent)
        return Step(index, start_us, end_us, {stream: window})

    for batch in batches:
        if not batch.timestamps.size:
            continue
        times = batch.timestamps
        if (last_us is not None and times[0] < last_us) or np.any(
            np.diff(times) < 0
        ):
            msg = f"events of stream {stream!r} are not time-sorted"
            raise ValueError(msg)
        last_us = int(times[-1])
        if start_us is None:
            start_us = int(times[0])
        pending = batch if pending is None else _joined(pending, batch)
        while int(pending.timestamps[-1]) >= start_us + step_us:
            head, pending = _split(pending, start_us + step_us)
            yield emit(head)
            index += 1
            start_us += step_us
    if pending is not None:
        yield emit(pending)


def zip_steps(*streams: Iterable[Step]) -> Iterator[Step]:
    """One step per index holding every stream's window.

    Streams must share one time grid (see `steps_from_batches`'s `start_us`)
    and end together, else `ValueError`.
    """
    for steps in zip(*streams, strict=True):
        if len({(s.index, s.t_start_us, s.t_end_us) for s in steps}) > 1:
            names = [key for s in steps for key in s.windows]
            msg = f"streams {names} are not aligned in time"
            raise ValueError(msg)
        keys = [key for s in steps for key in s.windows]
        if len(set(keys)) != len(keys):
            msg = f"streams share a window key: {keys}"
            raise ValueError(msg)
        yield Step(
            steps[0].index,
            steps[0].t_start_us,
            steps[0].t_end_us,
            {key: w for s in steps for key, w in s.windows.items()},
            tuple(c for s in steps for c in s.cues),
        )


def with_cues(
    steps: Iterable[Step], cues: Mapping[int, tuple[Cue, ...]]
) -> Iterator[Step]:
    """Each step with the cues recorded for its index (the `cues` tap), so a
    replay hands every stage exactly the cues the live run did."""
    for step in steps:
        yield replace(step, cues=step.cues + cues.get(step.index, ()))


def run(pipeline: Pipeline, steps: Iterable[Step]) -> Iterator[StepResult]:
    """Step `pipeline` through `steps` in order, yielding each result."""
    for step in steps:
        yield pipeline.step(step)
