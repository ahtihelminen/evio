"""The frame detector is the same method on a stream or on a mosaic."""

from __future__ import annotations

import pytest
from fakes.dtc_frame import Frame

from evio.dtc.facts import EXTENT, SEAMS, Facts
from evio.dtc.protocols import Context
from evio.dtc.registry import create
from evio.dtc.sources import seam_columns, stitch
from evio.dtc.types import Candidates, Step, Window
from tests.support.dtc import blob_events, step_of, window

pytestmark = pytest.mark.t1

STEP_US = 10_000
CONFIG = {"min_count": 10, "radius": 8}
LAYOUT = [("a", 0, 0), ("b", 1280, 0)]
SEAMS_FACT = seam_columns(LAYOUT, {"a": 1280, "b": 1280})


def blob(stream: str, cx: int, cy: int) -> Window:
    return window(
        stream, blob_events(0, STEP_US, cx, cy, count=2000, seed=1), 0, STEP_US
    )


def detect(step: Step, stream: str, facts: Facts) -> Candidates:
    context = Context(stream, (stream,), facts, STEP_US)
    return create("frame", CONFIG, context).step(step, ())


def centers(found: Candidates) -> list[tuple[float, float]]:
    return sorted(c.region.center for c in found.items)


def test_it_requires_only_the_extent() -> None:
    assert Frame.manifest.requires == (EXTENT,)


def test_mosaic_finds_what_the_single_streams_find() -> None:
    step = step_of(0, STEP_US, blob("a", 100, 50), blob("b", 200, 300))
    alone_a = detect(step.only("a"), "a", {EXTENT: (1280, 720)})
    alone_b = detect(step.only("b"), "b", {EXTENT: (1280, 720)})

    mosaic = detect(stitch(step, LAYOUT, "m"), "m", {EXTENT: (2560, 720)})

    (ax, ay), (bx, by) = centers(alone_a)[0], centers(alone_b)[0]
    assert len(alone_a.items) == len(alone_b.items) == 1
    assert centers(mosaic) == [(ax, ay), (bx + 1280, by)]


def test_seam_fact_changes_output_only_when_given() -> None:
    step = step_of(0, STEP_US, blob("a", 1278, 50), blob("b", 200, 300))
    mosaic = stitch(step, LAYOUT, "m")
    facts = {EXTENT: (2560, 720)}

    blind = detect(mosaic, "m", facts)
    aware = detect(mosaic, "m", {**facts, SEAMS: SEAMS_FACT})

    assert len(blind.items) == 2
    assert [1470 < c.region.center[0] < 1490 for c in aware.items] == [True]


def test_seam_fact_leaves_a_stream_without_a_straddler_unchanged() -> None:
    step = step_of(0, STEP_US, blob("a", 100, 50))
    facts = {EXTENT: (1280, 720)}

    assert detect(step, "a", facts) == detect(step, "a", {**facts, SEAMS: ()})
    assert detect(step, "a", facts) == detect(
        step, "a", {**facts, SEAMS: (1280,)}
    )


def test_reset_leaves_results_repeatable() -> None:
    step = step_of(0, STEP_US, blob("a", 100, 50))
    context = Context("a", ("a",), {EXTENT: (1280, 720)}, STEP_US)
    detector = create("frame", CONFIG, context)
    first = detector.step(step, ())

    detector.reset()

    assert detector.step(step, ()) == first
