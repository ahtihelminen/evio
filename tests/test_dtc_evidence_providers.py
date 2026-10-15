"""The built-in evidence providers: values, ring semantics, purity."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from evio.dtc import evidence_providers
from evio.dtc.evidence import PROVIDERS, EvidenceProvider
from evio.dtc.evidence_providers import (
    POINT_HALF_SIZE_PX,
    RING_LENGTH,
    ROI_MARGIN_PX,
    track_box,
)
from evio.dtc.protocols import Context
from evio.dtc.types import Region, Step, Track, Tracks
from tests.support import dtc

pytestmark = pytest.mark.t1

STEP_US = 10_000
STREAM = "cam"
CONTEXT = Context(STREAM, (STREAM,), {}, STEP_US)
BOX = (10.0, 10.0, 20.0, 20.0)
BOX_PX = (10, 10, 20, 20)


def provider(name: str) -> EvidenceProvider:
    return PROVIDERS[name](CONTEXT)


def track(
    id_: str = "a", region: Region | None = None, **features: float
) -> Track:
    return Track(STREAM, id_, region or Region(box=BOX), 0.9, features)


def tracks_of(*items: Track, live: set[str] | None = None) -> Tracks:
    ids = {t.id for t in items} if live is None else live
    return Tracks(items, frozenset(ids))


def step_with(
    t_us: list[int], x: list[int], y: list[int], index: int = 0
) -> Step:
    start = index * STEP_US
    batch = dtc.events(t_us, x, y)
    return dtc.step_of(
        index, STEP_US, dtc.window(STREAM, batch, start, start + STEP_US)
    )


def blob_step(index: int) -> Step:
    start = index * STEP_US
    batch = dtc.blob_events(start, start + STEP_US, 15, 15, seed=index)
    return dtc.step_of(
        index, STEP_US, dtc.window(STREAM, batch, start, start + STEP_US)
    )


def empty_step(index: int = 0) -> Step:
    return step_with([], [], [], index)


def test_all_four_providers_are_registered() -> None:
    names = {"kinematics", "roi_events", "patch_ring", "event_series"}
    assert names <= PROVIDERS.keys()


def test_kinematics_passes_features_through_as_a_copy() -> None:
    t = track(speed=3.5, accel=-1.0)
    value = provider("kinematics").compute(empty_step(), tracks_of(t))
    assert value == {"a": {"speed": 3.5, "accel": -1.0}}
    assert value["a"] is not t.features


def test_point_region_gets_a_fixed_box() -> None:
    x0, y0, x1, y1 = track_box(Region(point=(30.0, 40.0)))
    assert (x1 - x0, y1 - y0) == (2 * POINT_HALF_SIZE_PX,) * 2
    assert ((x0 + x1) / 2, (y0 + y1) / 2) == (30.0, 40.0)


def test_roi_events_keeps_events_inside_box_plus_margin() -> None:
    inside = 20 + ROI_MARGIN_PX
    outside = inside + 1
    step = step_with([1, 2, 3], [15, inside, outside], [15, 15, 15])
    value = provider("roi_events").compute(step, tracks_of(track()))
    assert value["a"].x_coords.tolist() == [15, inside]
    assert value["a"].timestamps.tolist() == [1, 2]


def test_roi_events_copies_rather_than_views() -> None:
    step = step_with([1], [15], [15])
    value = provider("roi_events").compute(step, tracks_of(track()))
    value["a"].x_coords[0] = 99
    assert step.windows[STREAM].events.x_coords[0] == 15


def test_roi_events_follows_a_point_only_track() -> None:
    step = step_with([1, 2], [100, 130], [100, 100])
    t = track(region=Region(point=(100.0, 100.0)))
    value = provider("roi_events").compute(step, tracks_of(t))
    assert value["a"].x_coords.tolist() == [100]


def test_event_series_bins_polarity_at_half_a_millisecond() -> None:
    batch = dtc.events([0, 499, 500, 1_200], [15] * 4, [15] * 4)
    batch.polarities[2] = 0
    window = dtc.window(STREAM, batch, 0, STEP_US)
    step = dtc.step_of(0, STEP_US, window)
    series = provider("event_series").compute(step, tracks_of(track()))["a"]
    assert series["t_start_us"] == 0
    assert series["on"].tolist()[:3] == [2, 0, 1]
    assert series["off"].tolist()[:3] == [0, 1, 0]
    assert len(series["on"]) == len(series["off"]) == STEP_US // 500


def test_roi_events_margin_is_inclusive_on_both_edges() -> None:
    low = 10 - ROI_MARGIN_PX
    step = step_with([1, 2], [low, low - 1], [15, 15])
    value = provider("roi_events").compute(step, tracks_of(track()))
    assert value["a"].x_coords.tolist() == [low]


def test_event_series_covers_a_window_that_is_not_a_whole_bin() -> None:
    batch = dtc.events([0, 1_499], [15] * 2, [15] * 2)
    window = dtc.window(STREAM, batch, 0, 1_501)
    step = dtc.step_of(0, 1_501, window)
    series = provider("event_series").compute(step, tracks_of(track()))["a"]
    assert series["on"].tolist() == [1, 0, 1, 0]


def test_event_series_is_anchored_at_the_window_start() -> None:
    step = step_with([STEP_US + 600], [15], [15], index=1)
    series = provider("event_series").compute(step, tracks_of(track()))["a"]
    assert series["t_start_us"] == STEP_US
    assert series["on"].tolist()[:3] == [0, 1, 0]


def test_empty_window_yields_an_empty_roi_batch() -> None:
    value = provider("roi_events").compute(empty_step(), tracks_of(track()))
    assert len(value["a"].timestamps) == 0


def test_empty_window_series_is_all_zero() -> None:
    series = provider("event_series").compute(empty_step(), tracks_of(track()))
    assert series["a"]["on"].sum() == series["a"]["off"].sum() == 0


def feed(ring: EvidenceProvider, indices: range) -> dict[str, object]:
    for i in indices:
        value = ring.compute(blob_step(i), tracks_of(track()))
    return value


def test_patch_ring_is_absent_until_full() -> None:
    ring = provider("patch_ring")
    assert feed(ring, range(RING_LENGTH - 1)) == {}
    full = feed(ring, range(RING_LENGTH - 1, RING_LENGTH))["a"]
    assert full.shape == (RING_LENGTH, 32, 32, 3)
    assert full.dtype == np.uint8


def test_patch_ring_is_ordered_oldest_first() -> None:
    from evio.detection.reid import extract_event_patch

    full = feed(provider("patch_ring"), range(RING_LENGTH))["a"]
    for i in (0, RING_LENGTH - 1):
        events = blob_step(i).windows[STREAM].events
        assert np.array_equal(full[i], extract_event_patch(events, BOX_PX))


def test_patch_ring_slides_after_full() -> None:
    slid = feed(provider("patch_ring"), range(RING_LENGTH + 3))["a"]
    late = feed(provider("patch_ring"), range(3, RING_LENGTH + 3))["a"]
    assert np.array_equal(slid, late)


def test_patch_ring_drops_a_dead_track_and_restarts_it() -> None:
    ring = provider("patch_ring")
    for i in range(RING_LENGTH):
        ring.compute(blob_step(i), tracks_of(track()))
    ring.compute(empty_step(RING_LENGTH), tracks_of(live=set()))
    revived = ring.compute(blob_step(RING_LENGTH + 1), tracks_of(track()))
    assert revived == {}


def test_patch_ring_keeps_a_coasting_track_ring() -> None:
    ring = provider("patch_ring")
    for i in range(RING_LENGTH - 1):
        ring.compute(blob_step(i), tracks_of(track()))
    ring.compute(empty_step(), tracks_of(live={"a"}))
    value = ring.compute(blob_step(RING_LENGTH), tracks_of(track()))
    assert "a" in value


def test_patch_ring_reset_clears_rings() -> None:
    ring = provider("patch_ring")
    for i in range(RING_LENGTH):
        ring.compute(blob_step(i), tracks_of(track()))
    ring.reset()
    assert ring.compute(blob_step(0), tracks_of(track())) == {}


def test_patch_ring_keeps_tracks_apart() -> None:
    ring = provider("patch_ring")
    pair = tracks_of(track("a"), track("b"))
    for i in range(RING_LENGTH):
        value = ring.compute(blob_step(i), pair)
    assert value.keys() == {"a", "b"}
    assert value["a"].shape == value["b"].shape
    assert np.array_equal(value["a"], value["b"])


def test_patch_ring_does_not_share_patches_between_tracks() -> None:
    ring = provider("patch_ring")
    far = Region(box=(100.0, 100.0, 110.0, 110.0))
    pair = tracks_of(track("a"), track("b", far))
    for i in range(RING_LENGTH):
        value = ring.compute(blob_step(i), pair)
    assert not np.array_equal(value["a"], value["b"])


@pytest.mark.parametrize(
    "name", ["kinematics", "roi_events", "patch_ring", "event_series"]
)
def test_providers_do_not_mutate_their_input(name: str) -> None:
    step = blob_step(0)
    given = tracks_of(track(speed=1.0))
    before = (copy.deepcopy(step), copy.deepcopy(given))
    provider(name).compute(step, given)
    assert step.windows[STREAM].events.x_coords.tolist() == (
        before[0].windows[STREAM].events.x_coords.tolist()
    )
    assert given == before[1]


@pytest.mark.parametrize(
    "name", ["kinematics", "roi_events", "patch_ring", "event_series"]
)
def test_providers_are_deterministic(name: str) -> None:
    def run() -> list[object]:
        p = provider(name)
        return [
            repr(p.compute(blob_step(i), tracks_of(track(speed=1.0))))
            for i in range(RING_LENGTH)
        ]

    assert run() == run()


def test_module_imports_no_heavy_dependency_at_import_time() -> None:
    source = Path(evidence_providers.__file__).read_text()
    module = ast.parse(source)
    top = {
        ast.unparse(node)
        for node in module.body
        if isinstance(node, ast.Import | ast.ImportFrom)
    }
    heavy = ("torch", "openvino", "cv2", "evio.detection")
    assert not [i for i in top if any(h in i for h in heavy)]
    lazy = [
        node
        for fn in ast.walk(module)
        if isinstance(fn, ast.FunctionDef)
        for node in fn.body
        if isinstance(node, ast.ImportFrom)
    ]
    assert [n.module for n in lazy] == ["evio.detection.reid"]
