"""The frame-free spiking point method against the DTC contract."""

from __future__ import annotations

import pytest
from fakes.dtc_point import SpikingPoint

from evio.dtc.errors import ConfigError
from evio.dtc.manifest import Stage
from evio.dtc.protocols import Context
from evio.dtc.registry import create
from evio.dtc.types import Candidates, Step
from tests.support.dtc import blob_events, empty_events, events, step_of, window

pytestmark = pytest.mark.t1

STEP_US = 10_000
CONTEXT = Context("cam", ("cam",), {}, STEP_US)


def _blob_step(
    index: int = 0,
    *,
    cx: int = 100,
    cy: int = 50,
    extent: tuple[int, int] | None = (1280, 720),
) -> Step:
    start = index * STEP_US
    batch = blob_events(start, start + STEP_US, cx, cy)
    return step_of(
        index, STEP_US, window("cam", batch, start, start + STEP_US, extent)
    )


def _built() -> SpikingPoint:
    built = create("point", {}, CONTEXT)
    assert isinstance(built, SpikingPoint)
    return built


def test_manifest_declares_a_detector_of_points_that_needs_no_facts() -> None:
    manifest = SpikingPoint.manifest

    assert manifest.stages == (Stage.DETECT,)
    assert manifest.geometry == ("point",)
    assert manifest.requires == ()


def test_blob_yields_point_only_candidates_near_it() -> None:
    candidates = _built().step(_blob_step(), ())

    assert candidates.items
    for candidate in candidates.items:
        assert candidate.region.box is None
        x, y = candidate.region.center
        assert abs(x - 100) <= 4
        assert abs(y - 50) <= 4
        assert candidate.score >= 1


def test_runs_without_an_extent() -> None:
    with_extent = _built().step(_blob_step(), ())
    without = _built().step(_blob_step(extent=None), ())

    assert without == with_extent


def test_empty_window_yields_nothing() -> None:
    batch = empty_events()
    step = step_of(0, STEP_US, window("cam", batch, 0, STEP_US, None))

    assert _built().step(step, ()) == Candidates()


def test_sparse_noise_never_spikes() -> None:
    batch = blob_events(0, STEP_US, 100, 50, count=5)
    step = step_of(0, STEP_US, window("cam", batch, 0, STEP_US))

    assert _built().step(step, ()).items == ()


def test_far_apart_blobs_spike_in_separate_cells() -> None:
    near = blob_events(0, STEP_US, 100, 50, seed=1)
    far = blob_events(0, STEP_US, 900, 600, seed=2)
    rows = sorted(
        zip(
            near.timestamps.tolist() + far.timestamps.tolist(),
            near.x_coords.tolist() + far.x_coords.tolist(),
            near.y_coords.tolist() + far.y_coords.tolist(),
            strict=True,
        )
    )
    t, x, y = (list(column) for column in zip(*rows, strict=True))
    step = step_of(0, STEP_US, window("cam", events(t, x, y), 0, STEP_US))

    xs = [c.region.center[0] for c in _built().step(step, ()).items]

    assert any(x < 200 for x in xs)
    assert any(x > 800 for x in xs)


def test_reset_returns_to_the_freshly_built_state() -> None:
    method = _built()
    first = method.step(_blob_step(), ())

    method.reset()

    assert method.cells == {}
    assert method.step(_blob_step(), ()) == first


def test_same_input_same_output() -> None:
    assert _built().step(_blob_step(), ()) == _built().step(_blob_step(), ())


def test_silent_cells_are_forgotten() -> None:
    method = _built()
    method.step(_blob_step(), ())
    assert method.cells

    late = step_of(
        1000, STEP_US, window("cam", empty_events(), 10_000_000, 10_010_000)
    )
    method.step(late, ())

    assert method.cells == {}


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ConfigError):
        create("point", {"cell_size": 8}, CONTEXT)
