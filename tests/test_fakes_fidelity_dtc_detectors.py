"""Fidelity for the DTC gate fixtures `frame`, `joint`, `point` and `voxel_cnn`.

Each is a registered DTC detector with no production twin of its own, so it
runs beside the real `rate-excess` detector on the same steps: the call
shape, the output type and where the candidates lie must agree, and the
output must pass the real consumers' checks (the conformance kit and a
built chain). `tests/fakes/dtc_voxel_cnn.py` also claims the shape of the
grid it bins into, which is checked here.
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.dtc.chain import build
from evio.dtc.conformance import check_method
from evio.dtc.facts import EXTENT
from evio.dtc.manifest import Scope, Stage
from evio.dtc.protocols import Context, Detector
from evio.dtc.registry import METHODS, create
from evio.dtc.types import Candidate, Candidates, Region, State, Step, Window
from tests.support.dtc_chain import spec_of

pytestmark = pytest.mark.t1

importlib.import_module("evio.dtc.methods")
for fixture in ("frame", "joint", "point", "voxel_cnn"):
    importlib.import_module(f"fakes.dtc_{fixture}")

STEP_US = 10_000
STREAMS = ("cam0", "cam1")
EXTENT_PX = (128, 96)
FACTS = {EXTENT: EXTENT_PX}
REAL = ("rate-excess", {"warmup_steps": 2})
DOUBLES = ("frame", "joint", "point", "voxel_cnn")
BLOBS = ((30, 30, 300), (90, 60, 150))
TOLERANCE_PX = 8


def scene(index: int) -> Step:
    """The same scene in both streams: two dense blobs from step 2 on."""
    t0 = index * STEP_US
    rng = np.random.default_rng(index)
    blobs = BLOBS if index >= 2 else ()
    x = [rng.integers(0, EXTENT_PX[0], 20)]
    y = [rng.integers(0, EXTENT_PX[1], 20)]
    for cx, cy, count in blobs:
        x.append(cx + rng.integers(-1, 2, count))
        y.append(cy + rng.integers(-1, 2, count))
    x, y = np.concatenate(x), np.concatenate(y)
    t = np.sort(rng.integers(t0, t0 + STEP_US, x.size)).astype(np.int64)
    batch = EventsBatch(
        t, x.astype(np.int16), y.astype(np.int16), np.ones(x.size, np.int8)
    )
    windows = {
        s: Window(s, batch, t0, t0 + STEP_US, EXTENT_PX) for s in STREAMS
    }
    return Step(index, t0, t0 + STEP_US, windows)


def detect(name: str, config: dict, steps: list[Step]) -> list[Candidates]:
    """Candidates of the last step, per stream, for a built detector."""
    if METHODS[name].manifest.scope is Scope.JOINT:
        joint = create(name, config, Context(None, STREAMS, FACTS, STEP_US))
        return [joint.step(step, ()) for step in steps]
    built = {
        s: create(name, config, Context(s, STREAMS, FACTS, STEP_US))
        for s in STREAMS
    }
    return [
        Candidates(
            tuple(
                c
                for s, d in built.items()
                for c in d.step(step.only(s), ()).items
            )
        )
        for step in steps
    ]


def lies_near(region: Region, point: tuple[float, float]) -> bool:
    if region.point is not None:
        x, y = region.point
        return np.hypot(x - point[0], y - point[1]) <= TOLERANCE_PX
    x0, y0, x1, y1 = region.box
    return (
        x0 - TOLERANCE_PX <= point[0] <= x1 + TOLERANCE_PX
        and y0 - TOLERANCE_PX <= point[1] <= y1 + TOLERANCE_PX
    )


def final(name: str, config: dict | None = None) -> Candidates:
    return detect(name, config or {}, [scene(i) for i in range(4)])[-1]


@pytest.mark.parametrize("name", DOUBLES)
def test_a_double_is_registered_with_the_stage_and_call_of_a_real_detector(
    name: str,
) -> None:
    double, real = METHODS[name], METHODS[REAL[0]]
    assert double.manifest.stages == real.manifest.stages == (Stage.DETECT,)
    for member in ("step", "reset"):
        assert list(inspect.signature(getattr(double, member)).parameters) == (
            list(inspect.signature(getattr(Detector, member)).parameters)
        ), member
    assert list(inspect.signature(real.step).parameters) == list(
        inspect.signature(double.step).parameters
    )


@pytest.mark.parametrize("name", DOUBLES)
def test_a_double_finds_each_blob_where_the_real_detector_finds_it(
    name: str,
) -> None:
    real = final(*REAL)
    double = final(name)
    assert real.items, "the real detector must find the blobs for this test"
    assert real.status.state is State.OK
    for candidate in real.items:
        assert all(isinstance(c, Candidate) for c in double.items)
        assert any(
            c.stream == candidate.stream
            and lies_near(c.region, candidate.region.center)
            for c in double.items
        ), (name, candidate.region)
    for blob in BLOBS:
        for stream in STREAMS:
            assert any(
                c.stream == stream and lies_near(c.region, blob[:2])
                for c in double.items
            ), (name, stream, blob)


@pytest.mark.parametrize("name", DOUBLES)
def test_a_double_is_quiet_where_the_real_detector_is_quiet(name: str) -> None:
    quiet = [scene(i) for i in range(2)]
    assert detect(REAL[0], REAL[1], quiet)[-1].items == ()
    assert detect(name, {}, quiet)[-1].items == ()


@pytest.mark.parametrize("name", DOUBLES)
def test_a_double_passes_the_conformance_kit_as_the_real_detector_does(
    name: str,
) -> None:
    assert check_method(REAL[0], facts=FACTS, config=REAL[1]) == []
    assert check_method(name, facts=FACTS) == []


@pytest.mark.parametrize("name", DOUBLES)
def test_a_built_chain_returns_the_candidates_of_a_double_like_a_real_one(
    name: str,
) -> None:
    for method in (REAL, (name, {})):
        pipeline = build(
            spec_of(method), FACTS, streams=STREAMS, step_us=STEP_US
        )
        results = [pipeline.step(scene(i)) for i in range(4)]
        assert results[-1].candidates, method
        assert all(
            isinstance(c, Candidate) and c.stream in STREAMS
            for c in results[-1].candidates
        )


def test_the_joint_double_reports_a_missing_stream_as_degraded() -> None:
    joint = create("joint", {}, Context(None, STREAMS, FACTS, STEP_US))
    both = scene(3)
    assert len(joint.step(both, ()).items) == len(BLOBS) * len(STREAMS)
    one = joint.step(both.only("cam0"), ())
    assert one.items == ()
    assert one.status.state is State.DEGRADED


@pytest.mark.parametrize(
    ("time_bins", "cell_px"), [(4, 16), (3, 10), (1, 128)]
)
def test_the_voxel_double_bins_into_a_time_by_height_by_width_grid(
    time_bins: int, cell_px: int
) -> None:
    config = {"time_bins": time_bins, "cell_px": cell_px}
    cnn = create("voxel_cnn", config, Context("cam0", STREAMS, FACTS, STEP_US))
    window = scene(3).windows["cam0"]
    grid = cnn.voxel_grid(window)
    width, height = EXTENT_PX
    rows, cols = -(-height // cell_px), -(-width // cell_px)
    assert grid.shape == (time_bins, rows, cols)
    assert grid.sum() == len(window.events.timestamps)
