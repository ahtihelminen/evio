"""The shipped fallback chain tracks a moving blob and classifies it."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.dtc.chain import build
from evio.dtc.facts import EXTENT
from evio.dtc.fallback import fallback_spec
from evio.dtc.registry import METHODS
from evio.dtc.result import StepResult
from tests.support import dtc

pytestmark = pytest.mark.t1

for module in ("rate_excess", "blob_tracker", "kinematic", "verdict_gate"):
    importlib.import_module(f"evio.dtc.methods.{module}")

STEP_US = 10_000
EXTENT_PX = (256, 192)
STREAM = "cam"
WARMUP_STEPS = 25
NOISE_EVENTS = 300
BLOB_EVENTS = 300


def test_names_the_four_methods_in_order() -> None:
    spec = fallback_spec()
    assert [stage.method for stage in spec.stages] == [
        "rate-excess",
        "blob-tracker",
        "kinematic",
        "verdict-gate",
    ]
    assert all(stage.method in METHODS for stage in spec.stages)


def step_events(index: int, blob: tuple[int, int] | None) -> EventsBatch:
    """Uniform background noise, plus a blob of events at `blob` if given."""
    rng = np.random.default_rng(index)
    x = rng.integers(0, EXTENT_PX[0], NOISE_EVENTS)
    y = rng.integers(0, EXTENT_PX[1], NOISE_EVENTS)
    if blob is not None:
        x = np.concatenate([x, blob[0] + rng.integers(-3, 4, BLOB_EVENTS)])
        y = np.concatenate([y, blob[1] + rng.integers(-3, 4, BLOB_EVENTS)])
    start = index * STEP_US
    t = np.sort(rng.integers(start, start + STEP_US, x.size))
    return dtc.events(t.tolist(), x.tolist(), y.tolist())


def run_chain(path: list[tuple[int, int]]) -> list[StepResult]:
    """Background noise for the warmup, then a blob along `path`."""
    pipeline = build(
        fallback_spec(),
        {EXTENT: EXTENT_PX},
        streams=[STREAM],
        step_us=STEP_US,
    )
    results = []
    for index, blob in enumerate([None] * WARMUP_STEPS + path):
        start = index * STEP_US
        window = dtc.window(
            STREAM, step_events(index, blob), start, start + STEP_US, EXTENT_PX
        )
        results.append(pipeline.step(dtc.step_of(index, STEP_US, window)))
    return results


def test_a_steady_straight_blob_raises_an_alarm() -> None:
    results = run_chain([(30 + 3 * i, 90) for i in range(60)])
    assert any(t.features for r in results for t in r.tracks)
    assert any(v.probability > 0.5 for r in results for v in r.verdicts)
    assert any(r.alarms for r in results)


def test_a_stationary_blob_raises_no_alarm() -> None:
    results = run_chain([(120, 90)] * 60)
    assert any(r.tracks for r in results)
    assert not any(r.alarms for r in results)
