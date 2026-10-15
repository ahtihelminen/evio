"""The simplified detection pipeline, adapted as a DTC detector."""

from __future__ import annotations

from functools import cache

import pytest

from evio.dtc.conformance import check_method
from evio.dtc.errors import ConfigError
from evio.dtc.facts import EXTENT
from evio.dtc.manifest import Cost, Scope, Stage
from evio.dtc.methods.simplified import Simplified, SimplifiedConfig
from evio.dtc.protocols import Context
from evio.dtc.registry import METHODS, parse_config
from evio.dtc.types import Candidates, State
from tests.support import dtc

pytestmark = pytest.mark.t1

STEP_US = 10_000
STREAM = "cam"
EXTENT_PX = (64, 48)
CALIBRATION_STEPS = 4
BLOB_FROM = 8
CONFIG = SimplifiedConfig(
    rows=12,
    cols=16,
    calibration_time_us=3 * STEP_US,
    calibration_windows=2,
    z_threshold=3,
)


def build(config: SimplifiedConfig = CONFIG) -> Simplified:
    context = Context(STREAM, (STREAM,), {EXTENT: EXTENT_PX}, STEP_US)
    return Simplified(config, context)


def drive(detector: Simplified, count: int) -> list[Candidates]:
    """Wide dim noise, then from BLOB_FROM a dense blob drifting right."""
    outputs = []
    for i in range(count):
        start, end = i * STEP_US, (i + 1) * STEP_US
        batch = (
            dtc.blob_events(start, end, 10 + i, 24, count=300, radius=2, seed=i)
            if i >= BLOB_FROM
            else dtc.blob_events(
                start, end, 32, 24, count=20, radius=15, seed=i
            )
        )
        window = dtc.window(STREAM, batch, start, end, EXTENT_PX)
        outputs.append(detector.step(dtc.step_of(i, STEP_US, window), ()))
    return outputs


@cache
def first_run() -> tuple[Candidates, ...]:
    return tuple(drive(build(), 16))


def test_registered_as_a_heavy_stream_detector_needing_extent() -> None:
    manifest = METHODS["simplified"].manifest
    assert manifest.stages == (Stage.DETECT,)
    assert manifest.scope is Scope.STREAM
    assert manifest.requires == (EXTENT,)
    assert manifest.cost is Cost.HEAVY


def test_calibrates_with_rising_progress_then_is_ok() -> None:
    outputs = first_run()
    calibrating = [o.status for o in outputs[:CALIBRATION_STEPS]]
    assert {s.state for s in calibrating} == {State.CALIBRATING}
    progress = [s.progress for s in calibrating]
    assert progress == sorted(progress)
    assert 0 < progress[0] < progress[-1] <= 1
    assert outputs[CALIBRATION_STEPS].status.state is State.OK
    assert all(o.items == () for o in outputs[:BLOB_FROM])


def test_detection_becomes_a_candidate_with_features() -> None:
    (candidate,) = first_run()[-1].items
    x0, y0, x1, y1 = candidate.region.box
    assert 0 <= x0 < x1 <= EXTENT_PX[0]
    assert 0 <= y0 < y1 <= EXTENT_PX[1]
    assert candidate.stream == STREAM
    assert candidate.region.velocity_px_s is not None
    assert candidate.score > 0
    assert set(candidate.features) == {
        "z_score",
        "best_psr",
        "event_count",
        "bg_class",
    }


def test_empty_windows_give_no_candidates_and_do_not_raise() -> None:
    detector = build()
    for i in range(CALIBRATION_STEPS + 2):
        window = dtc.window(
            STREAM, dtc.empty_events(), i * STEP_US, (i + 1) * STEP_US, None
        )
        output = detector.step(dtc.step_of(i, STEP_US, window), ())
        assert output.items == ()


def test_reset_returns_to_the_fresh_state() -> None:
    detector = build()
    first = drive(detector, 12)
    detector.reset()
    assert drive(detector, 12) == first


def test_grid_extent_comes_from_the_fact_unless_overridden() -> None:
    grid = build().pipeline.grid
    assert (grid.width, grid.height, grid.window_us) == (*EXTENT_PX, STEP_US)
    override = SimplifiedConfig(width=32, height=24, rows=6, cols=8)
    assert build(override).pipeline.grid.width == 32


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ConfigError):
        parse_config(SimplifiedConfig, {"z_treshold": 3})


def test_deployed_defaults_mirror_the_flat_factory_keys() -> None:
    config = SimplifiedConfig()
    assert (config.rows, config.cols, config.z_threshold) == (72, 128, 10)
    assert config.calibration_time_us == 1_000_000
    assert config.coactivity_weight == 100


def test_passes_the_conformance_kit() -> None:
    config = {"calibration_time_us": 30_000, "calibration_windows": 2}
    assert (
        check_method("simplified", facts={EXTENT: (128, 96)}, config=config)
        == []
    )
