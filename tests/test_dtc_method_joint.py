"""The joint method finds cross-stream coincidence on any stream naming."""

from __future__ import annotations

import pytest
from fakes.dtc_joint import Joint

from evio.dtc.errors import ConfigError
from evio.dtc.protocols import Context
from evio.dtc.registry import create
from evio.dtc.types import State, Step
from tests.support.dtc import blob_events, empty_events, step_of, window

pytestmark = pytest.mark.t1

STEP_US = 10_000
NAMES = ("mast/north", "7", "ΔΩ")


def make(streams: tuple[str, ...] = NAMES, **config: object) -> Joint:
    context = Context(None, streams, {}, STEP_US)
    joint = create("joint", config, context)
    assert isinstance(joint, Joint)
    return joint


def blobs(at: dict[str, int | None], cy: int = 100) -> Step:
    """One step with a blob at x per stream; None leaves the stream empty."""
    return step_of(
        0,
        STEP_US,
        *(
            window(
                name,
                empty_events()
                if cx is None
                else blob_events(0, STEP_US, cx, cy, radius=2),
                0,
                STEP_US,
            )
            for name, cx in at.items()
        ),
    )


def streams_of(step: Step, joint: Joint) -> list[str]:
    return [c.stream for c in joint.step(step, ()).items]


def test_manifest_is_joint_detect() -> None:
    assert make().manifest.scope == "joint"


def test_coincidence_keeps_each_streams_key() -> None:
    joint = make(min_streams=2)
    step = blobs({"mast/north": 200, "7": 200, "ΔΩ": None})
    assert sorted(streams_of(step, joint)) == ["7", "mast/north"]


def test_single_stream_activity_is_not_enough() -> None:
    step = blobs({"mast/north": 200, "7": 600, "ΔΩ": None})
    assert streams_of(step, make(min_streams=2)) == []


def test_offset_aligns_positions_across_streams() -> None:
    step = blobs({"mast/north": 200, "7": 100})
    plain = make(streams=("mast/north", "7"))
    shifted = make(streams=("mast/north", "7"), x_offsets={"7": 100})
    assert streams_of(step, plain) == []
    candidates = shifted.step(step, ()).items
    assert {c.stream for c in candidates} == {"mast/north", "7"}
    seven = next(c for c in candidates if c.stream == "7")
    assert seven.region.center[0] == pytest.approx(100, abs=16)


def test_works_with_one_stream_and_with_many() -> None:
    solo = make(streams=("only",), min_streams=1)
    assert streams_of(blobs({"only": 50}), solo) == ["only"]
    many = tuple(f"cam{i}" for i in range(9))
    joint = make(streams=many, min_streams=9)
    assert len(streams_of(blobs(dict.fromkeys(many, 50)), joint)) == 9


def test_missing_stream_degrades_but_still_detects() -> None:
    joint = make(min_streams=2)
    step = blobs({"mast/north": 200, "7": 200})
    result = joint.step(step, ())
    assert result.status.state is State.DEGRADED
    assert "ΔΩ" in result.status.detail
    assert len(result.items) == 2


def test_full_step_is_ok_and_score_is_stream_fraction() -> None:
    result = make().step(blobs(dict.fromkeys(NAMES, 200)), ())
    assert result.status.state is State.OK
    assert {c.score for c in result.items} == {1.0}


def test_empty_windows_yield_no_candidates_and_stay_ok() -> None:
    result = make().step(blobs(dict.fromkeys(NAMES, None)), ())
    assert result.items == ()
    assert result.status.state is State.OK


def test_reset_returns_to_fresh_behaviour() -> None:
    joint = make()
    step = blobs({"mast/north": 200, "7": 200})
    before = joint.step(step, ())
    joint.reset()
    assert joint.step(step, ()) == before


def test_input_windows_are_not_mutated() -> None:
    step = blobs({"mast/north": 200, "7": 200})
    before = {k: w.events.x_coords.copy() for k, w in step.windows.items()}
    make().step(step, ())
    for key, window_ in step.windows.items():
        assert (window_.events.x_coords == before[key]).all()


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ConfigError):
        make(cell=3)
