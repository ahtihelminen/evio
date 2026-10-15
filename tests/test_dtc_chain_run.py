"""The chain runs stages in order and owns priors, filtering, cadence."""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import pytest
from fakes.dtc_timer import FakeTimer
from fakes.dtc_toys import LOG, registered

from evio.dtc.chain import build, worst
from evio.dtc.errors import ConfigError
from evio.dtc.facts import EXTENT
from evio.dtc.result import Pipeline, StepResult
from evio.dtc.spec import CandidateFilter
from evio.dtc.types import OK, State, Status
from tests.support.dtc_chain import STEP_US, STREAMS, spec_of, step_at

pytestmark = pytest.mark.t1

importlib.import_module("evio.dtc.methods")
BOTH = STREAMS
RATE_EXCESS = ("rate-excess", {"warmup_steps": 2})


@pytest.fixture(autouse=True)
def toys() -> Iterator[None]:
    with registered():
        yield


def make(
    *methods: str | tuple[str, dict],
    streams: tuple[str, ...] = STREAMS,
    candidate_filter: CandidateFilter | None = None,
) -> Pipeline:
    return build(
        spec_of(*methods, candidate_filter=candidate_filter),
        {EXTENT: (1280, 720)},
        streams=streams,
        step_us=STEP_US,
        timer=FakeTimer(),
    )


def run(pipeline: Pipeline, count: int, *present: str) -> list[StepResult]:
    return [pipeline.step(step_at(i, *present)) for i in range(count)]


def streams_of(items: tuple) -> list[str]:
    return [item.stream for item in items]


def test_a_detector_only_chain_emits_candidates_and_nothing_later() -> None:
    result = run(make("toy-detector"), 1, *BOTH)[0]
    assert streams_of(result.candidates) == ["cam0", "cam1"]
    assert (result.tracks, result.verdicts, result.alarms) == ((), (), ())
    assert result.live_ids == frozenset()


def test_a_stream_without_events_yields_nothing_and_stays_ok() -> None:
    result = run(make("toy-detector"), 1, "cam1")[0]
    assert streams_of(result.candidates) == ["cam1"]
    assert result.status == {"cam0": OK, "cam1": OK}


def test_a_tracker_sees_only_its_own_streams_candidates() -> None:
    result = run(make("toy-detector", "toy-tracker"), 1, *BOTH)[0]
    assert [t.id for t in result.tracks] == ["cam0:0", "cam1:0"]
    assert result.live_ids == {"cam0:0", "cam1:0"}


def test_tracker_priors_reach_the_detector_of_the_same_stream_next_step() -> (
    None
):
    pipeline = make("toy-detector", "toy-tracker")
    first = pipeline.step(step_at(0, *BOTH))
    pipeline.step(step_at(1, *BOTH))
    regions = {c.stream: c.region for c in first.candidates}
    assert LOG["priors"] == [
        ("cam0", ()),
        ("cam1", ()),
        ("cam0", (regions["cam0"],)),
        ("cam1", (regions["cam1"],)),
    ]


def test_a_chain_with_a_real_detector_runs_end_to_end() -> None:
    pipeline = make(
        RATE_EXCESS,
        "toy-tracker",
        "toy-classifier-every-step",
        "toy-decision",
        streams=("cam0",),
    )
    results = [pipeline.step(step_at(i, streams=("cam0",))) for i in range(2)]
    assert [r.status["cam0"].state for r in results] == [State.CALIBRATING] * 2
    found = pipeline.step(step_at(2, "cam0", streams=("cam0",)))
    assert found.candidates
    assert found.tracks
    assert [v.age_steps for v in found.verdicts] == [0] * len(found.tracks)
    assert found.alarms


def test_a_fused_methods_candidates_are_not_filtered_by_the_chain() -> None:
    config = {"boxes": 5, "spread_px": 20}
    pipeline = make(("toy-detect-track", config), streams=("cam0",))
    kept = CandidateFilter(top_k=2)
    filtered = make(
        ("toy-detector", config), streams=("cam0",), candidate_filter=kept
    )
    step = step_at(0, "cam0", streams=("cam0",))
    assert len(pipeline.step(step).candidates) == 5
    assert len(filtered.step(step).candidates) == 2


def test_the_chain_filters_candidates_once_after_the_detector() -> None:
    pipeline = make(
        ("toy-detector", {"boxes": 5}),
        "toy-tracker",
        candidate_filter=CandidateFilter(top_k=2, iou=None),
    )
    result = run(pipeline, 1, *BOTH)[0]
    assert streams_of(result.candidates) == ["cam0", "cam0", "cam1", "cam1"]
    assert len(result.tracks) == 4


def test_overlapping_candidates_are_deduped_by_the_chain() -> None:
    pipeline = make(
        ("toy-detector", {"boxes": 3, "spread_px": 0}),
        candidate_filter=CandidateFilter(top_k=None, iou=0.3),
    )
    assert len(run(pipeline, 1, *BOTH)[0].candidates) == 2


def test_a_joint_method_gets_every_stream_and_its_output_splits_back() -> None:
    pipeline = make(("joint", {"min_events": 20}), "toy-tracker")
    result = run(pipeline, 1, *BOTH)[0]
    assert set(streams_of(result.candidates)) == {"cam0", "cam1"}
    assert {t.stream for t in result.tracks} == {"cam0", "cam1"}
    assert set(pipeline.stats) == {"joint", "toy-tracker"}


def test_a_joint_detector_gets_each_joint_prior_once() -> None:
    pipeline = make("toy-joint-detector", "toy-joint-tracker")
    first = pipeline.step(step_at(0, *BOTH))
    pipeline.step(step_at(1, *BOTH))
    assert LOG["joint-priors"] == [
        (),
        tuple(track.region for track in first.tracks),
    ]


def test_a_joint_method_with_a_missing_stream_degrades_that_step() -> None:
    pipeline = make(("joint", {"min_events": 20}))
    only_cam0 = step_at(0, *BOTH).only("cam0")
    result = pipeline.step(only_cam0)
    assert result.status["cam0"].state is State.DEGRADED
    assert "missing" in result.status["cam0"].detail


def test_verdicts_age_between_due_steps_of_a_cadence_classifier() -> None:
    pipeline = make("toy-detector", "toy-tracker", "toy-classifier")
    results = run(pipeline, 4, *BOTH)
    assert [r.verdicts[0].age_steps for r in results] == [0, 1, 0, 1]
    assert LOG["classifier"] == [
        ("cam0", 0),
        ("cam1", 0),
        ("cam0", 2),
        ("cam1", 2),
    ]
    assert len(results[1].verdicts) == 2


def test_aged_verdicts_of_a_dead_track_are_dropped() -> None:
    pipeline = make("toy-detector", "toy-tracker", "toy-classifier")
    pipeline.step(step_at(0, *BOTH))
    assert pipeline.step(step_at(1, "cam1")).verdicts[0].stream == "cam1"
    assert pipeline.step(step_at(2)).verdicts == ()


def test_a_decision_stage_runs_every_step_on_aged_verdicts() -> None:
    pipeline = make(
        "toy-detector", "toy-tracker", "toy-classifier", "toy-decision"
    )
    results = run(pipeline, 2, "cam0")
    assert [len(r.alarms) for r in results] == [1, 1]


def test_a_batch_classifier_is_called_once_for_all_due_streams() -> None:
    pipeline = make("toy-detector", "toy-tracker", "toy-batch-classifier")
    results = run(pipeline, 2, *BOTH)
    assert LOG["batch"] == [2, 2]
    assert [v.probability for v in results[0].verdicts] == [0.5, 0.5]


def test_evidence_is_computed_once_per_stream_per_step_and_shared() -> None:
    pipeline = make(
        "toy-detector",
        "toy-tracker",
        "toy-classifier-every-step",
        "toy-decision",
    )
    result = run(pipeline, 2, *BOTH)[1]
    assert LOG["provider"] == [
        ("cam0", 0),
        ("cam1", 0),
        ("cam0", 1),
        ("cam1", 1),
    ]
    assert [v.probability for v in result.verdicts] == [2.0, 2.0]
    assert len(result.alarms) == 2


def test_evidence_follows_every_step_even_when_the_classifier_skips() -> None:
    pipeline = make("toy-detector", "toy-tracker", "toy-classifier")
    run(pipeline, 3, "cam0")
    computed = [index for stream, index in LOG["provider"] if stream == "cam0"]
    assert computed == [0, 1, 2]
    assert [i for _, i in LOG["classifier"]] == [0, 0, 2, 2]


def test_steps_must_arrive_with_increasing_index() -> None:
    pipeline = make("toy-detector")
    pipeline.step(step_at(3, *BOTH))
    for index in (3, 2):
        with pytest.raises(ConfigError, match="does not follow 3"):
            pipeline.step(step_at(index, *BOTH))


def test_a_window_of_an_unbuilt_stream_is_a_config_error() -> None:
    pipeline = make("toy-detector", streams=("cam0",))
    with pytest.raises(ConfigError, match="cam1"):
        pipeline.step(step_at(0, *BOTH))


def test_reset_gives_the_same_results_as_a_fresh_run() -> None:
    pipeline = make("toy-detector", "toy-tracker", "toy-classifier")
    first = run(pipeline, 5, *BOTH)
    pipeline.reset()
    assert pipeline.stats["toy-detector"].count == 0
    assert run(pipeline, 5, *BOTH) == first


def test_reset_accepts_step_zero_again_and_forgets_priors() -> None:
    pipeline = make("toy-detector", "toy-tracker")
    run(pipeline, 2, *BOTH)
    pipeline.reset()
    LOG["priors"].clear()
    pipeline.step(step_at(0, *BOTH))
    assert LOG["priors"] == [("cam0", ()), ("cam1", ())]


def test_the_worst_status_wins_and_defaults_to_ok() -> None:
    calibrating = Status(State.CALIBRATING, 0.5)
    degraded = Status(State.DEGRADED, detail="x")
    assert worst([OK, calibrating, degraded]) is degraded
    assert worst([OK, calibrating]) is calibrating
    assert worst([]) is OK
